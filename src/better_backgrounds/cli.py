"""Developer and worker commands for Better Backgrounds."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast
from uuid import uuid4

import cv2
import typer
from platformdirs import user_cache_path, user_data_path

from better_backgrounds.fake_worker import FakeOutcome, run_fake_job
from better_backgrounds.live_matting import MattingConfig
from better_backgrounds.managed_tools import (
    SampleInstaller,
    ToolInstaller,
    ToolManifest,
    artifact_filename,
    diagnose_environment,
    load_tool_manifest,
    platform_key,
    project_executable_paths,
    resolved_executable_paths,
)
from better_backgrounds.matanyone_runtime import packaged_checkpoint_path
from better_backgrounds.matting_benchmark import load_video_frames, run_matting_benchmark
from better_backgrounds.matting_engine import (
    CompletedMatte,
    EngineFailure,
    EngineReady,
    ProcessMattingEngine,
)
from better_backgrounds.protocol import ErrorEvent
from better_backgrounds.pycolmap_worker import worker_command as pycolmap_worker_command
from better_backgrounds.reconstruction import ReconstructionCommands, ReconstructionQuality
from better_backgrounds.reconstruction_worker import (
    ReconstructionWorker,
    emit_stdout,
    watch_control,
)
from better_backgrounds.video_analysis import analyse_video_file

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from better_backgrounds.matanyone_runtime import DeviceRequest

app = typer.Typer(
    name="better-backgrounds",
    help="Run Better Backgrounds desktop and diagnostic commands.",
    no_args_is_help=True,
)


@app.command("desktop")
def desktop_command() -> None:
    """Open the Python-owned desktop application."""
    from better_backgrounds.desktop.app import main  # noqa: PLC0415

    raise typer.Exit(main())


def _application_roots() -> tuple[Path, Path]:
    cache = Path(user_cache_path("Better Backgrounds", "Better Backgrounds"))
    data = Path(user_data_path("Better Backgrounds", "Better Backgrounds"))
    return cache, data


def _required_tool(path: Path | None) -> Path:
    if path is None:
        msg = "required tool path was not resolved"
        raise RuntimeError(msg)
    return path


def _install_reviewed_tools(
    cache_root: Path,
    manifest: ToolManifest,
    artifact_dir: Path | None,
) -> tuple[list[str], list[str], list[str]]:
    installed: list[str] = []
    available_from_project: list[str] = []
    unavailable: list[str] = []
    installer = ToolInstaller(cache_root / "tools-v1")
    project_tools = project_executable_paths(manifest=manifest)
    for tool in manifest.tools:
        artifact = next(
            (item for item in tool.artifacts if item.platform == platform_key()),
            None,
        )
        label = f"{tool.tool_id} {tool.version}"
        if artifact is None:
            (available_from_project if tool.tool_id in project_tools else unavailable).append(label)
            continue
        if artifact_dir is None:
            installer.install(tool.tool_id, tool.version, artifact)
        else:
            archive = artifact_dir / artifact_filename(artifact)
            if not archive.is_file():
                msg = f"Missing offline artifact: {archive.name}"
                raise typer.BadParameter(msg, param_hint="--artifact-dir")
            installer.install_archive(tool.tool_id, tool.version, artifact, archive)
        installed.append(label)
    return installed, available_from_project, unavailable


@app.command("setup")
def setup_command(
    tools: Annotated[bool, typer.Option("--tools", help="Install pinned native tools.")] = False,
    samples: Annotated[
        bool, typer.Option("--samples", help="Install prepared sample inputs.")
    ] = False,
    all_resources: Annotated[
        bool,
        typer.Option("--all", help="Install tools and samples."),
    ] = False,
    artifact_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifact-dir",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Install exact manifest archives transferred into this directory.",
        ),
    ] = None,
) -> None:
    """Install reviewed resources into application-managed directories."""
    if not (tools or samples or all_resources):
        msg = "Choose --tools, --samples, or --all."
        raise typer.BadParameter(msg)
    cache_root, _data_root = _application_roots()
    manifest = load_tool_manifest()
    installed: list[str] = []
    available_from_project: list[str] = []
    unavailable: list[str] = []
    if tools or all_resources:
        installed, available_from_project, unavailable = _install_reviewed_tools(
            cache_root,
            manifest,
            artifact_dir,
        )
    if samples or all_resources:
        sample_installer = SampleInstaller(cache_root / "samples-v1")
        for sample in manifest.samples:
            sample_installer.install(sample)
            installed.append(f"sample:{sample.sample_id} {sample.version}")
        if not manifest.samples:
            unavailable.append("prepared sample video")
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "installed": installed,
                "available_from_project": available_from_project,
                "unavailable_for_platform": unavailable,
                "platform": platform_key(),
            },
            indent=2,
        )
    )


@app.command("doctor")
def doctor_command() -> None:
    """Report reconstruction and sample-mode support without changing jobs."""
    cache_root, _data_root = _application_roots()
    executables = resolved_executable_paths(cache_root / "tools-v1")
    report = diagnose_environment(executables, storage_root=cache_root)
    typer.echo(report.model_dump_json(indent=2))


@app.command("analyse")
def analyse_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    ffprobe: Annotated[
        Path | None,
        typer.Option(help="Explicit reviewed ffprobe executable."),
    ] = None,
) -> None:
    """Analyze video constraints and capture-quality evidence."""
    cache_root, _data_root = _application_roots()
    executable = ffprobe or resolved_executable_paths(cache_root / "tools-v1").get("ffprobe")
    if executable is None:
        typer.echo("ffprobe is not installed; run `better-backgrounds setup --tools`.", err=True)
        raise typer.Exit(2)
    try:
        diagnostics = analyse_video_file(video, executable)
    except (TypeError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(3) from error
    typer.echo(diagnostics.model_dump_json(indent=2))
    if not diagnostics.suitable:
        raise typer.Exit(4)


@app.command("matting-benchmark")
def matting_benchmark_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    mask: Annotated[Path, typer.Option(exists=True, dir_okay=False, resolve_path=True)],
    frame_limit: Annotated[
        int,
        typer.Option(min=2, max=300, help="Maximum source frames used at each resolution."),
    ] = 60,
    device: Annotated[
        str,
        typer.Option(help="MatAnyone 2 device: auto, cuda, mps, or cpu."),
    ] = "auto",
) -> None:
    """Benchmark the pinned stateful MatAnyone 2 step API and print its gate report."""
    if device not in {"auto", "cuda", "mps", "cpu"}:
        msg = "Choose auto, cuda, mps, or cpu."
        raise typer.BadParameter(msg, param_hint="--device")
    seed_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if seed_mask is None:
        msg = "Unable to decode the seed mask."
        raise typer.BadParameter(msg, param_hint="--mask")
    seed_mask = cast("NDArray[np.uint8]", seed_mask)
    frames = load_video_frames(video, frame_limit=frame_limit)
    if seed_mask.shape != frames[0].shape[:2]:
        seed_mask = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                seed_mask,
                (frames[0].shape[1], frames[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
    report = run_matting_benchmark(
        frames,
        seed_mask,
        packaged_checkpoint_path(),
        requested_device=cast("DeviceRequest", device),
    )
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(5)


@app.command("matting-worker-smoke", hidden=True)
def matting_worker_smoke_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    mask: Annotated[Path, typer.Option(exists=True, dir_okay=False, resolve_path=True)],
) -> None:
    """Exercise the spawned worker and shared ring with a short decoded sequence."""
    frames = load_video_frames(video, frame_limit=8)
    seed_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if seed_mask is None:
        msg = "Unable to decode the seed mask."
        raise typer.BadParameter(msg, param_hint="--mask")
    seed_mask = cast("NDArray[np.uint8]", seed_mask)
    if seed_mask.shape != frames[0].shape[:2]:
        seed_mask = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                seed_mask,
                (frames[0].shape[1], frames[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
    engine = ProcessMattingEngine(packaged_checkpoint_path())
    completed = 0
    ready = False
    started = time.monotonic()
    try:
        engine.start(
            frames[0],
            seed_mask,
            MattingConfig(internal_size=360, warmup_iterations=2),
        )
        deadline = time.monotonic() + 30.0
        next_frame = 1
        while time.monotonic() < deadline and completed + engine.dropped_frames < len(frames) - 1:
            for event in engine.poll():
                if isinstance(event, EngineReady):
                    ready = True
                elif isinstance(event, CompletedMatte):
                    completed += 1
                elif isinstance(event, EngineFailure):
                    typer.echo(event.message, err=True)
                    raise typer.Exit(6)
            if (
                ready
                and next_frame < len(frames)
                and engine.submit(frames[next_frame], captured_at=time.monotonic() * 1000.0)
            ):
                next_frame += 1
            time.sleep(0.005)
    finally:
        engine.close()
    typer.echo(
        json.dumps(
            {
                "ready": ready,
                "completed": completed,
                "dropped": engine.dropped_frames,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
            },
            indent=2,
        ),
    )
    if not ready or completed == 0:
        raise typer.Exit(6)


@app.command("reconstruct")
def reconstruct_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    job_id: Annotated[str | None, typer.Option(help="Stable job identifier.")] = None,
    resume: Annotated[bool, typer.Option(help="Reuse matching verified stages.")] = False,
    quality: Annotated[
        ReconstructionQuality,
        typer.Option(help="Bound frame resolution, frame count, and Brush training work."),
    ] = ReconstructionQuality.BALANCED,
    ffprobe: Annotated[Path | None, typer.Option(help="Explicit ffprobe executable.")] = None,
    ffmpeg: Annotated[Path | None, typer.Option(help="Explicit FFmpeg executable.")] = None,
    brush: Annotated[Path | None, typer.Option(help="Explicit Brush executable.")] = None,
    splat_transform: Annotated[
        Path | None,
        typer.Option("--splat-transform", help="Explicit pinned SplatTransform executable."),
    ] = None,
) -> None:
    """Reconstruct one video through a cancellable versioned worker."""
    cache_root, data_root = _application_roots()
    managed = resolved_executable_paths(cache_root / "tools-v1")
    selected = {
        "ffprobe": ffprobe or managed.get("ffprobe"),
        "ffmpeg": ffmpeg or managed.get("ffmpeg"),
        "brush": brush or managed.get("brush"),
        "splat-transform": splat_transform or managed.get("splat-transform"),
    }
    missing = [name for name, path in selected.items() if path is None]
    actual_job_id = job_id or uuid4().hex
    if missing:
        emit_stdout(
            ErrorEvent(
                job_id=actual_job_id,
                code="tools_unavailable",
                message=f"Missing managed tools: {', '.join(missing)}.",
                recovery_action="Run setup --tools and doctor before reconstruction.",
            )
        )
        raise typer.Exit(2)
    worker = ReconstructionWorker(
        job_id=actual_job_id,
        video=video,
        job_root=data_root / "jobs-v1" / actual_job_id,
        scene_cache_root=cache_root / "scenes-v1",
        catalogue_path=data_root / "scene-catalogue-v1.json",
        ffprobe=_required_tool(selected["ffprobe"]),
        commands=ReconstructionCommands(
            ffmpeg=_required_tool(selected["ffmpeg"]),
            pycolmap=pycolmap_worker_command(),
            brush=_required_tool(selected["brush"]),
            splat_transform=_required_tool(selected["splat-transform"]),
        ),
        emit=emit_stdout,
        resume=resume,
        quality=quality,
    )
    watch_control(worker)
    raise typer.Exit(worker.run())


@app.command("fake-job", hidden=True)
def fake_job_command(
    job_id: Annotated[str, typer.Option(help="Stable job identifier.")],
    outcome: Annotated[FakeOutcome, typer.Option()] = "success",
    delay: Annotated[float, typer.Option(min=0.0, max=5.0)] = 0.08,
) -> None:
    """Run the deterministic Phase 2 protocol worker."""
    raise typer.Exit(run_fake_job(job_id, outcome=outcome, delay=delay))


if __name__ == "__main__":
    app()
