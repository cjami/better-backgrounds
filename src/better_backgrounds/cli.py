"""Developer, diagnostic, and worker commands for Better Backgrounds."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast
from uuid import uuid4

import cv2
import typer
from platformdirs import user_cache_path, user_data_path

from better_backgrounds.jobs.fake_worker import FakeOutcome, run_fake_job
from better_backgrounds.matting.benchmark import load_video_frames, run_matting_benchmark
from better_backgrounds.matting.contracts import MattingConfig
from better_backgrounds.matting.engine import (
    CompletedMatte,
    EngineFailure,
    EngineReady,
    ProcessMattingEngine,
)
from better_backgrounds.matting.runtime import packaged_checkpoint_path
from better_backgrounds.reconstruction.sharp import (
    SHARP_BUILDER_REVISION,
    SharpCheckpointInstaller,
    probe_sharp_capabilities,
)
from better_backgrounds.reconstruction.sharp.worker import (
    SharpBuildWorker,
    SharpCheckpointWorker,
    watch_control,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from better_backgrounds.matting.runtime import DeviceRequest
    from better_backgrounds.reconstruction.sharp.runtime import SharpDeviceRequest

app = typer.Typer(
    name="better-backgrounds",
    help="Run Better Backgrounds desktop and local model commands.",
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


def _sharp_device(value: str) -> SharpDeviceRequest:
    if value not in {"auto", "cuda", "mps", "cpu"}:
        msg = "Choose auto, cuda, mps, or cpu."
        raise typer.BadParameter(msg, param_hint="--device")
    return cast("SharpDeviceRequest", value)


@app.command("doctor")
def doctor_command(
    device: Annotated[str, typer.Option(help="SHARP device to probe.")] = "auto",
) -> None:
    """Report local MatAnyone and SHARP readiness without changing caches."""
    cache_root, _data_root = _application_roots()
    installer = SharpCheckpointInstaller(cache_root / "models-v1" / "sharp")
    try:
        capabilities = probe_sharp_capabilities(_sharp_device(device))
        sharp_runtime: dict[str, object] = {
            "available": True,
            "device_type": capabilities.device_type,
            "accelerated": capabilities.accelerated,
        }
    except RuntimeError as error:
        sharp_runtime = {"available": False, "error": str(error)}
    typer.echo(
        json.dumps(
            {
                "schema_version": 2,
                "matanyone_checkpoint_ready": packaged_checkpoint_path().is_file(),
                "sharp": {
                    **sharp_runtime,
                    "builder_revision": SHARP_BUILDER_REVISION,
                    "checkpoint_ready": installer.is_ready(),
                    "checkpoint_path": str(installer.checkpoint_path),
                    "license": installer.manifest.license_name,
                },
            },
            indent=2,
        )
    )


@app.command("prepare-sharp")
def prepare_sharp_command(
    accept_model_license: Annotated[
        bool,
        typer.Option(
            "--accept-model-license",
            help="Accept Apple's research-only model license before download.",
        ),
    ] = False,
    job_id: Annotated[str | None, typer.Option(help="Stable worker job identifier.")] = None,
    model_root: Annotated[
        Path | None,
        typer.Option(file_okay=False, resolve_path=True, help="Override the managed model cache."),
    ] = None,
) -> None:
    """Download and SHA-256 verify the pinned research-only SHARP checkpoint."""
    cache_root, _data_root = _application_roots()
    worker = SharpCheckpointWorker(
        job_id=job_id or uuid4().hex,
        model_root=model_root or cache_root / "models-v1" / "sharp",
        license_accepted=accept_model_license,
    )
    watch_control(worker.job_id, worker.cancel)
    raise typer.Exit(worker.run())


@app.command("sharp-build")
def sharp_build_command(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    job_id: Annotated[str | None, typer.Option(help="Stable worker job identifier.")] = None,
    device: Annotated[str, typer.Option(help="SHARP device: auto, cuda, mps, or cpu.")] = "auto",
    source_kind: Annotated[str, typer.Option(help="Image source: upload or camera.")] = "upload",
    checkpoint: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    scene_cache: Annotated[
        Path | None,
        typer.Option(file_okay=False, resolve_path=True),
    ] = None,
    catalogue: Annotated[
        Path | None,
        typer.Option(dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Build one upload-first SHARP PLY through the versioned worker boundary."""
    if source_kind not in {"upload", "camera"}:
        msg = "Choose upload or camera."
        raise typer.BadParameter(msg, param_hint="--source-kind")
    cache_root, data_root = _application_roots()
    installer = SharpCheckpointInstaller(cache_root / "models-v1" / "sharp")
    actual_job_id = job_id or uuid4().hex
    worker = SharpBuildWorker(
        job_id=actual_job_id,
        image=image,
        source_kind=source_kind,
        device=_sharp_device(device),
        checkpoint_path=checkpoint or installer.checkpoint_path,
        scene_cache_root=scene_cache or cache_root / "scenes-v1",
        catalogue_path=catalogue or data_root / "scene-catalogue-v1.json",
    )
    watch_control(actual_job_id, worker.cancel)
    raise typer.Exit(worker.run())


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


@app.command("fake-job", hidden=True)
def fake_job_command(
    job_id: Annotated[str, typer.Option(help="Stable job identifier.")],
    outcome: Annotated[FakeOutcome, typer.Option()] = "success",
    delay: Annotated[float, typer.Option(min=0.0, max=5.0)] = 0.08,
) -> None:
    """Run the deterministic desktop smoke worker."""
    raise typer.Exit(run_fake_job(job_id, outcome=outcome, delay=delay))


if __name__ == "__main__":
    app()
