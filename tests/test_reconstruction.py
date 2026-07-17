"""Resumable reconstruction orchestration tests."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from better_backgrounds.protocol import WarningEvent
from better_backgrounds.reconstruction import (
    JobManifest,
    ReconstructionCommands,
    ReconstructionQuality,
    StageArtifact,
    analyse_colmap_model,
    fingerprint_file,
    reconstruction_preset,
    select_largest_model,
)
from better_backgrounds.reconstruction_worker import (
    NativeCommandError,
    NativeProcessRunner,
    ReconstructionCancelledError,
    ReconstructionWorker,
    build_generated_scene_reference,
)
from better_backgrounds.scene import colmap_scene_transform

if TYPE_CHECKING:
    from collections.abc import Sequence

SHORT_TIMEOUT = 0.05
EXPECTED_LONGEST_GAP = 10


@pytest.mark.parametrize(
    ("quality", "selected_frames", "maximum_image_edge", "brush_steps"),
    [
        (ReconstructionQuality.PREVIEW, 60, 1280, 3_000),
        (ReconstructionQuality.BALANCED, 80, 1600, 6_000),
        (ReconstructionQuality.QUALITY, 100, 1920, 12_000),
    ],
)
def test_reconstruction_quality_presets_bound_expensive_work(
    quality: ReconstructionQuality,
    selected_frames: int,
    maximum_image_edge: int,
    brush_steps: int,
) -> None:
    """Keep speed and quality choices explicit and deterministic."""
    preset = reconstruction_preset(quality)

    assert preset.selected_frames == selected_frames
    assert preset.maximum_image_edge == maximum_image_edge
    assert preset.brush_steps == brush_steps


def test_cache_hit_requires_input_and_configuration_fingerprints(tmp_path: Path) -> None:
    """Changed source or settings must not reuse completed work."""
    source = tmp_path / "room.mp4"
    source.write_bytes(b"video")
    manifest = JobManifest(
        job_id="job-1",
        input_path=source,
        input_fingerprint=fingerprint_file(source),
        configuration_fingerprint="config-a",
        stages={
            "extract_frames": StageArtifact(
                fingerprint="stage-a",
                outputs=("frames/0001.png",),
            ),
        },
    )
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "0001.png").write_bytes(b"frame")

    assert manifest.can_resume(source, "config-a", tmp_path)
    assert not manifest.can_resume(source, "config-b", tmp_path)
    source.write_bytes(b"changed")
    assert not manifest.can_resume(source, "config-a", tmp_path)


def test_generated_scene_records_colmap_coordinate_normalization(tmp_path: Path) -> None:
    """Keep every locally trained room upright in the PlayCanvas renderer."""
    root = tmp_path / "scene"
    root.mkdir()
    (root / "meta.json").write_text('{"version":2,"count":1}', encoding="utf-8")

    reference = build_generated_scene_reference(
        "generated-room-v1",
        Path("room.mp4"),
        root,
    )

    assert reference.default_viewpoint.scene_transform == colmap_scene_transform()


def test_commands_preserve_spaces_and_unicode_as_single_arguments(tmp_path: Path) -> None:
    """Use argv construction rather than shell interpolation."""
    commands = ReconstructionCommands(
        ffmpeg=Path("C:/Program Files/ffmpeg.exe"),
        pycolmap=(sys.executable, "-m", "better_backgrounds.pycolmap_worker"),
        brush=Path("C:/Tools/brush.exe"),
        splat_transform=Path("C:/Program Files/node/splat-transform.cmd"),
    )
    video = tmp_path / "部屋 video.mp4"

    command = commands.extract_frames(
        video,
        tmp_path / "selected frames",
        fps=3.0,
        maximum_image_edge=1280,
    )

    assert Path(command[0]) == Path("C:/Program Files/ffmpeg.exe")
    assert str(video) in command
    assert str(tmp_path / "selected frames" / "%06d.png") in command
    video_filter = command[command.index("-vf") + 1]
    assert "min(iw,1280)" in video_filter
    assert "min(ih,1280)" in video_filter


def test_brush_command_uses_the_pinned_cli_training_contract(tmp_path: Path) -> None:
    """Keep bounded training compatible with Brush 0.3.0."""
    commands = ReconstructionCommands(
        ffmpeg=Path("ffmpeg"),
        pycolmap=(sys.executable, "-m", "better_backgrounds.pycolmap_worker"),
        brush=Path("brush"),
        splat_transform=Path("splat-transform"),
    )

    command = commands.brush_training(
        tmp_path / "dataset",
        tmp_path / "output/room.ply",
        iterations=12_000,
    )

    assert command[command.index("--total-steps") + 1] == "12000"
    assert command[command.index("--export-name") + 1] == "room.ply"
    assert "--total-train-iters" not in command


def test_splat_transform_uses_gpu_autodetection_with_explicit_cpu_fallback(
    tmp_path: Path,
) -> None:
    """Prefer WebGPU without hard-coding an adapter and retain a portable fallback."""
    commands = ReconstructionCommands(
        ffmpeg=Path("ffmpeg"),
        pycolmap=(sys.executable, "-m", "better_backgrounds.pycolmap_worker"),
        brush=Path("brush"),
        splat_transform=Path("splat-transform"),
    )

    automatic = commands.convert_sog(tmp_path / "room.ply", tmp_path / "scene.sog")
    cpu = commands.convert_sog(
        tmp_path / "room.ply",
        tmp_path / "scene.sog",
        gpu="cpu",
    )

    assert "-g" not in automatic
    assert cpu[cpu.index("-g") + 1] == "cpu"


def test_sog_conversion_retries_on_cpu_after_gpu_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep WebGPU initialization failures recoverable inside one job."""
    commands = ReconstructionCommands(
        ffmpeg=Path("ffmpeg"),
        pycolmap=(sys.executable, "-m", "better_backgrounds.pycolmap_worker"),
        brush=Path("brush"),
        splat_transform=Path("splat-transform"),
    )
    events: list[object] = []
    worker = ReconstructionWorker(
        job_id="job-1",
        video=tmp_path / "room.mp4",
        job_root=tmp_path / "job",
        scene_cache_root=tmp_path / "scenes",
        catalogue_path=tmp_path / "catalogue.json",
        ffprobe=Path("ffprobe"),
        commands=commands,
        emit=events.append,
    )
    calls: list[list[str]] = []

    def run(command: Sequence[str], *, stage: str, timeout: float) -> None:
        del stage, timeout
        calls.append(list(command))
        if len(calls) == 1:
            stage = "GPU SOG conversion"
            raise NativeCommandError(stage, 1, tmp_path / "native.log")

    monkeypatch.setattr(worker.runner, "run", run)

    worker._convert_sog(tmp_path / "room.ply", tmp_path / "scene.sog")  # noqa: SLF001

    assert "-g" not in calls[0]
    assert calls[1][calls[1].index("-g") + 1] == "cpu"
    assert any(isinstance(event, WarningEvent) for event in events)


def test_pycolmap_commands_use_the_isolated_worker_prefix(tmp_path: Path) -> None:
    """Keep native Python stages under the cancellable process supervisor."""
    prefix = (sys.executable, "-m", "better_backgrounds.pycolmap_worker")
    commands = ReconstructionCommands(
        ffmpeg=Path("ffmpeg"),
        pycolmap=prefix,
        brush=Path("brush"),
        splat_transform=Path("splat-transform"),
    )

    extraction = commands.feature_extraction(
        tmp_path / "images",
        tmp_path / "database.db",
    )
    matching = commands.sequential_matching(tmp_path / "database.db")

    assert extraction[: len(prefix)] == list(prefix)
    assert extraction[len(prefix)] == "feature-extractor"
    assert matching[: len(prefix)] == list(prefix)
    assert matching[len(prefix)] == "sequential-matcher"


def test_largest_useful_colmap_model_is_selected(tmp_path: Path) -> None:
    """Prefer the model with the most registered images deterministically."""
    first = tmp_path / "0"
    second = tmp_path / "1"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "images.txt").write_text("# Number of images: 12\n", encoding="utf-8")
    (second / "images.txt").write_text("# Number of images: 48\n", encoding="utf-8")

    assert select_largest_model(tmp_path) == second


def test_colmap_quality_measures_temporal_coverage_and_reprojection(tmp_path: Path) -> None:
    """Reject fragmented camera models using explainable sparse-model evidence."""
    images = []
    for image_id, frame_index in enumerate((0, 10, 20, 30, 40, 50, 59), start=1):
        images.extend(
            (
                f"{image_id} 1 0 0 0 0 0 0 1 {frame_index:06d}.png",
                "1.0 2.0 -1",
            )
        )
    (tmp_path / "images.txt").write_text("\n".join(images), encoding="utf-8")
    (tmp_path / "points3D.txt").write_text(
        "1 0 0 0 255 255 255 1.25 1 0\n2 0 0 0 255 255 255 1.75 2 0\n",
        encoding="utf-8",
    )

    quality = analyse_colmap_model(tmp_path, selected_count=60)

    assert quality.registered_images == len(images) // 2
    assert quality.temporal_span == pytest.approx(1.0)
    assert quality.longest_gap == EXPECTED_LONGEST_GAP
    assert quality.median_reprojection_error == pytest.approx(1.5)


def test_native_command_timeout_terminates_the_process_tree(tmp_path: Path) -> None:
    """Bound a stalled native stage and retain its durable log."""
    runner = NativeProcessRunner(threading.Event(), tmp_path / "native.log")

    with pytest.raises(NativeCommandError, match="timed out"):
        runner.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stage="test stage",
            timeout=SHORT_TIMEOUT,
        )

    assert (tmp_path / "native.log").is_file()


def test_native_command_uses_the_job_directory_as_its_working_directory(
    tmp_path: Path,
) -> None:
    """Keep tool-generated caches and relative outputs inside the job workspace."""
    job_root = tmp_path / "job"
    runner = NativeProcessRunner(threading.Event(), job_root / "native.log")

    runner.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('working-directory.txt').touch()",
        ],
        stage="working directory",
        timeout=1,
    )

    assert (job_root / "working-directory.txt").is_file()


def test_pre_cancelled_native_command_never_starts(tmp_path: Path) -> None:
    """Honor cooperative cancellation before launching expensive work."""
    cancelled = threading.Event()
    cancelled.set()
    runner = NativeProcessRunner(cancelled, tmp_path / "native.log")

    with pytest.raises(ReconstructionCancelledError):
        runner.run([sys.executable, "--version"], stage="test", timeout=1)

    assert not (tmp_path / "native.log").exists()
