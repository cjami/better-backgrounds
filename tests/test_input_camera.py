"""Tests for input-camera discovery and persisted selection."""

from itertools import pairwise
from typing import TYPE_CHECKING

from better_backgrounds.desktop.camera_capture import FrameRateLimiter, camera_format_score
from better_backgrounds.input_camera import InputCameraSelectionStore

if TYPE_CHECKING:
    from pathlib import Path


def test_input_camera_selection_round_trips_atomically(tmp_path: Path) -> None:
    """Restore the stable device identifier selected by the user."""
    store = InputCameraSelectionStore(tmp_path / "input-camera-v1.json")

    store.save("camera-b")

    assert store.load() == "camera-b"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_input_camera_selection_is_ignored(tmp_path: Path) -> None:
    """Recover from corrupt or incompatible preference data."""
    path = tmp_path / "input-camera-v1.json"
    path.write_text('{"schema_version":2,"device_id":"camera-b"}', encoding="utf-8")

    assert InputCameraSelectionStore(path).load() is None


def test_camera_format_prefers_720p_at_the_target_frame_rate() -> None:
    """Do not select a 60 fps capture mode that overloads the 30 fps pipeline."""
    target = camera_format_score(1280, 720, 30.0, 30.0)

    assert target < camera_format_score(1280, 720, 60.0, 60.0)
    assert target < camera_format_score(1920, 1080, 30.0, 30.0)
    assert camera_format_score(1280, 720, 60.0, 60.0) < camera_format_score(
        1920,
        1080,
        30.0,
        30.0,
    )
    assert camera_format_score(1920, 1080, 30.0, 30.0) < camera_format_score(
        1280,
        720,
        15.0,
        15.0,
    )


def test_camera_rate_limiter_evenly_samples_an_overproducing_backend() -> None:
    """Prefer a stable cadence when the source cannot divide evenly to 30 fps."""
    limiter = FrameRateLimiter(target_frame_rate=30.0)

    emitted = [timestamp for timestamp in range(0, 5_000, 20) if limiter.allows(float(timestamp))]

    assert 124 <= len(emitted) <= 126
    assert emitted[-1] >= 4_960
    assert {second - first for first, second in pairwise(emitted)} == {40}


def test_camera_rate_limiter_tolerates_nominal_thirty_fps_jitter() -> None:
    """Do not turn slightly early 30 fps frames into visible 66 ms gaps."""
    limiter = FrameRateLimiter(target_frame_rate=30.0)
    timestamps = [float(index * 33) for index in range(31)]

    emitted = [timestamp for timestamp in timestamps if limiter.allows(timestamp)]

    assert emitted == timestamps
