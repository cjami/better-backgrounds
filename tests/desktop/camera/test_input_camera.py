"""Feature-first: Tests for input-camera discovery and persisted selection."""

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from better_backgrounds.desktop.camera import InputCameraSelectionStore, InputResolutionStore
from better_backgrounds.desktop.camera.capture import (
    FrameRateLimiter,
    camera_format_score,
    capture_profile,
    fit_frame_to_output,
    normalize_capture_frame,
)

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


def test_input_resolution_round_trips_with_a_1080p_default(tmp_path: Path) -> None:
    """Persist only an explicit user-selected live input tier."""
    path = tmp_path / "input-resolution-v1.json"
    store = InputResolutionStore(path)

    assert store.load() == "1080p"

    store.save("720p")

    assert store.load() == "720p"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_input_resolution_falls_back_to_1080p(tmp_path: Path) -> None:
    """Ignore unknown or incompatible persisted resolution data."""
    path = tmp_path / "input-resolution-v1.json"
    path.write_text('{"schema_version":1,"resolution":"480p"}', encoding="utf-8")

    assert InputResolutionStore(path).load() == "1080p"


def test_camera_format_prefers_native_1080p_at_the_target_frame_rate() -> None:
    """Prefer native 1080p without paying for unnecessary 4K or 60 fps capture."""
    target = camera_format_score(1920, 1080, 30.0, 30.0)

    assert target < camera_format_score(3840, 2160, 30.0, 30.0)
    assert target < camera_format_score(1920, 1080, 60.0, 60.0)
    assert target < camera_format_score(1280, 720, 30.0, 30.0)


def test_camera_format_targets_the_selected_720p_tier() -> None:
    """Prefer the requested native tier when the user selects 720p."""
    target = camera_format_score(1280, 720, 30.0, 30.0, "720p")

    assert target < camera_format_score(1920, 1080, 30.0, 30.0, "720p")
    assert target < camera_format_score(960, 540, 30.0, 30.0, "720p")


def test_capture_profile_maps_source_tiers_without_upscaling() -> None:
    """Map 4K and 1080p to 1080p, retain 720p, and preserve smaller sources."""
    assert capture_profile(3840, 2160, 30.0, 30.0).processing_width == 1920
    assert capture_profile(1920, 1080, 30.0, 30.0).processing_height == 1080
    assert capture_profile(1280, 720, 30.0, 30.0).processing_height == 720
    smaller = capture_profile(960, 540, 30.0, 30.0)
    assert (smaller.processing_width, smaller.processing_height) == (960, 540)


def test_capture_profile_caps_processing_at_selected_720p_tier() -> None:
    """Reduce all higher-resolution inputs when the user selects 720p."""
    profile = capture_profile(1920, 1080, 30.0, 30.0, "720p")

    assert (profile.processing_width, profile.processing_height) == (1280, 720)
    assert (profile.output_geometry(4 / 3).width, profile.output_geometry(4 / 3).height) == (
        960,
        720,
    )


def test_output_geometry_uses_tier_height_and_requested_aspect() -> None:
    """Treat 1080p as vertical resolution for every supported output aspect."""
    profile = capture_profile(3840, 2160, 30.0, 30.0)

    assert (profile.output_geometry(16 / 9).width, profile.output_geometry(16 / 9).height) == (
        1920,
        1080,
    )
    assert profile.output_geometry(4 / 3).width == 1440
    assert profile.output_geometry(1.0).width == 1080


def test_higher_resolution_capture_is_normalized_once() -> None:
    """Downsample 4K input to the selected 1080p processing canvas."""
    profile = capture_profile(3840, 2160, 30.0, 30.0)
    source = np.zeros((2160, 3840, 3), dtype=np.uint8)

    normalized = normalize_capture_frame(source, profile)

    assert normalized.shape == (1080, 1920, 3)


def test_source_and_alpha_are_cropped_together_for_narrower_output() -> None:
    """Preserve foreground alignment while moving from 16:9 to 4:3."""
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    alpha = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    geometry = capture_profile(8, 6, 30.0, 30.0).output_geometry(1.0)

    fitted_source, fitted_alpha = fit_frame_to_output(source, alpha, geometry)

    assert fitted_source.shape == (6, 6, 3)
    assert fitted_alpha.shape == (6, 6)
    assert np.array_equal(fitted_source[:, :, 0], source[:, 1:7, 0])
    assert np.array_equal(fitted_alpha, alpha[:, 1:7])


def test_narrow_source_is_background_padded_without_stretching() -> None:
    """Keep a narrow camera's pixels unchanged inside a wider output canvas."""
    source = np.full((4, 3, 3), 19, dtype=np.uint8)
    alpha = np.full((4, 3), 255, dtype=np.uint8)
    geometry = capture_profile(3, 4, 30.0, 30.0).output_geometry(1.5)

    fitted_source, fitted_alpha = fit_frame_to_output(source, alpha, geometry)

    assert fitted_source.shape == (4, 6, 3)
    assert np.array_equal(fitted_source[:, 1:4], source)
    assert np.count_nonzero(fitted_alpha[:, :1]) == 0


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


def test_capture_profile_preserves_non_widescreen_source_shape() -> None:
    """Normalize by tier height without stretching a 4:3 camera to 16:9."""
    profile = capture_profile(1440, 1080, 30.0, 30.0)

    assert (profile.processing_width, profile.processing_height) == (1440, 1080)
