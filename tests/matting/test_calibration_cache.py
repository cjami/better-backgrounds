"""Feature-first: Tests for reusable MatAnyone device calibration."""

from __future__ import annotations

import multiprocessing
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np

from better_backgrounds.matting.calibration import (
    CalibrationIdentity,
    CalibrationProfile,
    CalibrationProfiles,
    CalibrationProfileStore,
)
from better_backgrounds.matting.contracts import MattingCapabilities, MattingConfig
from better_backgrounds.matting.ring import FrameRingDescriptor
from better_backgrounds.matting.worker import _calibrated_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from better_backgrounds.matting.runtime import MatAnyoneRuntime


def identity(*, width: int = 1280) -> CalibrationIdentity:
    """Build one deterministic local runtime identity."""
    return CalibrationIdentity(
        checkpoint_sha256="a" * 64,
        upstream_revision="revision",
        device_type="cuda",
        device_name="Test GPU",
        torch_version="2.9",
        accelerator_version="13.0",
        capture_width=width,
        capture_height=720,
        latency_budget_ms=33.3,
    )


def test_calibration_store_requires_an_exact_runtime_identity(tmp_path: Path) -> None:
    """Invalidate saved performance when capture geometry or runtime identity changes."""
    store = CalibrationProfileStore(tmp_path / "matting-calibration-v1.json")
    profile = CalibrationProfile(
        identity=identity(),
        selected_internal_size=432,
        measured_p95_ms=27.0,
    )

    store.save(profile)

    assert store.find(identity()) == profile
    assert store.find(identity(width=1920)) is None


def test_first_calibration_stops_at_the_highest_passing_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid benchmarking lower resolutions after the quality-first size passes."""
    measured: list[int] = []

    def measure(
        _runtime: object,
        _config: object,
        size: int,
        _frame: np.ndarray,
        _mask: np.ndarray,
        *,
        frames: int,
    ) -> float:
        measured.append(size)
        assert frames == 20
        return 20.0

    monkeypatch.setattr("better_backgrounds.matting.worker._measure_size", measure)
    runtime = SimpleNamespace(
        capabilities=MattingCapabilities(device_type="cuda", accelerated=True),
        calibration_device_identity=lambda: ("Test GPU", "2.9", "13.0"),
    )
    descriptor = FrameRingDescriptor("f", "a", "p", 1280, 720, 1280, 720)
    store = CalibrationProfileStore(tmp_path / "matting-calibration-v1.json")
    events = multiprocessing.get_context("spawn").Queue()

    try:
        config, selected = _calibrated_config(
            cast("MatAnyoneRuntime", runtime),
            MattingConfig(calibrate=True, latency_budget_ms=33.3),
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2), 255, dtype=np.uint8),
            descriptor,
            store,
            {},
            events,
            1,
        )
    finally:
        events.close()
        events.join_thread()

    assert measured == [540]
    assert selected == 540
    assert config.internal_size == 540
    assert not config.calibrate


def test_calibration_uses_minimum_size_when_accelerated_device_misses_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep matting available at the best achievable cadence on slower GPUs."""
    measured: list[tuple[int, int]] = []

    def measure(
        _runtime: object,
        _config: object,
        size: int,
        _frame: np.ndarray,
        _mask: np.ndarray,
        *,
        frames: int,
    ) -> float:
        measured.append((size, frames))
        return {540: 120.0, 432: 100.0, 360: 80.0}[size]

    monkeypatch.setattr("better_backgrounds.matting.worker._measure_size", measure)
    runtime = SimpleNamespace(
        capabilities=MattingCapabilities(device_type="cuda", accelerated=True),
        calibration_device_identity=lambda: ("Slow GPU", "2.9", "13.0"),
    )
    descriptor = FrameRingDescriptor("f", "a", "p", 1280, 720, 1280, 720)
    profile_path = tmp_path / "matting-calibration-v1.json"
    store = CalibrationProfileStore(profile_path)
    events = multiprocessing.get_context("spawn").Queue()
    requested = MattingConfig(calibrate=True, latency_budget_ms=33.3)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.full((2, 2), 255, dtype=np.uint8)

    try:
        config, selected = _calibrated_config(
            cast("MatAnyoneRuntime", runtime),
            requested,
            frame,
            mask,
            descriptor,
            store,
            {},
            events,
            1,
        )
    finally:
        events.close()
        events.join_thread()

    assert measured == [(540, 20), (432, 20), (360, 20)]
    assert selected == 360
    assert config.internal_size == 360
    assert not config.calibrate
    profiles = CalibrationProfiles.model_validate_json(profile_path.read_text(encoding="utf-8"))
    profile = profiles.profiles[0]
    assert profile.selected_internal_size == 360
    assert profile.measured_p95_ms == 80.0

    measured.clear()
    cached_events = multiprocessing.get_context("spawn").Queue()
    try:
        cached_config, cached_selected = _calibrated_config(
            cast("MatAnyoneRuntime", runtime),
            requested,
            frame,
            mask,
            descriptor,
            store,
            {},
            cached_events,
            2,
        )
    finally:
        cached_events.close()
        cached_events.join_thread()

    assert measured == [(360, 5)]
    assert cached_selected == 360
    assert cached_config.internal_size == 360
