"""Tests for conservative room-scoped foreground harmonization."""

from __future__ import annotations

import numpy as np

from better_backgrounds.harmonization import (
    DEGRADATION_STEPS,
    AcceleratedAppearanceHarmonizer,
    AppearanceHarmonizer,
    HarmonizationBudget,
    HarmonizationSettings,
    preprocess_room,
)


def room_image() -> np.ndarray:
    """Return a non-uniform warm room image."""
    horizontal = np.linspace(35, 180, 24, dtype=np.uint8)[None, :, None]
    room = np.broadcast_to(horizontal, (16, 24, 1))
    return np.concatenate((room, room * 3 // 4, room // 2), axis=2).copy()


def test_room_preprocessing_caches_robust_and_directional_evidence() -> None:
    """Derive bounded scene evidence once when the room snapshot changes."""
    room = room_image()

    descriptor = preprocess_room(room, revision=7)

    assert descriptor.revision == 7
    assert descriptor.image.shape == room.shape
    assert descriptor.blurred_image.shape == room.shape
    assert descriptor.ambient_rgb.shape == (3,)
    assert descriptor.left_rgb[0] < descriptor.right_rgb[0]
    assert np.all(np.isfinite(descriptor.ambient_rgb))
    assert not descriptor.image.flags.writeable


def test_all_harmonization_components_have_an_off_state() -> None:
    """Keep the exact-frame standard composite as the product default."""
    settings = HarmonizationSettings()

    assert not settings.active
    assert not settings.global_appearance
    assert not settings.directional_shading
    assert not settings.edge_decontamination
    assert not settings.light_wrap
    assert not settings.detail_match
    assert not settings.depth_effects


def test_harmonizer_uses_only_high_confidence_foreground_statistics() -> None:
    """Ignore background and uncertain boundary pixels during parameter estimation."""
    room = room_image()
    source = np.full_like(room, [45, 70, 110])
    source[:, :8] = [250, 10, 250]
    alpha = np.zeros(room.shape[:2], dtype=np.uint8)
    alpha[:, 8:] = 255
    harmonizer = AppearanceHarmonizer(
        HarmonizationSettings(global_appearance=True),
        statistics_interval=0.0,
    )
    harmonizer.set_room(room, revision=1)

    result = harmonizer.apply(source, alpha, room, captured_at=1.0)

    assert result.statistics_updated
    assert np.all(result.parameters.channel_gains >= 0.92)
    assert np.all(result.parameters.channel_gains <= 1.08)
    assert result.foreground is not None
    assert not np.array_equal(result.foreground[:, 8:], source[:, 8:])
    assert np.array_equal(result.foreground[:, :8], source[:, :8])


def test_parameter_estimation_is_rate_limited_and_smoothed() -> None:
    """Update targets at low frequency while interpolating parameters each frame."""
    room = room_image()
    source = np.full_like(room, 55)
    alpha = np.full(room.shape[:2], 255, dtype=np.uint8)
    harmonizer = AppearanceHarmonizer(
        HarmonizationSettings(global_appearance=True),
        statistics_interval=0.2,
        smoothing_time=0.5,
    )
    harmonizer.set_room(room, revision=2)

    first = harmonizer.apply(source, alpha, room, captured_at=10.0)
    second = harmonizer.apply(source, alpha, room, captured_at=10.05)
    third = harmonizer.apply(source, alpha, room, captured_at=10.25)

    assert first.statistics_updated
    assert not second.statistics_updated
    assert third.statistics_updated
    assert first.parameters.exposure_stops != third.parameters.exposure_stops
    assert abs(first.parameters.exposure_stops) < abs(third.parameters.exposure_stops)


def test_room_and_camera_changes_reset_live_parameters() -> None:
    """Prevent appearance estimates leaking across rooms or camera sources."""
    room = room_image()
    alpha = np.full(room.shape[:2], 255, dtype=np.uint8)
    harmonizer = AppearanceHarmonizer(
        HarmonizationSettings(global_appearance=True),
        statistics_interval=0.0,
    )
    harmonizer.set_room(room, revision=3)
    changed = harmonizer.apply(np.full_like(room, 30), alpha, room, captured_at=1.0)
    assert changed.parameters.exposure_stops != 0.0

    harmonizer.reset_camera()
    assert harmonizer.parameters.exposure_stops == 0.0

    harmonizer.set_room(np.flip(room, axis=1).copy(), revision=4)
    assert harmonizer.parameters.exposure_stops == 0.0
    assert harmonizer.room_revision == 4


def test_directional_shading_is_bounded_and_foreground_only() -> None:
    """Apply broad scene direction without altering definite background pixels."""
    room = room_image()
    source = np.full_like(room, 120)
    alpha = np.zeros(room.shape[:2], dtype=np.uint8)
    alpha[:, 4:20] = 255
    harmonizer = AppearanceHarmonizer(
        HarmonizationSettings(directional_shading=True),
    )
    harmonizer.set_room(room, revision=5)

    result = harmonizer.apply(source, alpha, room, captured_at=1.0)
    assert result.foreground is not None
    foreground = result.foreground[:, 4:20].astype(np.int16)

    assert np.max(np.abs(foreground - 120)) <= 20
    assert foreground[:, 0].mean() < foreground[:, -1].mean()
    assert np.array_equal(result.foreground[:, :4], source[:, :4])


def test_unsupported_depth_effects_degrade_to_standard_output() -> None:
    """Require reliable renderer evidence before applying depth-dependent effects."""
    room = room_image()
    source = np.full_like(room, 100)
    alpha = np.full(room.shape[:2], 128, dtype=np.uint8)
    harmonizer = AppearanceHarmonizer(HarmonizationSettings(depth_effects=True))
    harmonizer.set_room(room, revision=6)

    result = harmonizer.apply(source, alpha, room, captured_at=1.0)

    assert result.foreground is not None
    assert np.array_equal(result.foreground, source)
    assert result.degraded_components == ("depth_effects",)
    assert not result.applied


def test_frame_budget_degrades_in_the_declared_order() -> None:
    """Fall back predictably after sustained rather than isolated overruns."""
    budget = HarmonizationBudget(2.0, overrun_limit=2)

    for expected in DEGRADATION_STEPS:
        budget.observe(2.1)
        budget.observe(2.1)
        assert budget.degraded_components[-1] == expected

    assert budget.degraded_components == DEGRADATION_STEPS


def test_unbudgeted_reference_preview_does_not_silently_disable_effects() -> None:
    """Keep opt-in visual evaluation available before an accelerated pass exists."""
    room = room_image()
    source = np.full_like(room, 70)
    alpha = np.full(room.shape[:2], 255, dtype=np.uint8)
    harmonizer = AppearanceHarmonizer(
        HarmonizationSettings(global_appearance=True),
        statistics_interval=0.0,
        budget=HarmonizationBudget(0.0001, overrun_limit=1, enforced=False),
    )
    harmonizer.set_room(room, revision=8)

    results = [harmonizer.apply(source, alpha, room, captured_at=frame / 30) for frame in range(10)]

    assert all(result.applied for result in results)
    assert harmonizer.degradation_stage == 0


def test_accelerated_backend_preserves_the_result_contract() -> None:
    """Keep the CUDA/Metal tensor path interchangeable with the reference renderer."""
    room = room_image()
    source = np.full_like(room, [60, 80, 100])
    alpha = np.full(room.shape[:2], 255, dtype=np.uint8)
    harmonizer = AcceleratedAppearanceHarmonizer(preferred_device="cpu")
    harmonizer.configure(
        HarmonizationSettings(
            global_appearance=True,
            directional_shading=True,
            edge_decontamination=True,
            light_wrap=True,
            detail_match=True,
        )
    )
    harmonizer.set_room(room, revision=9)

    result = harmonizer.apply(source, alpha, room, captured_at=1.0)

    assert result.applied
    assert result.foreground is None
    assert result.image is not None
    assert result.image.shape == source.shape
    assert harmonizer.backend_name == "CPU"
