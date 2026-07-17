"""Verify automatic live-matte boundary refinement."""

from __future__ import annotations

import numpy as np

from better_backgrounds.matte_refinement import (
    TemporalAlphaStabilizer,
    decontaminate_foreground,
)


def test_temporal_stabilizer_damps_small_boundary_changes_only() -> None:
    """Suppress boundary shimmer without delaying decisive matte movement."""
    stabilizer = TemporalAlphaStabilizer()
    first = np.array([[0, 100, 255]], dtype=np.uint8)
    small_change = np.array([[0, 120, 255]], dtype=np.uint8)
    large_change = np.array([[0, 220, 255]], dtype=np.uint8)

    assert np.array_equal(stabilizer.apply(first, captured_at=0.0), first)

    stabilized = stabilizer.apply(small_change, captured_at=1_000.0 / 30.0)
    released = stabilizer.apply(large_change, captured_at=2_000.0 / 30.0)

    assert 100 < int(stabilized[0, 1]) < 120
    assert int(released[0, 1]) == 220
    assert np.array_equal(stabilized[:, [0, 2]], small_change[:, [0, 2]])


def test_temporal_stabilizer_releases_after_a_frame_stall() -> None:
    """Never blend a stale boundary into a resumed camera frame."""
    stabilizer = TemporalAlphaStabilizer()
    stabilizer.apply(np.array([[100]], dtype=np.uint8), captured_at=0.0)
    current = np.array([[120]], dtype=np.uint8)

    assert np.array_equal(stabilizer.apply(current, captured_at=200.0), current)


def test_edge_decontamination_removes_a_bright_original_background() -> None:
    """Recover foreground colour from soft pixels captured against a white wall."""
    height, width = 32, 32
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[:, 8:17] = np.linspace(0, 255, 9, dtype=np.uint8)
    alpha[:, 17:] = 255
    foreground_value = 30.0
    original_background_value = 240.0
    weight = alpha.astype(np.float32) / 255.0
    observed = foreground_value * weight + original_background_value * (1.0 - weight)
    source = np.repeat(np.rint(observed)[..., None], 3, axis=2).astype(np.uint8)

    cleaned = decontaminate_foreground(source, alpha)

    uncertain = (alpha >= 32) & (alpha <= 224)
    original_error = np.abs(source[uncertain].astype(np.float32) - foreground_value).mean()
    cleaned_error = np.abs(cleaned[uncertain].astype(np.float32) - foreground_value).mean()
    assert cleaned_error < original_error * 0.45
    assert np.array_equal(cleaned[alpha == 0], source[alpha == 0])
    assert np.array_equal(cleaned[alpha == 255], source[alpha == 255])
