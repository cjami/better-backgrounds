"""Feature-first: Verify automatic live-matte boundary refinement."""

from __future__ import annotations

import numpy as np
import torch

from better_backgrounds.matting.accelerated import CudaLiveEngine, TensorAlphaStabilizer
from better_backgrounds.matting.refinement import (
    TemporalAlphaStabilizer,
    decode_srgb,
    decontaminate_foreground,
    encode_srgb,
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


def test_tensor_temporal_stabilizer_matches_the_reference_policy() -> None:
    """Keep CUDA temporal output within uint8 rounding of the NumPy path."""
    reference = TemporalAlphaStabilizer()
    accelerated = TensorAlphaStabilizer()
    frames = (
        np.array([[0, 100, 255], [12, 180, 245]], dtype=np.uint8),
        np.array([[0, 120, 255], [15, 160, 245]], dtype=np.uint8),
        np.array([[0, 220, 255], [9, 150, 245]], dtype=np.uint8),
    )

    for index, frame in enumerate(frames):
        captured_at = index * 1_000.0 / 30.0
        expected = reference.apply(frame, captured_at=captured_at)
        actual = accelerated.apply(
            torch.from_numpy(frame).reshape(1, 1, *frame.shape),
            captured_at=captured_at,
        )[0, 0].numpy()

        assert np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max() <= 1


def test_edge_decontamination_removes_a_bright_original_background() -> None:
    """Recover foreground colour from soft pixels captured against a white wall."""
    height, width = 32, 32
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[:, 8:17] = np.linspace(0, 255, 9, dtype=np.uint8)
    alpha[:, 17:] = 255
    foreground_value = 30.0
    original_background_value = 240.0
    weight = alpha.astype(np.float32) / 255.0
    foreground = np.full((height, width, 3), foreground_value, dtype=np.uint8)
    original_background = np.full_like(foreground, original_background_value)
    observed = decode_srgb(foreground) * weight[..., None]
    observed += decode_srgb(original_background) * (1.0 - weight[..., None])
    source = encode_srgb(observed)

    cleaned = decontaminate_foreground(source, alpha)

    uncertain = (alpha >= 32) & (alpha <= 224)
    original_error = np.abs(source[uncertain].astype(np.float32) - foreground_value).mean()
    cleaned_error = np.abs(cleaned[uncertain].astype(np.float32) - foreground_value).mean()
    assert cleaned_error < original_error * 0.45
    assert np.array_equal(cleaned[alpha == 0], source[alpha == 0])
    assert np.array_equal(cleaned[alpha == 255], source[alpha == 255])


def test_tensor_decontamination_matches_the_linear_reference() -> None:
    """Keep CUDA edge recovery in the same radiometric space as the portable path."""
    height, width = 32, 32
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[:, 8:17] = np.linspace(0, 255, 9, dtype=np.uint8)
    alpha[:, 17:] = 255
    foreground = np.full((height, width, 3), 30, dtype=np.uint8)
    original_background = np.full_like(foreground, 240)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    observed = decode_srgb(foreground) * weight
    observed += decode_srgb(original_background) * (1.0 - weight)
    source = encode_srgb(observed)
    expected = decontaminate_foreground(source, alpha)
    source_tensor = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float().div(255.0)

    actual = CudaLiveEngine._decontaminate(source_tensor, alpha_tensor)  # noqa: SLF001
    actual = actual[0].mul(255.0).round().byte().permute(1, 2, 0).numpy()

    assert np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max() <= 4
