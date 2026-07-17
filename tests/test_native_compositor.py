"""Tests for exact-frame native alpha composition."""

from __future__ import annotations

import numpy as np
import pytest

from better_backgrounds.compositor import compose_live_frame
from better_backgrounds.live_matting import FramePacket, MatteResult


def test_compositor_uses_exact_source_alpha_and_background() -> None:
    """Blend the matching source without a stale or future matte."""
    source = np.array([[[200, 100, 50], [20, 30, 40]]], dtype=np.uint8)
    background = np.array([[[0, 0, 0], [100, 110, 120]]], dtype=np.uint8)
    alpha = np.array([[255, 0]], dtype=np.uint8)
    packet = FramePacket(1, 10.0, 2, 1, 0)
    matte = MatteResult(1, 10.0, 0, 12.0)

    composite = compose_live_frame(packet, matte, source, alpha, background, revision=4)

    assert np.array_equal(composite.image, np.array([[[200, 100, 50], [100, 110, 120]]]))
    assert composite.frame_id == 1
    assert composite.background_revision == 4


def test_compositor_rejects_mismatched_frame_and_matte() -> None:
    """Make tearing through frame/mask mismatch structurally impossible."""
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    alpha = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="same source frame"):
        compose_live_frame(
            FramePacket(1, 10.0, 2, 2, 0),
            MatteResult(2, 20.0, 0, 10.0),
            source,
            alpha,
            source,
            revision=0,
        )
