"""Feature-first: Tests for the fixed shared-memory webcam frame ring."""

from __future__ import annotations

import numpy as np
import pytest

from better_backgrounds.matting.ring import SharedFrameRing


def test_frame_ring_round_trips_rgb_and_alpha_by_slot() -> None:
    """Share exact source pixels and their matching alpha without pickling them."""
    ring = SharedFrameRing.create(width=4, height=3)
    source = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    alpha = np.arange(12, dtype=np.uint8).reshape(3, 4)
    primary = np.flip(source, axis=1).copy()
    standard = np.flip(source, axis=0).copy()
    try:
        ring.write_frame(1, source)
        ring.write_alpha(1, alpha)
        ring.write_output(1, primary)
        ring.write_output(1, standard, standard=True)

        assert np.array_equal(ring.read_frame(1), source)
        assert np.array_equal(ring.read_alpha(1), alpha)
        assert np.array_equal(ring.read_output(1), primary)
        assert np.array_equal(ring.read_output(1, standard=True), standard)
    finally:
        ring.close(unlink=True)


def test_attached_ring_observes_writes_from_the_owner() -> None:
    """Allow a spawned worker to attach using serializable descriptors."""
    owner = SharedFrameRing.create(width=2, height=2)
    attached = SharedFrameRing.attach(owner.descriptor)
    source = np.full((2, 2, 3), 123, dtype=np.uint8)
    try:
        owner.write_frame(0, source)

        assert np.array_equal(attached.read_frame(0), source)
    finally:
        attached.close()
        owner.close(unlink=True)


def test_frame_ring_rejects_wrong_frame_shape() -> None:
    """Prevent a camera format change from corrupting adjacent slots."""
    ring = SharedFrameRing.create(width=4, height=3)
    try:
        with pytest.raises(ValueError, match="frame shape"):
            ring.write_frame(0, np.zeros((2, 4, 3), dtype=np.uint8))
    finally:
        ring.close(unlink=True)
