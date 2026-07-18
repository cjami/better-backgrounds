"""Feature-first: Tests for one-shot MediaPipe person seed preparation."""

from __future__ import annotations

import numpy as np

from better_backgrounds.matting.seed import StableFrameSelector, largest_person_component


def test_largest_person_component_removes_smaller_people() -> None:
    """Initialize MatAnyone 2 with one unambiguous target person."""
    confidence = np.zeros((10, 10), dtype=np.float32)
    confidence[1:7, 1:5] = 0.9
    confidence[7:9, 7:9] = 0.9

    mask = largest_person_component(confidence, threshold=0.5)

    assert np.all(mask[1:7, 1:5] == 255)
    assert np.all(mask[7:9, 7:9] == 0)


def test_stable_frame_selector_waits_for_consecutive_low_motion_frames() -> None:
    """Avoid seeding temporal memory from a camera transition."""
    selector = StableFrameSelector(required_stable_frames=2, motion_threshold=3.0)
    dark = np.zeros((8, 8, 3), dtype=np.uint8)
    bright = np.full((8, 8, 3), 255, dtype=np.uint8)

    assert selector.offer(dark) is None
    assert selector.offer(bright) is None
    assert selector.offer(bright) is None
    selected = selector.offer(bright)

    assert selected is not None
    assert np.array_equal(selected, bright)


def test_stable_frame_selector_returns_an_owned_copy() -> None:
    """Keep the confirmed seed immutable while camera slots are recycled."""
    selector = StableFrameSelector(required_stable_frames=1, motion_threshold=1.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    selector.offer(frame)
    selected = selector.offer(frame)
    assert selected is not None

    frame.fill(255)

    assert not np.any(selected)
