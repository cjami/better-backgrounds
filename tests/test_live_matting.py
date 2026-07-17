"""Tests for the MatAnyone 2 live-frame contracts and scheduler."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
from pydantic import ValidationError

from better_backgrounds.live_matting import (
    FrameMismatchError,
    FramePacket,
    LatestFrameScheduler,
    MatteResult,
    MattingConfig,
    SlidingFrameRate,
    calibration_p95_latency,
    choose_internal_size,
)


def packet(frame_id: int, slot: int) -> FramePacket:
    """Build a valid packet with a deterministic timestamp."""
    return FramePacket(
        frame_id=frame_id,
        captured_at=float(frame_id * 10),
        width=1280,
        height=720,
        shared_slot=slot,
    )


def result(source: FramePacket) -> MatteResult:
    """Build a matching result for the provided packet."""
    return MatteResult(
        frame_id=source.frame_id,
        captured_at=source.captured_at,
        alpha_slot=source.shared_slot,
        inference_ms=20.0,
    )


def test_matting_config_restricts_calibrated_sizes() -> None:
    """Accept only the resolutions evaluated by startup calibration."""
    config = MattingConfig(internal_size=432)

    assert config.warmup_iterations == 10
    assert config.calibration_frames == 20

    with pytest.raises(ValidationError):
        MattingConfig(internal_size=500)  # ty: ignore[invalid-argument-type]


def test_calibration_p95_ignores_one_isolated_scheduling_spike() -> None:
    """Measure sustained inference rather than one unrelated startup stall."""
    samples = [20.0] * 19 + [80.0]

    assert calibration_p95_latency(samples) == pytest.approx(23.0)


def test_scheduler_keeps_one_active_and_only_the_newest_pending_frame() -> None:
    """Drop stale pending work without interrupting active inference."""
    scheduler = LatestFrameScheduler()
    first = scheduler.submit(packet(1, 0))
    second = scheduler.submit(packet(2, 1))
    third = scheduler.submit(packet(3, 2))

    assert first.dispatch == packet(1, 0)
    assert second.dispatch is None
    assert third.dropped == packet(2, 1)
    assert scheduler.dropped_frames == 1

    completion = scheduler.complete(result(packet(1, 0)))

    assert completion.completed == packet(1, 0)
    assert completion.dispatch == packet(3, 2)


def test_scheduler_rejects_a_matte_for_a_different_source_frame() -> None:
    """Never combine a source frame with another frame's alpha matte."""
    scheduler = LatestFrameScheduler()
    active = packet(1, 0)
    scheduler.submit(active)

    with pytest.raises(FrameMismatchError):
        scheduler.complete(replace(result(active), frame_id=2))


def test_scheduler_reset_returns_every_owned_slot() -> None:
    """Release active and pending shared-memory slots during teardown."""
    scheduler = LatestFrameScheduler()
    scheduler.submit(packet(1, 0))
    scheduler.submit(packet(2, 1))

    assert scheduler.reset() == (packet(1, 0), packet(2, 1))
    assert scheduler.active is None
    assert scheduler.pending is None


@pytest.mark.parametrize(
    ("latencies", "expected"),
    [
        ({360: 30.0, 432: 50.0, 540: 65.0}, 540),
        ({360: 30.0, 432: 60.0, 540: 80.0}, 432),
        ({360: 80.0, 432: 90.0, 540: 100.0}, 360),
    ],
)
def test_calibration_selects_highest_size_within_budget(
    latencies: dict[int, float],
    expected: int,
) -> None:
    """Prefer detail while preserving the fixed p95 latency budget."""
    assert choose_internal_size(latencies, budget_ms=66.0) == expected


def test_sliding_frame_rate_reports_recent_cadence_and_recovers_from_clock_reset() -> None:
    """Report current delivery rate instead of a lifetime average."""
    rate = SlidingFrameRate(window_ms=1_000.0)

    for index in range(31):
        rate.record(index * (1_000.0 / 30.0))

    assert rate.rate == pytest.approx(30.0)

    rate.record(0.0)

    assert rate.rate == 0.0


def test_frame_packet_rejects_invalid_shared_slot() -> None:
    """Keep packets inside the three-slot ring."""
    with pytest.raises(ValueError, match="shared slot"):
        packet(1, 3)


def test_alpha_arrays_are_not_part_of_process_messages() -> None:
    """Keep process messages small and carry pixels through shared memory."""
    matte = result(packet(1, 0))

    assert not any(isinstance(getattr(matte, field.name), np.ndarray) for field in fields(matte))
