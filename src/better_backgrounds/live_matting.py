"""Contracts and bounded scheduling for stateful live video matting."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

RING_SLOTS = 3
MINIMUM_RATE_SAMPLES = 2
INTERNAL_SIZES = (360, 432, 540)
InternalSize = Literal[360, 432, 540]


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Identify RGB pixels stored in one shared-memory slot."""

    frame_id: int
    captured_at: float
    width: int
    height: int
    shared_slot: int

    def __post_init__(self) -> None:
        """Reject corrupt process messages before they reach shared memory."""
        if self.frame_id < 0:
            msg = "frame id must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.captured_at) or self.captured_at < 0:
            msg = "capture timestamp must be finite and non-negative"
            raise ValueError(msg)
        if self.width <= 0 or self.height <= 0:
            msg = "frame dimensions must be positive"
            raise ValueError(msg)
        if not 0 <= self.shared_slot < RING_SLOTS:
            msg = "shared slot must be inside the three-slot ring"
            raise ValueError(msg)


class MattingConfig(BaseModel):
    """Select one supported MatAnyone 2 inference configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    internal_size: InternalSize = 432
    warmup_iterations: int = Field(default=10, ge=1, le=30)
    calibrate: bool = False
    latency_budget_ms: float = Field(default=66.0, gt=0, le=1_000, allow_inf_nan=False)
    calibration_frames: int = Field(default=3, ge=2, le=10)


@dataclass(frozen=True, slots=True)
class MatteResult:
    """Identify alpha pixels written by the worker for one source frame."""

    frame_id: int
    captured_at: float
    alpha_slot: int
    inference_ms: float

    def __post_init__(self) -> None:
        """Keep worker messages finite and bound to the shared ring."""
        if self.frame_id < 0:
            msg = "frame id must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.captured_at) or self.captured_at < 0:
            msg = "capture timestamp must be finite and non-negative"
            raise ValueError(msg)
        if not 0 <= self.alpha_slot < RING_SLOTS:
            msg = "alpha slot must be inside the three-slot ring"
            raise ValueError(msg)
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0:
            msg = "inference time must be finite and non-negative"
            raise ValueError(msg)


class MattingCapabilities(BaseModel):
    """Describe the effective MatAnyone 2 execution backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_type: Literal["cuda", "mps", "cpu"]
    accelerated: bool
    supported_sizes: tuple[InternalSize, ...] = INTERNAL_SIZES


class LiveDiagnostics(BaseModel):
    """Publish bounded local-only performance counters for the native pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture_fps: float = Field(ge=0, le=1_000, allow_inf_nan=False)
    display_fps: float = Field(ge=0, le=1_000, allow_inf_nan=False)
    mask_fps: float = Field(ge=0, le=1_000, allow_inf_nan=False)
    mask_age_ms: float = Field(ge=0, le=60_000, allow_inf_nan=False)
    dropped_frames: int = Field(ge=0)
    worker_time_ms: float = Field(ge=0, le=60_000, allow_inf_nan=False)
    capture_width: int = Field(gt=0, le=8_192)
    capture_height: int = Field(gt=0, le=8_192)
    processing_width: int = Field(gt=0, le=8_192)
    processing_height: int = Field(gt=0, le=8_192)
    device_type: Literal["cuda", "mps", "cpu"]


class SlidingFrameRate:
    """Measure recent cadence without hiding current stalls in a session average."""

    def __init__(self, *, window_ms: float = 2_000.0) -> None:
        """Retain timestamps from one bounded rolling window."""
        if not math.isfinite(window_ms) or window_ms <= 0:
            msg = "frame-rate window must be finite and positive"
            raise ValueError(msg)
        self.window_ms = window_ms
        self._timestamps: deque[float] = deque()

    @property
    def rate(self) -> float:
        """Return intervals per second across the retained window."""
        if len(self._timestamps) < MINIMUM_RATE_SAMPLES:
            return 0.0
        elapsed_ms = self._timestamps[-1] - self._timestamps[0]
        return 0.0 if elapsed_ms <= 0 else (len(self._timestamps) - 1) * 1_000.0 / elapsed_ms

    def record(self, timestamp_ms: float) -> float:
        """Add one monotonic event and return its updated recent rate."""
        if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
            msg = "frame timestamp must be finite and non-negative"
            raise ValueError(msg)
        if self._timestamps and timestamp_ms < self._timestamps[-1]:
            self.reset()
        self._timestamps.append(timestamp_ms)
        cutoff = timestamp_ms - self.window_ms
        while len(self._timestamps) > 1 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return self.rate

    def reset(self) -> None:
        """Forget cadence from the previous camera or worker session."""
        self._timestamps.clear()


@dataclass(frozen=True, slots=True)
class SchedulerSubmission:
    """Describe work to dispatch and stale work to release."""

    dispatch: FramePacket | None
    dropped: FramePacket | None = None


@dataclass(frozen=True, slots=True)
class SchedulerCompletion:
    """Return completed work and the pending work promoted behind it."""

    completed: FramePacket
    dispatch: FramePacket | None


class FrameMismatchError(RuntimeError):
    """Indicate that a worker result does not belong to the active frame."""


class LatestFrameScheduler:
    """Allow one active inference and one replaceable newest pending frame."""

    def __init__(self) -> None:
        """Start without owning any shared-memory slots."""
        self.active: FramePacket | None = None
        self.pending: FramePacket | None = None
        self.dropped_frames = 0

    def submit(self, packet: FramePacket) -> SchedulerSubmission:
        """Dispatch immediately or retain only the newest pending packet."""
        if self.active is None:
            self.active = packet
            return SchedulerSubmission(dispatch=packet)
        dropped = self.pending
        self.pending = packet
        if dropped is not None:
            self.dropped_frames += 1
        return SchedulerSubmission(dispatch=None, dropped=dropped)

    def complete(self, result: MatteResult) -> SchedulerCompletion:
        """Complete the exact active frame and promote pending work."""
        active = self.active
        if active is None or (
            result.frame_id != active.frame_id
            or result.captured_at != active.captured_at
            or result.alpha_slot != active.shared_slot
        ):
            msg = "matte result does not match the active source frame"
            raise FrameMismatchError(msg)
        promoted = self.pending
        self.active = promoted
        self.pending = None
        return SchedulerCompletion(completed=active, dispatch=promoted)

    def reset(self) -> tuple[FramePacket, ...]:
        """Forget and return all packets whose slots must be released."""
        owned = tuple(packet for packet in (self.active, self.pending) if packet is not None)
        self.active = None
        self.pending = None
        return owned


def choose_internal_size(
    p95_latency_by_size: dict[int, float],
    *,
    budget_ms: float,
) -> InternalSize:
    """Choose the highest calibrated size within budget, or the safe minimum."""
    eligible = [
        size
        for size in INTERNAL_SIZES
        if math.isfinite(p95_latency_by_size.get(size, math.inf))
        and p95_latency_by_size[size] <= budget_ms
    ]
    return cast("InternalSize", max(eligible, default=INTERNAL_SIZES[0]))
