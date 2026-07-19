"""Contracts and bounded scheduling for stateful live video matting."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

RING_SLOTS = 3
MINIMUM_RATE_SAMPLES = 2
INTERNAL_SIZES = (360, 432, 540)
InternalSize = Literal[360, 432, 540]
MASK_DIMENSIONS = 2

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Record the independently actionable latency stages for one output frame."""

    normalization_ms: float = 0.0
    queue_ms: float = 0.0
    matting_ms: float = 0.0
    post_processing_ms: float = 0.0
    readback_ms: float = 0.0

    def __post_init__(self) -> None:
        """Reject corrupt diagnostic values at process boundaries."""
        values = (
            self.normalization_ms,
            self.queue_ms,
            self.matting_ms,
            self.post_processing_ms,
            self.readback_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            msg = "stage timings must be finite and non-negative"
            raise ValueError(msg)


class LivePipelineConfig(BaseModel):
    """Describe immutable output and acceleration choices for one live session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_width: int = Field(gt=0, le=8_192)
    output_height: int = Field(gt=0, le=8_192)
    aspect_ratio: float = Field(gt=0, le=8.0, allow_inf_nan=False)
    prefer_cuda: bool = True
    mirrored: bool = True
    retain_standard: bool = False
    revision: int = Field(default=0, ge=0)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """Carry one exact completed frame and its final process-owned outputs."""

    packet: FramePacket
    primary: NDArray
    standard: NDArray | None
    mask_preview: NDArray
    background_revision: int
    occupancy: float
    timings: StageTimings
    pipeline_revision: int = 0
    harmonized: bool = False
    harmonization_ms: float = 0.0
    harmonization_degraded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate frame identity and bounded output evidence."""
        if self.background_revision < 0:
            msg = "background revision must be non-negative"
            raise ValueError(msg)
        if self.pipeline_revision < 0:
            msg = "pipeline revision must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.occupancy) or not 0.0 <= self.occupancy <= 1.0:
            msg = "occupancy must be between zero and one"
            raise ValueError(msg)
        expected = (self.packet.height, self.packet.width, 3)
        if self.primary.dtype.name != "uint8" or self.primary.shape != expected:
            msg = f"primary output must be {expected} uint8 RGB"
            raise ValueError(msg)
        if self.standard is not None and (
            self.standard.dtype.name != "uint8" or self.standard.shape != expected
        ):
            msg = f"standard output must be {expected} uint8 RGB"
            raise ValueError(msg)
        if self.mask_preview.dtype.name != "uint8" or self.mask_preview.ndim != MASK_DIMENSIONS:
            msg = "mask preview must be two-dimensional uint8"
            raise ValueError(msg)
        if not math.isfinite(self.harmonization_ms) or self.harmonization_ms < 0:
            msg = "harmonization time must be finite and non-negative"
            raise ValueError(msg)


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
    calibration_frames: int = Field(default=20, ge=2, le=120)


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


@dataclass(frozen=True, slots=True)
class ProcessedResult:
    """Identify final output pixels written by the fused worker."""

    frame_id: int
    captured_at: float
    alpha_slot: int
    output_width: int
    output_height: int
    background_revision: int
    occupancy: float
    mask_preview: NDArray
    timings: StageTimings
    standard_retained: bool = False
    pipeline_revision: int = 0
    harmonized: bool = False
    harmonization_ms: float = 0.0
    harmonization_degraded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the small result message before shared output is read."""
        MatteResult(
            self.frame_id,
            self.captured_at,
            self.alpha_slot,
            self.timings.matting_ms,
        )
        if self.output_width <= 0 or self.output_height <= 0:
            msg = "processed output dimensions must be positive"
            raise ValueError(msg)
        if self.background_revision < 0:
            msg = "background revision must be non-negative"
            raise ValueError(msg)
        if self.pipeline_revision < 0:
            msg = "pipeline revision must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.occupancy) or not 0.0 <= self.occupancy <= 1.0:
            msg = "occupancy must be between zero and one"
            raise ValueError(msg)
        if self.mask_preview.dtype.name != "uint8" or self.mask_preview.ndim != MASK_DIMENSIONS:
            msg = "processed mask preview must be two-dimensional uint8"
            raise ValueError(msg)
        if not math.isfinite(self.harmonization_ms) or self.harmonization_ms < 0:
            msg = "harmonization time must be finite and non-negative"
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
    background_refresh_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    normalization_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    queue_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    matting_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    post_processing_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    readback_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    capture_to_paint_ms: float = Field(default=0.0, ge=0, le=60_000, allow_inf_nan=False)
    model_preparation_ms: float = Field(default=0.0, ge=0, le=600_000, allow_inf_nan=False)
    seed_initialization_ms: float = Field(default=0.0, ge=0, le=600_000, allow_inf_nan=False)
    first_matte_ms: float = Field(default=0.0, ge=0, le=600_000, allow_inf_nan=False)
    output_width: int = Field(default=1, gt=0, le=8_192)
    output_height: int = Field(default=1, gt=0, le=8_192)


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


def calibration_p95_latency(latencies_ms: list[float]) -> float:
    """Interpolate p95 from enough synchronized samples to reject isolated jitter."""
    if not latencies_ms or any(
        not math.isfinite(latency) or latency < 0 for latency in latencies_ms
    ):
        msg = "calibration latencies must be finite and non-negative"
        raise ValueError(msg)
    ordered = sorted(latencies_ms)
    rank = (len(ordered) - 1) * 0.95
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)
