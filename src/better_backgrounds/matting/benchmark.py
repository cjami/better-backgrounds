"""Reproducible MatAnyone 2 stateful-step performance benchmark."""

from __future__ import annotations

import platform
import time
from typing import TYPE_CHECKING, Literal, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from better_backgrounds.matting.contracts import INTERNAL_SIZES, InternalSize, MattingConfig
from better_backgrounds.matting.runtime import MatAnyoneRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

MINIMUM_BENCHMARK_FRAMES = 2


class BenchmarkMeasurement(BaseModel):
    """Record synchronized latency and throughput for one internal size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    internal_size: InternalSize
    frames: int = Field(gt=0)
    p50_ms: float = Field(gt=0)
    p95_ms: float = Field(gt=0)
    mattes_per_second: float = Field(gt=0)
    passed: bool


class MattingBenchmarkReport(BaseModel):
    """Describe one machine's complete cross-size benchmark result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    operating_system: str
    architecture: str
    device_type: Literal["cuda", "mps", "cpu"]
    accelerated: bool
    warmup_iterations: int
    latency_budget_ms: float
    minimum_mattes_per_second: float
    selected_internal_size: InternalSize
    passed: bool
    measurements: tuple[BenchmarkMeasurement, ...]


def benchmark_measurement(
    internal_size: int,
    latencies_ms: list[float],
    *,
    budget_ms: float,
    minimum_rate: float = 15.0,
) -> BenchmarkMeasurement:
    """Summarize synchronized timings with the fixed release gate."""
    if internal_size not in INTERNAL_SIZES:
        msg = f"unsupported internal size: {internal_size}"
        raise ValueError(msg)
    if not latencies_ms or any(value <= 0 or not np.isfinite(value) for value in latencies_ms):
        msg = "benchmark requires positive finite frame timings"
        raise ValueError(msg)
    p50 = round(float(np.percentile(latencies_ms, 50)), 2)
    p95 = round(float(np.percentile(latencies_ms, 95)), 2)
    rate = round(1000.0 / float(np.mean(latencies_ms)), 2)
    return BenchmarkMeasurement(
        internal_size=internal_size,
        frames=len(latencies_ms),
        p50_ms=p50,
        p95_ms=p95,
        mattes_per_second=rate,
        passed=p95 <= budget_ms and rate >= minimum_rate,
    )


def load_video_frames(video: Path, *, frame_limit: int) -> list[NDArray[np.uint8]]:
    """Decode a bounded RGB sequence without using the benchmarked worker."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        msg = f"Unable to open benchmark video: {video}"
        raise ValueError(msg)
    frames: list[NDArray[np.uint8]] = []
    try:
        while len(frames) < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(
                cast(
                    "NDArray[np.uint8]",
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                ),
            )
    finally:
        capture.release()
    if len(frames) < MINIMUM_BENCHMARK_FRAMES:
        msg = "Benchmark video must contain at least two frames"
        raise ValueError(msg)
    return frames


def run_matting_benchmark(
    frames: list[NDArray[np.uint8]],
    seed_mask: NDArray[np.uint8],
    checkpoint: Path,
    *,
    requested_device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    warmup_iterations: int = 10,
    budget_ms: float = 66.0,
    minimum_rate: float = 15.0,
) -> MattingBenchmarkReport:
    """Run the same stateful step sequence at all calibrated resolutions."""
    measurements = []
    device_type: Literal["cuda", "mps", "cpu"] | None = None
    accelerated = False
    for size in INTERNAL_SIZES:
        runtime = MatAnyoneRuntime(
            checkpoint,
            MattingConfig(
                device=requested_device,
                internal_size=size,
                warmup_iterations=warmup_iterations,
            ),
        )
        try:
            runtime.initialize(frames[0], seed_mask)
            timings = []
            for frame in frames[1:]:
                runtime.synchronize()
                started = time.perf_counter()
                runtime.step(frame)
                runtime.synchronize()
                timings.append((time.perf_counter() - started) * 1000.0)
            measurements.append(
                benchmark_measurement(
                    size,
                    timings,
                    budget_ms=budget_ms,
                    minimum_rate=minimum_rate,
                ),
            )
            device_type = runtime.capabilities.device_type
            accelerated = runtime.capabilities.accelerated
        finally:
            runtime.close()
    passing = [measurement for measurement in measurements if measurement.passed]
    selected = max(
        (measurement.internal_size for measurement in passing),
        default=INTERNAL_SIZES[0],
    )
    return MattingBenchmarkReport(
        operating_system=platform.system(),
        architecture=platform.machine(),
        device_type=device_type or "cpu",
        accelerated=accelerated,
        warmup_iterations=warmup_iterations,
        latency_budget_ms=budget_ms,
        minimum_mattes_per_second=minimum_rate,
        selected_internal_size=selected,
        passed=accelerated and bool(passing),
        measurements=tuple(measurements),
    )
