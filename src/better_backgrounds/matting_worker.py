"""Spawn-safe MatAnyone 2 worker process using the shared frame ring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING

from better_backgrounds.frame_ring import FrameRingDescriptor, SharedFrameRing
from better_backgrounds.live_matting import (
    INTERNAL_SIZES,
    FramePacket,
    MatteResult,
    MattingCapabilities,
    MattingConfig,
    calibration_p95_latency,
    choose_internal_size,
)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkerReady:
    """Report successful model loading and seed warm-up."""

    capabilities: MattingCapabilities
    initialization_ms: float
    selected_internal_size: int


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Report one bounded worker error without a traceback or source pixels."""

    message: str
    frame_id: int | None = None


@dataclass(frozen=True, slots=True)
class StopWorker:
    """Request orderly model and shared-memory teardown."""


WorkerCommand = FramePacket | StopWorker
WorkerEvent = WorkerReady | MatteResult | WorkerFailure


def matting_worker_main(
    descriptor: FrameRingDescriptor,
    checkpoint: Path,
    config: MattingConfig,
    seed_packet: FramePacket,
    commands: Queue,
    events: Queue,
) -> None:
    """Load MatAnyone 2 once, then process sequential exact-frame commands."""
    ring = SharedFrameRing.attach(descriptor)
    runtime = None
    try:
        from better_backgrounds.matanyone_runtime import MatAnyoneRuntime  # noqa: PLC0415

        started = time.perf_counter()
        runtime = MatAnyoneRuntime(checkpoint, config)
        seed_frame = ring.read_frame(seed_packet.shared_slot)
        seed_mask = ring.read_alpha(seed_packet.shared_slot)
        selected_size = config.internal_size
        if config.calibrate:
            latencies: dict[int, float] = {}
            for size in INTERNAL_SIZES:
                trial = config.model_copy(
                    update={
                        "internal_size": size,
                        "calibrate": False,
                    },
                )
                runtime.reconfigure(trial)
                runtime.initialize(seed_frame, seed_mask)
                timings = []
                for _index in range(config.calibration_frames):
                    runtime.synchronize()
                    frame_started = time.perf_counter()
                    runtime.step(seed_frame)
                    runtime.synchronize()
                    timings.append((time.perf_counter() - frame_started) * 1000.0)
                latencies[size] = calibration_p95_latency(timings)
            if runtime.capabilities.accelerated and not any(
                latency <= config.latency_budget_ms for latency in latencies.values()
            ):
                _fail_latency_gate(latencies, budget_ms=config.latency_budget_ms)
            selected_size = choose_internal_size(
                latencies,
                budget_ms=config.latency_budget_ms,
            )
            config = config.model_copy(
                update={"internal_size": selected_size, "calibrate": False},
            )
            runtime.reconfigure(config)
        seed_alpha = runtime.initialize(seed_frame, seed_mask)
        ring.write_alpha(seed_packet.shared_slot, seed_alpha)
        events.put(
            WorkerReady(
                capabilities=runtime.capabilities,
                initialization_ms=(time.perf_counter() - started) * 1000.0,
                selected_internal_size=selected_size,
            ),
        )
        while True:
            try:
                command = commands.get(timeout=0.25)
            except Empty:
                continue
            if isinstance(command, StopWorker):
                return
            if not isinstance(command, FramePacket):
                events.put(WorkerFailure(message="Unknown matting worker command"))
                continue
            inference_started = time.perf_counter()
            try:
                alpha = runtime.step(ring.read_frame(command.shared_slot))
                runtime.synchronize()
                ring.write_alpha(command.shared_slot, alpha)
            except Exception as error:  # noqa: BLE001
                events.put(
                    WorkerFailure(
                        message=_safe_message(error),
                        frame_id=command.frame_id,
                    ),
                )
                return
            events.put(
                MatteResult(
                    frame_id=command.frame_id,
                    captured_at=command.captured_at,
                    alpha_slot=command.shared_slot,
                    inference_ms=(time.perf_counter() - inference_started) * 1000.0,
                ),
            )
    except Exception as error:  # noqa: BLE001
        events.put(WorkerFailure(message=_safe_message(error)))
    finally:
        if runtime is not None:
            runtime.close()
        ring.close()


def _safe_message(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:300]


def _fail_latency_gate(latencies: dict[int, float], *, budget_ms: float) -> None:
    measurements = ", ".join(f"{size}p={latencies[size]:.1f} ms" for size in INTERNAL_SIZES)
    msg = f"MatAnyone 2 missed the {budget_ms:.1f} ms inference gate: {measurements}"
    raise RuntimeError(msg)
