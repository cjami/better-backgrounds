"""UI-process facade for the dedicated MatAnyone 2 worker."""

from __future__ import annotations

import multiprocessing
import threading
import time
from dataclasses import dataclass, replace
from queue import Empty
from typing import TYPE_CHECKING, Protocol

import numpy as np

from better_backgrounds.matting.contracts import (
    RING_SLOTS,
    FrameMismatchError,
    FramePacket,
    LatestFrameScheduler,
    LivePipelineConfig,
    MatteResult,
    MattingCapabilities,
    MattingConfig,
    ProcessedFrame,
    ProcessedResult,
)
from better_backgrounds.matting.ring import SharedFrameRing
from better_backgrounds.matting.worker import (
    ConfigureGeometry,
    ConfigureHarmonization,
    ConfigurePresentation,
    InitializeTracking,
    ProcessFrame,
    ResetTracking,
    SetLiveBackground,
    StopWorker,
    WorkerFailure,
    WorkerMatte,
    WorkerPrepared,
    WorkerProcessed,
    WorkerProgress,
    WorkerReady,
    WorkerReset,
    matting_worker_main,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from better_backgrounds.harmonization import HarmonizationSettings
RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
ALPHA_MIDPOINT = 128
MINIMUM_SEED_OCCUPANCY = 0.01
MAXIMUM_SEED_OCCUPANCY = 0.95


@dataclass(frozen=True, slots=True)
class EnginePrepared:
    """Publish successful checkpoint verification and model loading."""

    capabilities: MattingCapabilities
    preparation_ms: float


@dataclass(frozen=True, slots=True)
class EngineProgress:
    """Publish a truthful preparation or initialization stage."""

    stage: str
    message: str
    completed: int = 0
    total: int = 0


@dataclass(frozen=True, slots=True)
class EngineReady:
    """Publish successful seed warm-up and the effective device."""

    capabilities: MattingCapabilities
    initialization_ms: float
    selected_internal_size: int
    real_time_supported: bool = False
    real_time_error: str | None = None


@dataclass(frozen=True, slots=True)
class CompletedMatte:
    """Return copied exact-frame pixels after their shared slot is released."""

    packet: FramePacket
    result: MatteResult
    source: NDArray[np.uint8]
    alpha: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class EngineFailure:
    """Publish one recoverable worker failure to the UI owner."""

    message: str


EngineEvent = (
    EnginePrepared | EngineProgress | EngineReady | CompletedMatte | ProcessedFrame | EngineFailure
)


class MattingEngine(Protocol):
    """Define the stateful MatAnyone 2 lifecycle used by the live session."""

    def prepare(self, config: MattingConfig) -> None:
        """Load and retain model weights without requiring a person seed."""

    def initialize(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
        pipeline_config: LivePipelineConfig | None = None,
    ) -> None:
        """Seed or reseed temporal memory using the retained model."""

    def start(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
        pipeline_config: LivePipelineConfig | None = None,
    ) -> None:
        """Start one worker and initialize its temporal memory."""

    def submit(self, frame: NDArray[np.uint8], *, captured_at: float) -> bool:
        """Accept a frame as active or newest pending work."""

    def poll(self) -> tuple[EngineEvent, ...]:
        """Return currently available non-blocking worker events."""

    def wait(self, timeout: float = 0.1) -> tuple[EngineEvent, ...]:
        """Block briefly until worker output is available."""

    def reset(self) -> None:
        """Discard temporal memory and all scheduled frames."""

    def close(self) -> None:
        """Stop the worker and release all shared memory."""


class ProcessMattingEngine:
    """Keep MatAnyone 2 and temporal state outside the UI process."""

    def __init__(self, checkpoint: Path, calibration_path: Path | None = None) -> None:
        """Retain lightweight paths until the background worker verifies them."""
        self._checkpoint = checkpoint
        self._calibration_path = calibration_path
        self._context = multiprocessing.get_context("spawn")
        self._commands = None
        self._events = None
        self._process = None
        self._ring: SharedFrameRing | None = None
        self._retired_rings: list[tuple[int, SharedFrameRing]] = []
        self._scheduler = LatestFrameScheduler()
        self._free_slots = set(range(RING_SLOTS))
        self._seed_slot: int | None = None
        self._next_frame_id = 0
        self._ready = False
        self._prepared = False
        self._failure_reported = False
        self.rejected_frames = 0
        self._event_lock = threading.RLock()
        self._pipeline_config: LivePipelineConfig | None = None
        self._fused = False
        self._generation = 0

    @property
    def prepared(self) -> bool:
        """Return whether the retained network weights finished loading."""
        return self._prepared

    @property
    def ready(self) -> bool:
        """Return whether model loading and seed warm-up completed."""
        return self._ready

    @property
    def dropped_frames(self) -> int:
        """Include replaced pending frames and frames rejected without a free slot."""
        return self._scheduler.dropped_frames + self.rejected_frames

    @property
    def fused(self) -> bool:
        """Return whether final live outputs are process-owned CUDA results."""
        return self._fused

    def start(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
        pipeline_config: LivePipelineConfig | None = None,
    ) -> None:
        """Prepare the retained worker and initialize one target."""
        self.prepare(config)
        self.initialize(seed_frame, seed_mask, config, pipeline_config)

    def prepare(self, config: MattingConfig) -> None:
        """Spawn checkpoint verification and model loading without camera pixels."""
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        self._commands = self._context.Queue()
        self._events = self._context.Queue()
        self._process = self._context.Process(
            target=matting_worker_main,
            args=(
                self._checkpoint,
                config,
                self._commands,
                self._events,
                self._calibration_path,
            ),
            name="matanyone2-live-worker",
            daemon=True,
        )
        self._ready = False
        self._prepared = False
        self._failure_reported = False
        self._process.start()

    def initialize(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
        pipeline_config: LivePipelineConfig | None = None,
    ) -> None:
        """Attach a fresh camera ring and reseed retained model weights."""
        self._validate_seed(seed_frame, seed_mask)
        self.prepare(config)
        if self._commands is None:
            msg = "MatAnyone 2 worker did not start"
            raise RuntimeError(msg)
        next_generation = self._generation + 1
        if self._ring is not None:
            self._retired_rings.append((next_generation, self._ring))
            self._ring = None
        self._scheduler.reset()
        height, width = seed_frame.shape[:2]
        self._pipeline_config = pipeline_config
        self._ring = SharedFrameRing.create(
            width=width,
            height=height,
            output_width=max(width, round(height * 16 / 9)),
            output_height=height,
        )
        self._free_slots = set(range(RING_SLOTS))
        seed_slot = self._allocate_slot()
        self._seed_slot = seed_slot
        self._ring.write_frame(seed_slot, seed_frame)
        self._ring.write_alpha(seed_slot, seed_mask)
        seed_packet = FramePacket(
            frame_id=0,
            captured_at=time.monotonic() * 1000.0,
            width=width,
            height=height,
            shared_slot=seed_slot,
        )
        self._generation = next_generation
        self._ready = False
        self._failure_reported = False
        self._next_frame_id = 1
        self.rejected_frames = 0
        self._scheduler = LatestFrameScheduler()
        self._commands.put(
            InitializeTracking(
                self._ring.descriptor,
                config,
                seed_packet,
                pipeline_config,
                self._generation,
            ),
        )

    def submit(self, frame: NDArray[np.uint8], *, captured_at: float) -> bool:
        """Write one RGB frame and keep it as active or newest pending work."""
        with self._event_lock:
            return self._submit(frame, captured_at=captured_at)

    def _submit(self, frame: NDArray[np.uint8], *, captured_at: float) -> bool:
        ring = self._ring
        if not self._ready or ring is None or self._commands is None:
            self.rejected_frames += 1
            return False
        expected = (ring.descriptor.height, ring.descriptor.width, 3)
        if frame.dtype != np.uint8 or frame.shape != expected:
            msg = f"live frame must be {expected} uint8 RGB"
            raise ValueError(msg)
        if not self._free_slots:
            self.rejected_frames += 1
            return False
        slot = self._allocate_slot()
        ring.write_frame(slot, frame)
        packet = FramePacket(
            frame_id=self._next_frame_id,
            captured_at=captured_at,
            width=ring.descriptor.width,
            height=ring.descriptor.height,
            shared_slot=slot,
        )
        self._next_frame_id += 1
        submission = self._scheduler.submit(packet)
        if submission.dropped is not None:
            self._release_slot(submission.dropped.shared_slot)
        if submission.dispatch is not None:
            self._commands.put(ProcessFrame(submission.dispatch, self._generation))
        return True

    def poll(self) -> tuple[EngineEvent, ...]:
        """Drain worker events without blocking the Qt event loop."""
        raw_events: list[object] = []
        if self._events is not None:
            while True:
                try:
                    raw_events.append(self._events.get_nowait())
                except Empty:
                    break
        return self._publish_events(raw_events)

    def set_live_background(
        self,
        background: NDArray[np.uint8] | None,
        reference: NDArray[np.uint8] | None,
        *,
        revision: int,
    ) -> None:
        """Send infrequent immutable room evidence to the process-owned pipeline."""
        if revision < 0:
            msg = "background revision must be non-negative"
            raise ValueError(msg)
        if self._commands is not None:
            self._commands.put(SetLiveBackground(background, reference, revision))

    def configure_presentation(
        self,
        *,
        mirrored: bool,
        retain_standard: bool,
        revision: int,
    ) -> None:
        """Update mirroring and Compare output without disturbing temporal identity."""
        if self._commands is not None:
            self._commands.put(ConfigurePresentation(mirrored, retain_standard, revision))
        config = self._pipeline_config
        if config is not None:
            self._pipeline_config = config.model_copy(
                update={
                    "mirrored": mirrored,
                    "retain_standard": retain_standard,
                    "revision": revision,
                },
            )

    def configure_harmonization(self, settings: HarmonizationSettings) -> None:
        """Apply appearance settings in the process that owns both CUDA models."""
        if self._commands is not None:
            self._commands.put(ConfigureHarmonization(settings))

    def configure_geometry(self, config: LivePipelineConfig) -> None:
        """Change the output crop while retaining MatAnyone temporal state."""
        ring = self._ring
        if ring is not None and (
            config.output_width > ring.descriptor.output_width
            or config.output_height > ring.descriptor.output_height
        ):
            msg = "output geometry exceeds the active capture profile"
            raise ValueError(msg)
        current = self._pipeline_config
        self._pipeline_config = (
            config
            if current is None
            else current.model_copy(
                update={
                    "output_width": config.output_width,
                    "output_height": config.output_height,
                    "aspect_ratio": config.aspect_ratio,
                    "revision": config.revision,
                },
            )
        )
        if self._commands is not None:
            self._commands.put(
                ConfigureGeometry(
                    config.output_width,
                    config.output_height,
                    config.aspect_ratio,
                    config.revision,
                ),
            )

    def wait(self, timeout: float = 0.1) -> tuple[EngineEvent, ...]:
        """Block for a worker result so Qt does not need a polling timer."""
        if timeout <= 0:
            msg = "worker wait timeout must be positive"
            raise ValueError(msg)
        events = self._events
        if events is None:
            return ()
        try:
            first = events.get(timeout=timeout)
        except Empty:
            return self._publish_events([])
        raw_events = [first]
        while True:
            try:
                raw_events.append(events.get_nowait())
            except Empty:
                break
        return self._publish_events(raw_events)

    def _publish_events(self, raw_events: list[object]) -> tuple[EngineEvent, ...]:
        published: list[EngineEvent] = []
        with self._event_lock:
            for event in raw_events:
                published.extend(self._publish_event(event))
            if (
                self._process is not None
                and not self._process.is_alive()
                and not self._failure_reported
            ):
                self._failure_reported = True
                self._ready = False
                published.append(EngineFailure("MatAnyone 2 worker stopped unexpectedly"))
        return tuple(published)

    def _publish_event(self, event: object) -> list[EngineEvent]:
        published: list[EngineEvent] = []
        if isinstance(event, WorkerPrepared):
            self._prepared = True
            published.append(EnginePrepared(event.capabilities, event.preparation_ms))
        elif isinstance(event, WorkerProgress):
            if event.generation is None or event.generation == self._generation:
                published.append(
                    EngineProgress(
                        event.stage,
                        event.message,
                        event.completed,
                        event.total,
                    ),
                )
        elif isinstance(event, WorkerReset):
            self._close_retired_rings(through_generation=event.generation)
        elif isinstance(event, WorkerReady):
            published.extend(self._publish_ready(event))
        elif isinstance(event, (WorkerMatte, WorkerProcessed)):
            published.extend(self._publish_result(event))
        elif isinstance(event, WorkerFailure) and (
            event.generation is None or event.generation == self._generation
        ):
            self._ready = False
            self._failure_reported = True
            published.append(EngineFailure(event.message))
        return published

    def _publish_result(self, event: WorkerMatte | WorkerProcessed) -> list[EngineEvent]:
        if event.generation != self._generation:
            return []
        completed = (
            self._complete(event.result)
            if isinstance(event, WorkerMatte)
            else self._complete_processed(event.result)
        )
        return [] if completed is None else [completed]

    def _publish_ready(self, event: WorkerReady) -> list[EngineEvent]:
        self._close_retired_rings(through_generation=event.generation)
        if event.generation != self._generation:
            return []
        self._ready = True
        self._fused = event.fused
        if self._seed_slot is not None:
            self._release_slot(self._seed_slot)
            self._seed_slot = None
        return [
            EngineReady(
                capabilities=event.capabilities,
                initialization_ms=event.initialization_ms,
                selected_internal_size=event.selected_internal_size,
                real_time_supported=event.fused,
                real_time_error=event.real_time_error,
            ),
        ]

    def reset(self) -> None:
        """Discard person identity while retaining the model process and weights."""
        with self._event_lock:
            self._generation += 1
            self._ready = False
            self._fused = False
            self._scheduler.reset()
            if self._ring is not None:
                self._retired_rings.append((self._generation, self._ring))
                self._ring = None
            self._free_slots = set(range(RING_SLOTS))
            self._seed_slot = None
            if self._commands is not None:
                self._commands.put(ResetTracking(self._generation))

    def close(self) -> None:
        """Stop the owned worker and unlink shared-memory blocks."""
        process = self._process
        if process is not None and process.is_alive() and self._commands is not None:
            self._commands.put(StopWorker())
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for queue in (self._commands, self._events):
            if queue is not None:
                # cancel_join_thread() so close()/join_thread() can never block the
                # Qt GUI thread on the queue's feeder flushing to a wedged or dead
                # worker. The graceful StopWorker/join above already had its chance;
                # any undelivered buffered commands are safe to drop during teardown.
                queue.cancel_join_thread()
                queue.close()
                queue.join_thread()
        self._commands = None
        self._events = None
        self._process = None
        self._ready = False
        self._prepared = False
        self._failure_reported = False
        self._scheduler.reset()
        if self._ring is not None:
            self._ring.close(unlink=True)
            self._ring = None
        self._close_retired_rings()
        self._free_slots = set(range(RING_SLOTS))
        self._seed_slot = None
        self._pipeline_config = None
        self._fused = False
        self._generation = 0

    def _complete(self, result: MatteResult) -> CompletedMatte | None:
        ring = self._ring
        if ring is None or self._commands is None:
            return None
        try:
            completion = self._scheduler.complete(result)
        except FrameMismatchError as error:
            self._ready = False
            self._failure_reported = True
            raise RuntimeError(str(error)) from error
        source = ring.read_frame(completion.completed.shared_slot)
        alpha = ring.read_alpha(completion.completed.shared_slot)
        self._release_slot(completion.completed.shared_slot)
        if completion.dispatch is not None:
            self._commands.put(ProcessFrame(completion.dispatch, self._generation))
        return CompletedMatte(completion.completed, result, source, alpha)

    def _complete_processed(self, result: ProcessedResult) -> ProcessedFrame | None:
        ring = self._ring
        config = self._pipeline_config
        if ring is None or self._commands is None or config is None:
            return None
        matte = MatteResult(
            result.frame_id,
            result.captured_at,
            result.alpha_slot,
            result.timings.matting_ms,
        )
        try:
            completion = self._scheduler.complete(matte)
        except FrameMismatchError as error:
            self._ready = False
            self._failure_reported = True
            raise RuntimeError(str(error)) from error
        primary = ring.read_output(
            completion.completed.shared_slot,
            width=result.output_width,
            height=result.output_height,
        )
        standard = (
            ring.read_output(
                completion.completed.shared_slot,
                width=result.output_width,
                height=result.output_height,
                standard=True,
            )
            if result.standard_retained
            else None
        )
        packet = replace(
            completion.completed,
            width=result.output_width,
            height=result.output_height,
        )
        self._release_slot(completion.completed.shared_slot)
        if completion.dispatch is not None:
            self._commands.put(ProcessFrame(completion.dispatch, self._generation))
        return ProcessedFrame(
            packet=packet,
            primary=primary,
            standard=standard,
            mask_preview=result.mask_preview,
            background_revision=result.background_revision,
            occupancy=result.occupancy,
            timings=result.timings,
            pipeline_revision=result.pipeline_revision,
            harmonized=result.harmonized,
            harmonization_ms=result.harmonization_ms,
            harmonization_degraded=result.harmonization_degraded,
        )

    def _allocate_slot(self) -> int:
        if not self._free_slots:
            msg = "shared frame ring is full"
            raise RuntimeError(msg)
        slot = min(self._free_slots)
        self._free_slots.remove(slot)
        return slot

    def _release_slot(self, slot: int) -> None:
        self._free_slots.add(slot)

    def _close_retired_rings(self, *, through_generation: int | None = None) -> None:
        retained = []
        for generation, ring in self._retired_rings:
            if through_generation is None or generation <= through_generation:
                ring.close(unlink=True)
            else:
                retained.append((generation, ring))
        self._retired_rings = retained

    @staticmethod
    def _validate_seed(
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> None:
        if (
            frame.dtype != np.uint8
            or frame.ndim != RGB_DIMENSIONS
            or frame.shape[2] != RGB_CHANNELS
        ):
            msg = "seed frame must be uint8 RGB"
            raise ValueError(msg)
        if mask.dtype != np.uint8 or mask.shape != frame.shape[:2]:
            msg = "seed mask must be uint8 and match the seed frame"
            raise ValueError(msg)
        occupancy = float(np.count_nonzero(mask >= ALPHA_MIDPOINT)) / mask.size
        if not MINIMUM_SEED_OCCUPANCY <= occupancy <= MAXIMUM_SEED_OCCUPANCY:
            msg = "seed mask must contain a plausible foreground person"
            raise ValueError(msg)
