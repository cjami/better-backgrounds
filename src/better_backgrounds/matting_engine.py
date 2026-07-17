"""UI-process facade for the dedicated MatAnyone 2 worker."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING, Protocol

import numpy as np

from better_backgrounds.frame_ring import SharedFrameRing
from better_backgrounds.live_matting import (
    RING_SLOTS,
    FrameMismatchError,
    FramePacket,
    LatestFrameScheduler,
    MatteResult,
    MattingCapabilities,
    MattingConfig,
)
from better_backgrounds.matting_worker import (
    StopWorker,
    WorkerFailure,
    WorkerReady,
    matting_worker_main,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
ALPHA_MIDPOINT = 128
MINIMUM_SEED_OCCUPANCY = 0.01
MAXIMUM_SEED_OCCUPANCY = 0.95


@dataclass(frozen=True, slots=True)
class EngineReady:
    """Publish successful seed warm-up and the effective device."""

    capabilities: MattingCapabilities
    initialization_ms: float
    selected_internal_size: int


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


EngineEvent = EngineReady | CompletedMatte | EngineFailure


class MattingEngine(Protocol):
    """Define the stateful MatAnyone 2 lifecycle used by the live session."""

    def start(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
    ) -> None:
        """Start one worker and initialize its temporal memory."""

    def submit(self, frame: NDArray[np.uint8], *, captured_at: float) -> bool:
        """Accept a frame as active or newest pending work."""

    def poll(self) -> tuple[EngineEvent, ...]:
        """Return currently available non-blocking worker events."""

    def reset(self) -> None:
        """Discard temporal memory and all scheduled frames."""

    def close(self) -> None:
        """Stop the worker and release all shared memory."""


class ProcessMattingEngine:
    """Keep MatAnyone 2 and temporal state outside the UI process."""

    def __init__(self, checkpoint: Path) -> None:
        """Retain the verified checkpoint path until a seed is confirmed."""
        self._checkpoint = checkpoint
        self._context = multiprocessing.get_context("spawn")
        self._commands = None
        self._events = None
        self._process = None
        self._ring: SharedFrameRing | None = None
        self._scheduler = LatestFrameScheduler()
        self._free_slots = set(range(RING_SLOTS))
        self._seed_slot: int | None = None
        self._next_frame_id = 0
        self._ready = False
        self._failure_reported = False
        self.rejected_frames = 0

    @property
    def ready(self) -> bool:
        """Return whether model loading and seed warm-up completed."""
        return self._ready

    @property
    def dropped_frames(self) -> int:
        """Include replaced pending frames and frames rejected without a free slot."""
        return self._scheduler.dropped_frames + self.rejected_frames

    def start(
        self,
        seed_frame: NDArray[np.uint8],
        seed_mask: NDArray[np.uint8],
        config: MattingConfig,
    ) -> None:
        """Allocate the ring and spawn one worker for a confirmed target mask."""
        self.close()
        self._validate_seed(seed_frame, seed_mask)
        height, width = seed_frame.shape[:2]
        self._ring = SharedFrameRing.create(width=width, height=height)
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
        self._commands = self._context.Queue()
        self._events = self._context.Queue()
        self._process = self._context.Process(
            target=matting_worker_main,
            args=(
                self._ring.descriptor,
                self._checkpoint,
                config,
                seed_packet,
                self._commands,
                self._events,
            ),
            name="matanyone2-live-worker",
            daemon=True,
        )
        self._ready = False
        self._failure_reported = False
        self._next_frame_id = 1
        self.rejected_frames = 0
        self._scheduler = LatestFrameScheduler()
        self._process.start()

    def submit(self, frame: NDArray[np.uint8], *, captured_at: float) -> bool:
        """Write one RGB frame and keep it as active or newest pending work."""
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
            self._commands.put(submission.dispatch)
        return True

    def poll(self) -> tuple[EngineEvent, ...]:
        """Drain worker events without blocking the Qt event loop."""
        published: list[EngineEvent] = []
        if self._events is not None:
            while True:
                try:
                    event = self._events.get_nowait()
                except Empty:
                    break
                if isinstance(event, WorkerReady):
                    self._ready = True
                    if self._seed_slot is not None:
                        self._release_slot(self._seed_slot)
                        self._seed_slot = None
                    published.append(
                        EngineReady(
                            capabilities=event.capabilities,
                            initialization_ms=event.initialization_ms,
                            selected_internal_size=event.selected_internal_size,
                        ),
                    )
                elif isinstance(event, MatteResult):
                    completed = self._complete(event)
                    if completed is not None:
                        published.append(completed)
                elif isinstance(event, WorkerFailure):
                    self._ready = False
                    self._failure_reported = True
                    published.append(EngineFailure(event.message))
        if (
            self._process is not None
            and not self._process.is_alive()
            and not self._failure_reported
        ):
            self._failure_reported = True
            self._ready = False
            published.append(EngineFailure("MatAnyone 2 worker stopped unexpectedly"))
        return tuple(published)

    def reset(self) -> None:
        """Discard the current person identity and temporal memory."""
        self.close()

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
                queue.close()
                queue.join_thread()
        self._commands = None
        self._events = None
        self._process = None
        self._ready = False
        self._failure_reported = False
        self._scheduler.reset()
        if self._ring is not None:
            self._ring.close(unlink=True)
            self._ring = None
        self._free_slots = set(range(RING_SLOTS))
        self._seed_slot = None

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
            self._commands.put(completion.dispatch)
        return CompletedMatte(completion.completed, result, source, alpha)

    def _allocate_slot(self) -> int:
        if not self._free_slots:
            msg = "shared frame ring is full"
            raise RuntimeError(msg)
        slot = min(self._free_slots)
        self._free_slots.remove(slot)
        return slot

    def _release_slot(self, slot: int) -> None:
        self._free_slots.add(slot)

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
