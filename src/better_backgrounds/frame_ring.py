"""Fixed shared-memory ring for RGB webcam frames and alpha mattes."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from better_backgrounds.live_matting import RING_SLOTS


@dataclass(frozen=True, slots=True)
class FrameRingDescriptor:
    """Provide the names and shape required by a spawned worker."""

    frame_name: str
    alpha_name: str
    width: int
    height: int
    slots: int = RING_SLOTS


class SharedFrameRing:
    """Own or attach to fixed-size RGB and alpha shared-memory arrays."""

    def __init__(
        self,
        descriptor: FrameRingDescriptor,
        frame_memory: SharedMemory,
        alpha_memory: SharedMemory,
    ) -> None:
        """Create typed views over two already-open shared-memory blocks."""
        self.descriptor = descriptor
        self._frame_memory = frame_memory
        self._alpha_memory = alpha_memory
        self._frames = np.ndarray(
            (descriptor.slots, descriptor.height, descriptor.width, 3),
            dtype=np.uint8,
            buffer=frame_memory.buf,
        )
        self._alphas = np.ndarray(
            (descriptor.slots, descriptor.height, descriptor.width),
            dtype=np.uint8,
            buffer=alpha_memory.buf,
        )

    @classmethod
    def create(cls, *, width: int, height: int) -> SharedFrameRing:
        """Allocate the three-slot ring in the owning process."""
        if width <= 0 or height <= 0:
            msg = "shared frame dimensions must be positive"
            raise ValueError(msg)
        frame_memory = SharedMemory(create=True, size=RING_SLOTS * width * height * 3)
        try:
            alpha_memory = SharedMemory(create=True, size=RING_SLOTS * width * height)
        except BaseException:
            frame_memory.close()
            frame_memory.unlink()
            raise
        descriptor = FrameRingDescriptor(
            frame_name=frame_memory.name,
            alpha_name=alpha_memory.name,
            width=width,
            height=height,
        )
        ring = cls(descriptor, frame_memory, alpha_memory)
        ring._frames.fill(0)
        ring._alphas.fill(0)
        return ring

    @classmethod
    def attach(cls, descriptor: FrameRingDescriptor) -> SharedFrameRing:
        """Attach without registering a second owner for unlinking."""
        frame_memory = SharedMemory(name=descriptor.frame_name, track=False)
        try:
            alpha_memory = SharedMemory(name=descriptor.alpha_name, track=False)
        except BaseException:
            frame_memory.close()
            raise
        return cls(descriptor, frame_memory, alpha_memory)

    def write_frame(self, slot: int, frame: NDArray[np.uint8]) -> None:
        """Copy one contiguous RGB frame into its owned slot."""
        expected = (self.descriptor.height, self.descriptor.width, 3)
        if frame.dtype != np.uint8 or frame.shape != expected:
            msg = f"frame shape must be {expected} uint8"
            raise ValueError(msg)
        self._frames[self._validate_slot(slot)] = frame

    def read_frame(self, slot: int) -> NDArray[np.uint8]:
        """Copy RGB pixels so the slot can immediately be recycled."""
        return self._frames[self._validate_slot(slot)].copy()

    def write_alpha(self, slot: int, alpha: NDArray[np.uint8]) -> None:
        """Copy one full-resolution alpha matte into its matching slot."""
        expected = (self.descriptor.height, self.descriptor.width)
        if alpha.dtype != np.uint8 or alpha.shape != expected:
            msg = f"alpha shape must be {expected} uint8"
            raise ValueError(msg)
        self._alphas[self._validate_slot(slot)] = alpha

    def read_alpha(self, slot: int) -> NDArray[np.uint8]:
        """Copy alpha pixels so the slot can immediately be recycled."""
        return self._alphas[self._validate_slot(slot)].copy()

    def close(self, *, unlink: bool = False) -> None:
        """Close views and optionally remove blocks owned by this process."""
        self._frames = np.empty((0, 0, 0, 3), dtype=np.uint8)
        self._alphas = np.empty((0, 0, 0), dtype=np.uint8)
        self._frame_memory.close()
        self._alpha_memory.close()
        if unlink:
            for memory in (self._frame_memory, self._alpha_memory):
                with suppress(FileNotFoundError):
                    memory.unlink()

    def _validate_slot(self, slot: int) -> int:
        if not 0 <= slot < self.descriptor.slots:
            msg = "slot must be inside the shared frame ring"
            raise ValueError(msg)
        return slot
