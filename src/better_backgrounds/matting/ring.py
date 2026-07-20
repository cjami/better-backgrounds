"""Fixed shared-memory ring for source, matte, and processed live frames."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from better_backgrounds.matting.contracts import RING_SLOTS

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class FrameRingDescriptor:
    """Provide the names and shape required by a spawned worker."""

    frame_name: str
    alpha_name: str
    primary_name: str
    width: int
    height: int
    output_width: int
    output_height: int
    slots: int = RING_SLOTS


class SharedFrameRing:
    """Own or attach to fixed-size RGB and alpha shared-memory arrays."""

    def __init__(
        self,
        descriptor: FrameRingDescriptor,
        frame_memory: SharedMemory,
        alpha_memory: SharedMemory,
        primary_memory: SharedMemory,
    ) -> None:
        """Create typed views over the already-open shared-memory blocks."""
        self.descriptor = descriptor
        self._frame_memory = frame_memory
        self._alpha_memory = alpha_memory
        self._primary_memory = primary_memory
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
        output_shape = (
            descriptor.slots,
            descriptor.output_height,
            descriptor.output_width,
            3,
        )
        self._primary = np.ndarray(output_shape, dtype=np.uint8, buffer=primary_memory.buf)

    @classmethod
    def create(
        cls,
        *,
        width: int,
        height: int,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> SharedFrameRing:
        """Allocate the three-slot ring in the owning process."""
        if width <= 0 or height <= 0:
            msg = "shared frame dimensions must be positive"
            raise ValueError(msg)
        output_width = width if output_width is None else output_width
        output_height = height if output_height is None else output_height
        if output_width <= 0 or output_height <= 0:
            msg = "shared output dimensions must be positive"
            raise ValueError(msg)
        memories: list[SharedMemory] = []
        try:
            frame_memory = SharedMemory(create=True, size=RING_SLOTS * width * height * 3)
            memories.append(frame_memory)
            alpha_memory = SharedMemory(create=True, size=RING_SLOTS * width * height)
            memories.append(alpha_memory)
            output_bytes = RING_SLOTS * output_width * output_height * 3
            primary_memory = SharedMemory(create=True, size=output_bytes)
            memories.append(primary_memory)
        except BaseException:
            for memory in memories:
                memory.close()
                memory.unlink()
            raise
        descriptor = FrameRingDescriptor(
            frame_name=frame_memory.name,
            alpha_name=alpha_memory.name,
            primary_name=primary_memory.name,
            width=width,
            height=height,
            output_width=output_width,
            output_height=output_height,
        )
        ring = cls(
            descriptor,
            frame_memory,
            alpha_memory,
            primary_memory,
        )
        ring._frames.fill(0)
        ring._alphas.fill(0)
        ring._primary.fill(0)
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
        attached = [frame_memory, alpha_memory]
        try:
            primary_memory = SharedMemory(name=descriptor.primary_name, track=False)
            attached.append(primary_memory)
        except BaseException:
            for memory in attached:
                memory.close()
            raise
        return cls(
            descriptor,
            frame_memory,
            alpha_memory,
            primary_memory,
        )

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

    def write_output(
        self,
        slot: int,
        pixels: NDArray[np.uint8],
    ) -> None:
        """Write final composited pixels into the matching frame slot."""
        maximum = (self.descriptor.output_height, self.descriptor.output_width)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != RGB_DIMENSIONS
            or pixels.shape[2] != RGB_CHANNELS
            or pixels.shape[0] > maximum[0]
            or pixels.shape[1] > maximum[1]
        ):
            msg = f"output must fit inside {maximum} uint8 RGB"
            raise ValueError(msg)
        output = self._primary[self._validate_slot(slot)]
        output[: pixels.shape[0], : pixels.shape[1]] = pixels

    def read_output(
        self,
        slot: int,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> NDArray[np.uint8]:
        """Copy one completed output before slot release."""
        width = self.descriptor.output_width if width is None else width
        height = self.descriptor.output_height if height is None else height
        if (
            not 0 < width <= self.descriptor.output_width
            or not 0 < height <= self.descriptor.output_height
        ):
            msg = "output read dimensions must fit inside the shared frame ring"
            raise ValueError(msg)
        return self._primary[self._validate_slot(slot), :height, :width].copy()

    def close(self, *, unlink: bool = False) -> None:
        """Close views and optionally remove blocks owned by this process."""
        self._frames = np.empty((0, 0, 0, 3), dtype=np.uint8)
        self._alphas = np.empty((0, 0, 0), dtype=np.uint8)
        self._primary = np.empty((0, 0, 0, 3), dtype=np.uint8)
        self._frame_memory.close()
        self._alpha_memory.close()
        self._primary_memory.close()
        if unlink:
            for memory in (
                self._frame_memory,
                self._alpha_memory,
                self._primary_memory,
            ):
                with suppress(FileNotFoundError):
                    memory.unlink()

    def _validate_slot(self, slot: int) -> int:
        if not 0 <= slot < self.descriptor.slots:
            msg = "slot must be inside the shared frame ring"
            raise ValueError(msg)
        return slot
