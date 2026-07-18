"""Asynchronous latest-frame composition coordination."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from better_backgrounds.desktop.live_preview.surface import PreparedComposite

if TYPE_CHECKING:
    from better_backgrounds.desktop.live_preview.surface import NativeCompositeSurface
    from better_backgrounds.matting.engine import CompletedMatte

PRESENTATION_BUFFER_SIZE = 2


class CompositionCoordinator(QObject):
    """Prepare mattes off-thread while retaining only the latest pending frame."""

    buffer_ready = Signal()
    failed = Signal(str)
    _prepared = Signal(int, object)
    _preparation_failed = Signal(int, str)

    def __init__(self, surface: NativeCompositeSurface, parent: QObject | None = None) -> None:
        """Create a coordinator for one retained presentation surface."""
        super().__init__(parent)
        self._surface = surface
        self._pending: CompletedMatte | None = None
        self._ready: deque[PreparedComposite] = deque(maxlen=PRESENTATION_BUFFER_SIZE)
        self._inflight = False
        self._revision = 0
        self._presentation_drops = 0
        self._prepared.connect(self._accept)
        self._preparation_failed.connect(self._reject)

    @property
    def presentation_drops(self) -> int:
        """Return frames replaced before presentation."""
        return self._presentation_drops

    def submit(self, completed: CompletedMatte) -> None:
        """Queue a matte, replacing an older pending matte when preparation is busy."""
        if self._inflight:
            if self._pending is not None:
                self._presentation_drops += 1
            self._pending = completed
            return
        self._start(completed)

    def take_ready(self) -> PreparedComposite | None:
        """Return the oldest prepared frame available for presentation."""
        return self._ready.popleft() if self._ready else None

    def reset(self) -> None:
        """Invalidate background work and discard every queued frame."""
        self._pending = None
        self._ready.clear()
        self._inflight = False
        self._revision += 1
        self._presentation_drops = 0

    def _start(self, completed: CompletedMatte) -> None:
        self._inflight = True
        revision = self._revision

        def compose() -> None:
            try:
                prepared = self._surface.prepare_matte(completed)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._preparation_failed.emit(revision, str(error)[:240])
            else:
                self._prepared.emit(revision, prepared)

        threading.Thread(target=compose, name="live-compositor", daemon=True).start()

    @Slot(int, object)
    def _accept(self, revision: int, prepared: object) -> None:
        if revision != self._revision or not isinstance(prepared, PreparedComposite):
            return
        self._inflight = False
        if len(self._ready) == PRESENTATION_BUFFER_SIZE:
            self._presentation_drops += 1
        self._ready.append(prepared)
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._start(pending)
        if len(self._ready) == PRESENTATION_BUFFER_SIZE:
            self.buffer_ready.emit()

    @Slot(int, str)
    def _reject(self, revision: int, message: str) -> None:
        if revision != self._revision:
            return
        self._inflight = False
        self.failed.emit(message)
