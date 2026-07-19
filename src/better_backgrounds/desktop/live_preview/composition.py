"""Asynchronous latest-frame composition coordination."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from better_backgrounds.desktop.live_preview.surface import PreparedComposite

if TYPE_CHECKING:
    from better_backgrounds.desktop.live_preview.surface import NativeCompositeSurface
    from better_backgrounds.matting.engine import CompletedMatte


class CompositionCoordinator(QObject):
    """Prepare mattes off-thread while retaining only the latest pending frame."""

    frame_ready = Signal()
    failed = Signal(str)
    _prepared = Signal(int, object)
    _preparation_failed = Signal(int, str)

    def __init__(self, surface: NativeCompositeSurface, parent: QObject | None = None) -> None:
        """Create a coordinator for one retained presentation surface."""
        super().__init__(parent)
        self._surface = surface
        self._pending: CompletedMatte | None = None
        self._ready: PreparedComposite | None = None
        self._inflight = False
        self._revision = 0
        self._presentation_drops = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-compositor")
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
        """Return the newest prepared frame available for immediate presentation."""
        prepared = self._ready
        self._ready = None
        return prepared

    def reset(self) -> None:
        """Invalidate background work and discard every queued frame."""
        self._pending = None
        self._ready = None
        self._inflight = False
        self._revision += 1
        self._presentation_drops = 0

    def close(self) -> None:
        """Stop accepting preparation work without blocking Qt shutdown."""
        self.reset()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start(self, completed: CompletedMatte) -> None:
        self._inflight = True
        revision = self._revision

        def compose() -> PreparedComposite:
            return self._surface.prepare_matte(completed)

        future = self._executor.submit(compose)

        def completed_future(result: Future[PreparedComposite]) -> None:
            try:
                prepared = result.result()
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._preparation_failed.emit(revision, str(error)[:240])
            else:
                self._prepared.emit(revision, prepared)

        future.add_done_callback(completed_future)

    @Slot(int, object)
    def _accept(self, revision: int, prepared: object) -> None:
        if revision != self._revision or not isinstance(prepared, PreparedComposite):
            return
        self._inflight = False
        if self._ready is not None:
            self._presentation_drops += 1
        self._ready = prepared
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._start(pending)
        self.frame_ready.emit()

    @Slot(int, str)
    def _reject(self, revision: int, message: str) -> None:
        if revision != self._revision:
            return
        self._inflight = False
        self.failed.emit(message)
