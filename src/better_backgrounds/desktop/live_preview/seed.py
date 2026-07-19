"""Asynchronous person-candidate generation coordination."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from better_backgrounds.matting.seed import MediaPipeSeedProvider

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _SeedRequest:
    revision: int
    frame: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _ReleaseProvider:
    pass


class SeedCoordinator(QObject):
    """Run every MediaPipe request on one retained background thread."""

    generated = Signal(object, object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a lazy provider and one serialized request queue."""
        super().__init__(parent)
        self._active = False
        self._revision = 0
        self._requests: Queue[_SeedRequest | _ReleaseProvider | None] = Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="mediapipe-seed",
            daemon=True,
        )
        self._thread.start()

    @property
    def active(self) -> bool:
        """Return whether a seed request is currently outstanding."""
        return self._active

    def generate(self, frame: NDArray[np.uint8]) -> None:
        """Queue person candidates from one stable camera frame."""
        if self._active:
            return
        self._active = True
        self._requests.put(_SeedRequest(self._revision, frame.copy()))

    def reset(self, *, release_provider: bool = False) -> None:
        """Invalidate outstanding results and optionally unload MediaPipe."""
        self._revision += 1
        self._active = False
        if release_provider:
            self._requests.put(_ReleaseProvider())

    def close(self) -> None:
        """Stop the serialized provider thread and release its native model."""
        self._revision += 1
        self._active = False
        self._requests.put(None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        provider = None
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                if isinstance(request, _ReleaseProvider):
                    if provider is not None:
                        provider.close()
                        provider = None
                    continue
                provider = self._generate(provider, request)
        finally:
            if provider is not None:
                provider.close()

    def _generate(
        self,
        provider: MediaPipeSeedProvider | None,
        request: _SeedRequest,
    ) -> MediaPipeSeedProvider | None:
        try:
            provider = provider or MediaPipeSeedProvider()
            candidates = provider.generate_candidates(request.frame)
        except (OSError, RuntimeError, ValueError) as error:
            if request.revision == self._revision:
                self._active = False
                self.failed.emit(str(error)[:240])
        else:
            if request.revision == self._revision:
                self._active = False
                self.generated.emit(request.frame, candidates)
        return provider
