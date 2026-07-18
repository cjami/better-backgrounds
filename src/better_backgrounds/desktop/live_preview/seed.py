"""Asynchronous person-seed generation coordination."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from better_backgrounds.matting.seed import MediaPipeSeedProvider

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class SeedCoordinator(QObject):
    """Own the lifetime and concurrency of MediaPipe seed generation."""

    generated = Signal(object, object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle seed coordinator."""
        super().__init__(parent)
        self._active = False
        self._revision = 0

    @property
    def active(self) -> bool:
        """Return whether a seed provider is currently running."""
        return self._active

    def generate(self, frame: NDArray[np.uint8]) -> None:
        """Generate a person mask for one stable camera frame."""
        if self._active:
            return
        self._active = True
        revision = self._revision

        def generate_seed() -> None:
            provider = None
            try:
                provider = MediaPipeSeedProvider()
                mask = provider.generate(frame)
            except (OSError, RuntimeError, ValueError) as error:
                if revision == self._revision:
                    self._active = False
                    self.failed.emit(str(error)[:240])
            else:
                if revision == self._revision:
                    self._active = False
                    self.generated.emit(frame, mask)
            finally:
                if provider is not None:
                    provider.close()

        threading.Thread(target=generate_seed, name="mediapipe-seed", daemon=True).start()

    def reset(self) -> None:
        """Invalidate an outstanding result and allow a new generation."""
        self._revision += 1
        self._active = False
