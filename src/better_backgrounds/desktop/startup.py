"""Immediate progressive startup presentation for the desktop shell."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen, QWidget


@dataclass(frozen=True, slots=True)
class StartupStage:
    """Describe one truthful application startup transition."""

    key: str
    message: str
    elapsed_ms: float


class StartupCoordinator(QObject):
    """Show startup progress until the interactive shell is visible."""

    stage_changed = Signal(object)
    failed = Signal(str)

    def __init__(self, application: QApplication) -> None:
        """Create an unshown splash and monotonic stage recorder."""
        super().__init__(application)
        self._application = application
        self._started_at = time.perf_counter()
        self._splash = QSplashScreen(self._pixmap())
        self._splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self._stages: list[StartupStage] = []

    @property
    def stages(self) -> tuple[StartupStage, ...]:
        """Return startup timing evidence for diagnostics and tests."""
        return tuple(self._stages)

    def show(self) -> None:
        """Paint the splash before constructing heavyweight application views."""
        self._splash.show()
        self.advance("starting", "Starting application")

    def advance(self, key: str, message: str) -> None:
        """Publish and paint one named startup stage."""
        stage = StartupStage(
            key,
            message,
            (time.perf_counter() - self._started_at) * 1_000.0,
        )
        self._stages.append(stage)
        self._splash.showMessage(
            message,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            QColor("#ebecef"),
        )
        self.stage_changed.emit(stage)
        self._application.processEvents()

    def finish(self, window: QWidget) -> None:
        """Dismiss startup presentation once the shell can receive input."""
        self.advance("interactive", "Opening camera preview")
        self._splash.finish(window)

    @staticmethod
    def _pixmap() -> QPixmap:
        pixmap = QPixmap(520, 260)
        pixmap.fill(QColor("#111318"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#e0a34a"))
        painter.setFont(QFont("Sans Serif", 24, QFont.Weight.DemiBold))
        painter.drawText(
            pixmap.rect().adjusted(0, 45, 0, -90),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Better Backgrounds",
        )
        painter.setPen(QColor("#9b9fa8"))
        painter.setFont(QFont("Sans Serif", 10))
        painter.drawText(
            pixmap.rect().adjusted(0, 120, 0, -55),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Local room reconstruction and live compositing",
        )
        painter.end()
        return pixmap
