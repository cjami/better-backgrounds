"""Lightweight Python-painted placeholders for later live render surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from pathlib import Path


class ScenePreview(QWidget):
    """Paint a warm room and subject placeholder without media behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an accessible minimum-size scene surface."""
        super().__init__(parent)
        self._scene_image = QPixmap()
        self.setMinimumSize(480, 300)
        self.setAccessibleName("Placeholder spatial room preview")

    def set_scene_image(self, path: Path | None) -> None:
        """Display a verified cached room preview or the painted fallback."""
        self._scene_image = QPixmap() if path is None else QPixmap(str(path))
        name = (
            "Reconstructed room preview"
            if not self._scene_image.isNull()
            else "Room preview unavailable"
        )
        self.setAccessibleName(name)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002, N802
        """Draw the Phase 2 scene placeholder."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if not self._scene_image.isNull():
            scaled = self._scene_image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            source = QRect(
                max(0, (scaled.width() - self.width()) // 2),
                max(0, (scaled.height() - self.height()) // 2),
                self.width(),
                self.height(),
            )
            painter.drawPixmap(self.rect(), scaled, source)
            return
        room = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
        room.setColorAt(0.0, QColor("#45423c"))
        room.setColorAt(0.52, QColor("#282722"))
        room.setColorAt(1.0, QColor("#17181b"))
        painter.setBrush(room)
        painter.setPen(QPen(QColor("#34363d"), 1))
        painter.drawRoundedRect(bounds, 14, 14)

        glow = QRadialGradient(bounds.width() * 0.72, bounds.height() * 0.16, bounds.width() * 0.5)
        glow.setColorAt(0.0, QColor(255, 238, 208, 130))
        glow.setColorAt(1.0, QColor(255, 238, 208, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRoundedRect(bounds, 14, 14)

        subject_width = min(150.0, bounds.width() * 0.2)
        subject = QRectF(
            bounds.center().x() - subject_width / 2,
            bounds.top() + bounds.height() * 0.34,
            subject_width,
            bounds.height() * 0.5,
        )
        skin = QLinearGradient(subject.topLeft(), subject.bottomRight())
        skin.setColorAt(0.0, QColor("#d9ac86"))
        skin.setColorAt(1.0, QColor("#9e7355"))
        painter.setBrush(skin)
        painter.setPen(QPen(QColor(255, 220, 190, 70), 1))
        painter.drawRoundedRect(subject, 18, 18)
