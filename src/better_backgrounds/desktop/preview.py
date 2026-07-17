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
        self.setAccessibleName("Placeholder reconstructed room preview")

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


class ComparisonPreview(QWidget):
    """Paint a draggable original-to-harmonised comparison placeholder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the comparison surface with a centered wipe."""
        super().__init__(parent)
        self._wipe = 52
        self._live_frame = QPixmap()
        self.setMinimumSize(640, 360)
        self.setAccessibleName("Placeholder original and harmonised comparison")

    def set_wipe(self, value: int) -> None:
        """Update the percentage of harmonised output shown."""
        self._wipe = min(100, max(0, value))
        self.update()

    def set_live_frame(self, frame: QPixmap) -> None:
        """Display the current retained-pipeline frame without new capture work."""
        self._live_frame = frame
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002, N802
        """Draw the split comparison."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if not self._live_frame.isNull():
            scaled = self._live_frame.scaled(
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
        self._paint_scene(painter, bounds, warm=False)
        split = bounds.left() + bounds.width() * self._wipe / 100
        painter.save()
        painter.setClipRect(
            QRectF(bounds.left(), bounds.top(), split - bounds.left(), bounds.height()),
        )
        self._paint_scene(painter, bounds, warm=True)
        painter.restore()

        painter.setPen(QPen(QColor("#e0a34a"), 2))
        painter.drawLine(int(split), int(bounds.top()), int(split), int(bounds.bottom()))
        badge_brush = QColor(16, 17, 21, 225)
        left_badge = QRectF(bounds.left() + 16, bounds.top() + 14, 190, 36)
        right_badge = QRectF(bounds.right() - 226, bounds.top() + 14, 210, 36)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(badge_brush)
        painter.drawRoundedRect(left_badge, 10, 10)
        painter.drawRoundedRect(right_badge, 10, 10)
        painter.setPen(QColor("#d2d4d9"))
        painter.drawText(
            left_badge,
            Qt.AlignmentFlag.AlignCenter,
            "ORIGINAL WEBCAM",
        )
        painter.setPen(QColor("#e0a34a"))
        painter.drawText(
            right_badge,
            Qt.AlignmentFlag.AlignCenter,
            "BETTER BACKGROUNDS",
        )
        handle = QRectF(split - 19, bounds.center().y() - 19, 38, 38)
        painter.setPen(QPen(QColor(20, 16, 10, 80), 1))
        painter.setBrush(QColor("#e0a34a"))
        painter.drawEllipse(handle)
        painter.setPen(QColor("#1a1204"))
        painter.drawText(handle, Qt.AlignmentFlag.AlignCenter, "↔")

    @staticmethod
    def _paint_scene(painter: QPainter, bounds: QRectF, *, warm: bool) -> None:
        gradient = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
        gradient.setColorAt(0.0, QColor("#484139" if warm else "#30343b"))
        gradient.setColorAt(1.0, QColor("#171614" if warm else "#15171b"))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#34363d"), 1))
        painter.drawRoundedRect(bounds, 14, 14)
        subject = QRectF(
            bounds.center().x() - 75,
            bounds.top() + bounds.height() * 0.34,
            150,
            bounds.height() * 0.5,
        )
        painter.setBrush(QColor("#d4a583" if warm else "#a88670"))
        painter.setPen(QPen(QColor(255, 220, 190, 80), 1))
        painter.drawRoundedRect(subject, 18, 18)
