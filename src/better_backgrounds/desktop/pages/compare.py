"""Compare-page exact-frame A/B wipe."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.pages.common import label as _label
from better_backgrounds.desktop.preview import ComparisonPreview

COMPARISON_FRAME_COUNT = 2


class ComparePage(QWidget):
    """Show only the A/B wipe for the selected room."""

    wipe_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the Python-painted comparison and wipe control."""
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(12)
        self._preview_host = QWidget()
        self._preview_layout = QVBoxLayout(self._preview_host)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = ComparisonPreview()
        self._preview_layout.addWidget(self._placeholder)
        root.addWidget(self._preview_host, 1)
        self._harmonization_status = _label(
            "Global harmonization is off; identical comparison sides are expected.",
            object_name="muted",
            word_wrap=True,
        )
        self._harmonization_status.setAccessibleName("Harmonisation comparison status")
        root.addWidget(self._harmonization_status)
        wipe_row = QHBoxLayout()
        wipe_row.addWidget(_label("STANDARD EXACT-FRAME", object_name="muted"))
        wipe = QSlider(Qt.Orientation.Horizontal)
        wipe.setRange(0, 100)
        wipe.setValue(52)
        wipe.setAccessibleName("Comparison wipe")
        wipe.valueChanged.connect(self._placeholder.set_wipe)
        wipe.valueChanged.connect(self.wipe_changed)
        wipe_row.addWidget(wipe, 1)
        wipe_row.addWidget(_label("BETTER BACKGROUNDS", object_name="stageActive"))
        root.addLayout(wipe_row)

    def set_live_frame(self, frame: object) -> None:
        """Show retained standard and enhanced images without recapturing the widget."""
        if not isinstance(frame, tuple) or len(frame) != COMPARISON_FRAME_COUNT:
            return
        standard, enhanced = frame
        if isinstance(standard, QImage) and isinstance(enhanced, QImage):
            self._placeholder.show()
            self._placeholder.set_live_frames(standard, enhanced)

    def set_room(self, room: str) -> None:
        """Update the room named by the comparison."""
        self.setAccessibleDescription(f"Compare output for {room}")

    def set_harmonization_status(self, message: str) -> None:
        """Explain identical output or reference-path performance explicitly."""
        self._harmonization_status.setText(message[:300])
