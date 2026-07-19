"""Shared construction helpers for desktop pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QResizeEvent

MINIMUM_ASPECT_RATIO = 0.5
MAXIMUM_ASPECT_RATIO = 4.0


class AspectRatioContainer(QWidget):
    """Fit one child inside a centered, mutable aspect-ratio viewport."""

    def __init__(
        self,
        child: QWidget,
        *,
        aspect_ratio: float = 16 / 9,
        parent: QWidget | None = None,
    ) -> None:
        """Retain the child and apply the initial output aspect."""
        super().__init__(parent)
        self._child = child
        self._aspect_ratio = 16 / 9
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_aspect_ratio(aspect_ratio)

    @property
    def aspect_ratio(self) -> float:
        """Return the ratio currently used to fit the child."""
        return self._aspect_ratio

    @property
    def viewport_rect(self) -> QRect:
        """Return the child's fitted rectangle for layout tests and diagnostics."""
        return self._child.geometry()

    def set_aspect_ratio(self, aspect_ratio: float) -> None:
        """Update the fitted output ratio without recreating the child surface."""
        if not MINIMUM_ASPECT_RATIO <= aspect_ratio <= MAXIMUM_ASPECT_RATIO:
            msg = "aspect ratio must be between 0.5 and 4.0"
            raise ValueError(msg)
        self._aspect_ratio = aspect_ratio
        self._fit_child()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Keep the child centered whenever the available page area changes."""
        super().resizeEvent(event)
        self._fit_child()

    def _fit_child(self) -> None:
        available_width = self.width()
        available_height = self.height()
        if available_width <= 0 or available_height <= 0:
            return
        width = available_width
        height = round(width / self._aspect_ratio)
        if height > available_height:
            height = available_height
            width = round(height * self._aspect_ratio)
        self._child.setGeometry(
            (available_width - width) // 2,
            (available_height - height) // 2,
            width,
            height,
        )


def label(text: str, *, object_name: str | None = None, word_wrap: bool = False) -> QLabel:
    """Create one consistently configured text label."""
    value = QLabel(text)
    if object_name is not None:
        value.setObjectName(object_name)
    value.setWordWrap(word_wrap)
    return value


def card() -> tuple[QFrame, QVBoxLayout]:
    """Create one standard card and its content layout."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(12)
    return frame, layout
