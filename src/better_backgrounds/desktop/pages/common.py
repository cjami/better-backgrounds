"""Shared construction helpers for desktop pages."""

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


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
