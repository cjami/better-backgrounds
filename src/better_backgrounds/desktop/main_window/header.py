"""Application header and primary tab navigation."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from better_backgrounds.desktop.icon import application_icon

TAB_NAMES = ("Show", "Build", "Adjust")


class TabHeader(QFrame):
    """Provide centered navigation between the product areas."""

    tab_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the brand, tabs, and local-status indicator."""
        super().__init__(parent)
        self.setObjectName("header")
        layout = QGridLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setHorizontalSpacing(16)
        layout.addWidget(self._create_brand_group(), 0, 0)
        layout.addWidget(self._create_tab_group(), 0, 1)
        layout.addWidget(self._create_room_group(), 0, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 1)

    @staticmethod
    def _create_brand_group() -> QWidget:
        brand_group = QWidget()
        brand_group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        brand_layout = QHBoxLayout(brand_group)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(8)
        logo = QLabel()
        logo.setFixedSize(22, 22)
        logo.setPixmap(application_icon().pixmap(22, 22))
        brand_layout.addWidget(logo)
        brand = QLabel("Better Backgrounds")
        brand.setObjectName("brand")
        brand_layout.addWidget(brand)
        brand_layout.addStretch()
        return brand_group

    def _create_tab_group(self) -> QWidget:
        tabs = QWidget()
        tab_layout = QHBoxLayout(tabs)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(8)
        self._tabs: list[QPushButton] = []
        for index, title in enumerate(TAB_NAMES):
            tab = QPushButton(title)
            tab.setObjectName("tab")
            tab.setCheckable(True)
            tab.setAutoExclusive(True)
            tab.setAccessibleName(f"Open {title} tab")
            tab.clicked.connect(
                lambda _checked=False, tab_index=index: self.tab_selected.emit(tab_index),
            )
            self._tabs.append(tab)
            tab_layout.addWidget(tab)
        return tabs

    def _create_room_group(self) -> QWidget:
        room_group = QWidget()
        room_group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        room_group_layout = QHBoxLayout(room_group)
        room_group_layout.setContentsMargins(0, 0, 0, 0)
        room_group_layout.addStretch()
        room_pill = QFrame()
        room_pill.setObjectName("roomPill")
        room_layout = QHBoxLayout(room_pill)
        room_layout.setContentsMargins(12, 0, 12, 0)
        room_layout.setSpacing(7)
        dot = QLabel("●")
        dot.setObjectName("success")
        room_layout.addWidget(dot)
        self._room = QLabel("No room selected")
        room_layout.addWidget(self._room)
        room_group_layout.addWidget(room_pill)
        return room_group

    def set_active_tab(self, index: int) -> None:
        """Highlight the selected product tab."""
        for tab_index, tab in enumerate(self._tabs):
            tab.setChecked(tab_index == index)
            tab.setProperty("active", tab_index == index)
            tab.style().unpolish(tab)
            tab.style().polish(tab)

    def set_room(self, room: str) -> None:
        """Show the room shared by the room-dependent tabs."""
        self._room.setText(room)
