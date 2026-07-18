"""Application header and primary tab navigation."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from better_backgrounds.desktop.icon import application_icon

TAB_NAMES = ("Show", "Build", "Adjust", "Compare")


class TabHeader(QFrame):
    """Provide direct navigation between the four product areas."""

    tab_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the brand, tabs, and local-status indicator."""
        super().__init__(parent)
        self.setObjectName("header")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        logo = QLabel()
        logo.setFixedSize(22, 22)
        logo.setPixmap(application_icon().pixmap(22, 22))
        layout.addWidget(logo)
        brand = QLabel("Better Backgrounds")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        divider = QFrame()
        divider.setObjectName("headerDivider")
        divider.setFixedSize(1, 26)
        layout.addSpacing(14)
        layout.addWidget(divider)
        layout.addSpacing(8)
        self._tabs: list[QPushButton] = []
        for index, title in enumerate(TAB_NAMES):
            tab = QPushButton(title)
            tab.setObjectName("tab")
            tab.setAccessibleName(f"Open {title} tab")
            tab.clicked.connect(
                lambda _checked=False, tab_index=index: self.tab_selected.emit(tab_index),
            )
            self._tabs.append(tab)
            layout.addWidget(tab)
        layout.addStretch()
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
        layout.addWidget(room_pill)
        for label, accessible_name in (("?", "Help"), ("⚙︎", "Settings")):
            action = QPushButton(label)
            action.setObjectName("headerIcon")
            action.setAccessibleName(accessible_name)
            layout.addWidget(action)

    def set_active_tab(self, index: int) -> None:
        """Highlight the selected product tab."""
        for tab_index, tab in enumerate(self._tabs):
            tab.setProperty("active", tab_index == index)
            tab.style().unpolish(tab)
            tab.style().polish(tab)

    def set_room(self, room: str) -> None:
        """Show the room shared by the room-dependent tabs."""
        self._room.setText(room)
