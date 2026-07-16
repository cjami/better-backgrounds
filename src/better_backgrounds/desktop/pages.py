"""Qt Widgets pages for the Phase 2 tabbed desktop shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedLayout,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.preview import ComparisonPreview

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from better_backgrounds.build_session import VideoSelection

STAGE_ORDER = (
    ("validation", "Validating video"),
    ("frame_selection", "Selecting frames"),
    ("camera_estimation", "Estimating camera poses"),
    ("scene_training", "Training spatial scene"),
    ("runtime_conversion", "Preparing runtime scene"),
)


def _label(text: str, *, object_name: str | None = None, word_wrap: bool = False) -> QLabel:
    value = QLabel(text)
    if object_name is not None:
        value.setObjectName(object_name)
    value.setWordWrap(word_wrap)
    return value


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(12)
    return frame, layout


class ShowPage(QWidget):
    """Preview and control the virtual camera for the selected room."""

    room_selected = Signal(str)
    build_requested = Signal()
    camera_changed = Signal(bool)

    def __init__(
        self,
        rooms: Sequence[str],
        preview_factory: Callable[[], QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Create the room picker, feed preview, and camera control."""
        super().__init__(parent)
        self._camera_active = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        feed_column = QWidget()
        feed_layout = QVBoxLayout(feed_column)
        feed_layout.setContentsMargins(22, 22, 22, 22)
        feed_layout.setSpacing(14)
        feed_surface = QFrame()
        feed_surface.setObjectName("feedSurface")
        feed_stack = QStackedLayout(feed_surface)
        feed_stack.setContentsMargins(0, 0, 0, 0)
        feed_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        preview = preview_factory()
        preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        feed_stack.addWidget(preview)

        overlay = QWidget()
        overlay.setObjectName("feedOverlay")
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setSpacing(6)
        feed_header = QHBoxLayout()
        self._feed_title = _label("", object_name="feedBadge")
        feed_header.addWidget(self._feed_title)
        feed_header.addStretch()
        self._feed_status = _label("●  IDLE", object_name="feedBadge")
        self._feed_status.setAccessibleName("Virtual camera status")
        feed_header.addWidget(self._feed_status)
        overlay_layout.addLayout(feed_header)
        overlay_layout.addStretch()
        self._preview_note = _label(
            "Preview only — the virtual camera is off",
            object_name="previewNote",
        )
        overlay_layout.addWidget(
            self._preview_note,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self._preview_hint = _label(
            "Start it to make this feed available in Zoom, Meet, and other apps",
            object_name="muted",
        )
        overlay_layout.addWidget(
            self._preview_hint,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        overlay_layout.addStretch()
        overlay_layout.addWidget(
            _label("1080p  ·  30 fps  ·  scene-aware harmonisation on", object_name="feedMeta"),
        )
        feed_stack.addWidget(overlay)
        feed_stack.setCurrentWidget(overlay)
        feed_layout.addWidget(feed_surface, 1)
        self._camera = QPushButton("●  Start virtual camera")
        self._camera.setObjectName("cameraToggle")
        self._camera.setProperty("active", False)  # noqa: FBT003
        self._camera.setAccessibleName("Start virtual camera")
        self._camera.clicked.connect(self._toggle_camera)
        feed_layout.addWidget(self._camera)
        root.addWidget(feed_column, 1)

        sidebar = QFrame()
        sidebar.setObjectName("roomRail")
        sidebar.setFixedWidth(330)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 20, 16, 20)
        sidebar_layout.setSpacing(12)
        sidebar_header = QHBoxLayout()
        sidebar_header.addWidget(_label("Rooms", object_name="section"))
        sidebar_header.addStretch()
        build = QPushButton("+  New room")
        build.setObjectName("railAction")
        build.clicked.connect(self.build_requested)
        sidebar_header.addWidget(build)
        sidebar_layout.addLayout(sidebar_header)
        self._rooms = QListWidget()
        self._rooms.setObjectName("roomList")
        self._rooms.setAccessibleName("Available rooms")
        self._rooms.currentItemChanged.connect(self._emit_room)
        sidebar_layout.addWidget(self._rooms, 1)
        root.addWidget(sidebar)

        self.set_rooms(rooms)

    @property
    def camera_active(self) -> bool:
        """Return whether the placeholder virtual camera is active."""
        return self._camera_active

    def set_rooms(self, rooms: Sequence[str], selected: str | None = None) -> None:
        """Replace the room list while preserving the requested selection."""
        target = selected or self.current_room
        self._rooms.blockSignals(True)  # noqa: FBT003
        self._rooms.clear()
        for room in rooms:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, room)
            item.setToolTip("Ready to use")
            item.setSizeHint(QSize(0, 82))
            self._rooms.addItem(item)
            self._rooms.setItemWidget(item, self._room_card(room))
        self._rooms.blockSignals(False)  # noqa: FBT003
        self.set_room(target or (rooms[0] if rooms else ""))

    @property
    def current_room(self) -> str:
        """Return the selected room name."""
        item = self._rooms.currentItem()
        return "" if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def set_room(self, room: str) -> None:
        """Select a room without emitting a redundant app-state update."""
        for index in range(self._rooms.count()):
            item = self._rooms.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == room:
                self._rooms.blockSignals(True)  # noqa: FBT003
                self._rooms.setCurrentItem(item)
                self._rooms.blockSignals(False)  # noqa: FBT003
                self._feed_title.setText(f"VIRTUAL CAMERA  ·  {room}")
                return

    def _emit_room(self, current: QListWidgetItem | None) -> None:
        if current is not None:
            room = str(current.data(Qt.ItemDataRole.UserRole))
            self._feed_title.setText(f"VIRTUAL CAMERA  ·  {room}")
            self.room_selected.emit(room)

    def _toggle_camera(self) -> None:
        self._camera_active = not self._camera_active
        if self._camera_active:
            self._feed_status.setText("●  LIVE")
            self._camera.setText("■  Stop virtual camera")
            self._camera.setAccessibleName("Stop virtual camera")
            self._preview_note.setText("Virtual camera is live")
            self._preview_hint.setText("This feed is available to your video apps")
        else:
            self._feed_status.setText("●  IDLE")
            self._camera.setText("●  Start virtual camera")
            self._camera.setAccessibleName("Start virtual camera")
            self._preview_note.setText("Preview only — the virtual camera is off")
            self._preview_hint.setText(
                "Start it to make this feed available in Zoom, Meet, and other apps",
            )
        self._camera.setProperty("active", self._camera_active)
        self._camera.style().unpolish(self._camera)
        self._camera.style().polish(self._camera)
        self.camera_changed.emit(self._camera_active)

    @staticmethod
    def _room_card(room: str) -> QWidget:
        metadata = {
            "Loft — North Window": "Opened just now",
            "Studio — West Wall": "3 days ago",
            "Living room": "Last week",
            "Bookshelf corner": "2 weeks ago",
        }
        card = QWidget()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(10)
        thumbnail = QLabel()
        thumbnail.setObjectName("roomThumbnail")
        thumbnail.setFixedSize(82, 58)
        layout.addWidget(thumbnail)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addStretch()
        text.addWidget(_label(room, object_name="roomName"))
        text.addWidget(_label(metadata.get(room, "Ready to use"), object_name="muted"))
        text.addStretch()
        layout.addLayout(text, 1)
        badge = _label("Ready", object_name="readyBadge")
        badge.setFixedSize(48, 22)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)
        return card


class BuildPage(QWidget):
    """Upload and process a video within one self-contained tab."""

    video_requested = Signal()
    sample_requested = Signal()
    build_requested = Signal(str)
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create upload, review, and processing states."""
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 32, 52, 32)
        self._content = QStackedWidget()
        self._content.setObjectName("buildContent")
        root.addWidget(self._content)
        self._create_upload()
        self._create_review()
        self._create_progress()

    def _create_upload(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addStretch()
        layout.addWidget(
            _label("NEW ROOM", object_name="eyebrow"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(
            _label("Turn a short video into a room", object_name="heroTitle"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        subtitle = _label(
            "Walk slowly around the empty space for 15-30 seconds. We rebuild it into "
            "a 3D scene your camera can sit inside.",
            object_name="subtitle",
            word_wrap=True,
        )
        subtitle.setMaximumWidth(650)
        subtitle.setMinimumHeight(44)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(subtitle, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10)
        drop = QFrame()
        drop.setObjectName("dropCard")
        drop.setMinimumSize(620, 180)
        drop.setMaximumWidth(760)
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(38, 28, 38, 28)
        drop_layout.setSpacing(8)
        drop_layout.addWidget(
            _label("↑", object_name="uploadIcon"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        choose = QPushButton("Drop a video here, or click to choose")
        choose.setObjectName("dropAction")
        choose.clicked.connect(self.video_requested)
        drop_layout.addWidget(choose, alignment=Qt.AlignmentFlag.AlignHCenter)
        drop_layout.addWidget(
            _label("MP4  /  MOV  ·  15-30 s  ·  720-1080p", object_name="feedMeta"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(drop, alignment=Qt.AlignmentFlag.AlignHCenter)
        sample = QPushButton("No footage handy?  Use a sample clip  →")
        sample.setObjectName("sampleAction")
        sample.clicked.connect(self.sample_requested)
        layout.addWidget(sample, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        self._content.addWidget(page)

    def _create_review(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        card, card_layout = _card()
        card.setMaximumWidth(780)
        header = QHBoxLayout()
        self._selection_name = _label("", object_name="title")
        header.addWidget(self._selection_name)
        header.addStretch()
        header.addWidget(_label("HIGH READINESS", object_name="success"))
        card_layout.addLayout(header)
        card_layout.addWidget(
            _label(
                "The capture looks suitable for a room build.",
                object_name="subtitle",
            ),
        )
        diagnostics = (
            ("Duration & resolution", "24 s · 1080p"),
            ("Sharpness", "Good"),
            ("Exposure", "Balanced"),
            ("Camera movement", "Smooth"),
            ("Frame overlap", "86%"),
            ("Moving objects", "Minor curtain movement"),
        )
        for title, value in diagnostics:
            row = QHBoxLayout()
            row.addWidget(_label("✓", object_name="success"))
            row.addWidget(_label(title))
            row.addStretch()
            row.addWidget(_label(value, object_name="muted"))
            card_layout.addLayout(row)
        footer = QHBoxLayout()
        back = QPushButton("Choose another video")
        back.setObjectName("quiet")
        back.clicked.connect(self.show_upload)
        footer.addWidget(back)
        footer.addStretch()
        footer.addWidget(_label("Developer outcome", object_name="muted"))
        self._outcome = QComboBox()
        self._outcome.addItem("Successful build", "success")
        self._outcome.addItem("Recoverable failure", "failure")
        self._outcome.addItem("Forced cancellation", "forced")
        self._outcome.setAccessibleName("Fake worker outcome")
        footer.addWidget(self._outcome)
        build = QPushButton("Build room")
        build.setObjectName("primary")
        build.clicked.connect(self._emit_build)
        footer.addWidget(build)
        card_layout.addLayout(footer)
        layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        self._content.addWidget(page)

    def _create_progress(self) -> None:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setSpacing(22)
        stages_card, stages_layout = _card()
        stages_card.setMaximumWidth(410)
        stages_layout.addWidget(_label("Building your room", object_name="title"))
        stages_layout.addWidget(
            _label("You can use the other tabs while this runs.", object_name="subtitle"),
        )
        self._stage_labels: dict[str, QLabel] = {}
        for key, title in STAGE_ORDER:
            label = _label(f"○  {title}", object_name="stagePending")
            label.setMinimumHeight(34)
            stages_layout.addWidget(label)
            self._stage_labels[key] = label
        stages_layout.addStretch()
        root.addWidget(stages_card, 2)

        detail_card, detail_layout = _card()
        self._status = _label("Preparing the build", object_name="section", word_wrap=True)
        detail_layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setAccessibleName("Room build progress")
        detail_layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setAccessibleName("Build event log")
        detail_layout.addWidget(self._log, 1)
        self._message = _label("", object_name="danger", word_wrap=True)
        self._message.hide()
        detail_layout.addWidget(self._message)
        actions = QHBoxLayout()
        actions.addStretch()
        self._retry = QPushButton("Retry")
        self._retry.setObjectName("primary")
        self._retry.clicked.connect(self.retry_requested)
        self._retry.hide()
        actions.addWidget(self._retry)
        self._new_build = QPushButton("Build another room")
        self._new_build.setObjectName("primary")
        self._new_build.clicked.connect(self.show_upload)
        self._new_build.hide()
        actions.addWidget(self._new_build)
        self._cancel = QPushButton("Cancel build")
        self._cancel.setObjectName("danger")
        self._cancel.clicked.connect(self.cancel_requested)
        actions.addWidget(self._cancel)
        detail_layout.addLayout(actions)
        root.addWidget(detail_card, 3)
        self._content.addWidget(page)

    def show_upload(self) -> None:
        """Show the upload surface."""
        self._content.setCurrentIndex(0)

    def show_review(self, selection: VideoSelection) -> None:
        """Show capture diagnostics for the selected video."""
        suffix = " · prepared sample" if selection.sample else ""
        self._selection_name.setText(f"{selection.display_name}{suffix}")
        self._content.setCurrentIndex(1)

    def reset_progress(self) -> None:
        """Prepare the progress surface for a new job."""
        self._content.setCurrentIndex(2)
        self._status.setText("Preparing the build")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._log.clear()
        self._message.hide()
        self._retry.hide()
        self._new_build.hide()
        self._cancel.show()
        for key, title in STAGE_ORDER:
            label = self._stage_labels[key]
            label.setText(f"○  {title}")
            self._set_label_style(label, "stagePending")

    def set_progress(self, stage: str, progress: float | None, message: str) -> None:
        """Render one validated progress or warning event."""
        self._status.setText(message)
        if progress is None:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(round(progress * 100))
        active_index = next(
            (index for index, item in enumerate(STAGE_ORDER) if item[0] == stage),
            0,
        )
        for index, (key, title) in enumerate(STAGE_ORDER):
            label = self._stage_labels[key]
            if index < active_index:
                label.setText(f"✓  {title}")
                object_name = "stageDone"
            elif index == active_index:
                label.setText(f"●  {title}")
                object_name = "stageActive"
            else:
                label.setText(f"○  {title}")
                object_name = "stagePending"
            self._set_label_style(label, object_name)
        self._log.append(f"> {message}")

    def set_failed(self, message: str, recovery: str | None) -> None:
        """Show a stable user-safe failure and retry action."""
        text = message if recovery is None else f"{message}\n{recovery}"
        self._message.setText(text)
        self._set_label_style(self._message, "danger")
        self._message.show()
        self._retry.show()
        self._cancel.hide()

    def set_completed(self, room_name: str) -> None:
        """Keep a successful build visible without changing tabs."""
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._status.setText("Room ready")
        self._message.setText(f"{room_name} is now selected in Show, Adjust, and Compare.")
        self._set_label_style(self._message, "success")
        self._message.show()
        self._cancel.hide()
        self._new_build.show()
        self._log.append("> Runtime room is ready")
        for key, title in STAGE_ORDER:
            label = self._stage_labels[key]
            label.setText(f"✓  {title}")
            self._set_label_style(label, "stageDone")

    def _emit_build(self) -> None:
        self.build_requested.emit(str(self._outcome.currentData()))

    @staticmethod
    def _set_label_style(label: QLabel, object_name: str) -> None:
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)


class AdjustPage(QWidget):
    """Adjust the current room's viewpoint and presentation settings."""

    def __init__(
        self,
        renderer_factory: Callable[[], QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Create the renderer and Python-owned adjustment controls."""
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(18, 18, 18, 18)
        scene_layout.setSpacing(12)
        overlays = QHBoxLayout()
        for index, name in enumerate(("Depth", "Confidence", "Coverage", "Subject region")):
            chip = QPushButton(name)
            chip.setObjectName("overlayChip")
            chip.setCheckable(True)
            chip.setChecked(index > 1)
            overlays.addWidget(chip)
        overlays.addStretch()
        overlays.addWidget(_label("RECONSTRUCTED VIEWPOINT  ·  SPLAT", object_name="feedBadge"))
        scene_layout.addLayout(overlays)
        renderer = renderer_factory()
        renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scene_layout.addWidget(renderer, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        reset = QPushButton("Reset view")
        reset.setObjectName("quietAction")
        actions.addWidget(reset)
        save = QPushButton("Save settings")
        save.setObjectName("primary")
        actions.addWidget(save)
        scene_layout.addLayout(actions)
        root.addWidget(scene, 1)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setFixedWidth(310)
        controls = QVBoxLayout(inspector)
        controls.setContentsMargins(16, 18, 16, 16)
        controls.setSpacing(7)
        controls.addWidget(_label("Viewpoint", object_name="section"))
        presets = QHBoxLayout()
        for index, name in enumerate(("Eye level", "Low", "High", "Wide")):
            preset = QPushButton(name)
            preset.setObjectName("preset")
            preset.setProperty("active", index == 0)
            presets.addWidget(preset)
        controls.addLayout(presets)
        self._add_slider(controls, "Field of view", 24, 90, 42)
        self._add_slider(controls, "Horizon", -100, 100, -15)
        controls.addSpacing(6)
        controls.addWidget(_label("Subject", object_name="section"))
        self._add_slider(controls, "Depth in scene", 5, 50, 24)
        self._add_slider(controls, "Virtual focus", 5, 50, 26)
        controls.addSpacing(6)
        controls.addWidget(_label("Harmonisation", object_name="section"))
        self._add_slider(controls, "Scene match", 0, 100, 72)
        self._add_slider(controls, "Exposure", 0, 100, 44)
        self._add_slider(controls, "Colour temp", 0, 100, 56)
        self._add_slider(controls, "Colour spill", 0, 100, 34)
        self._add_slider(controls, "Foreground focus", 0, 100, 52)
        self._add_slider(controls, "Edge integration", 0, 100, 66)
        controls.addStretch()
        root.addWidget(inspector)

    def set_room(self, room: str) -> None:
        """Update the room named by the inspector."""
        self.setAccessibleDescription(f"Adjust settings for {room}")

    @staticmethod
    def _add_slider(
        layout: QVBoxLayout,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
    ) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(_label(title, object_name="muted"))
        row.addStretch()
        value_label = _label(AdjustPage._format_slider(title, value), object_name="controlValue")
        row.addWidget(value_label)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setAccessibleName(title)
        slider.valueChanged.connect(
            lambda current, name=title, label=value_label: label.setText(
                AdjustPage._format_slider(name, current),
            ),
        )
        layout.addWidget(slider)
        return slider

    @staticmethod
    def _format_slider(title: str, value: int) -> str:
        if title == "Field of view":
            return f"{value}°"
        if title == "Horizon":
            return f"{value / 10:.1f}°"
        if title in {"Depth in scene", "Virtual focus"}:
            return f"{value / 10:.1f} m"
        return f"{value}%"


class ComparePage(QWidget):
    """Show only the A/B wipe for the selected room."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the Python-painted comparison and wipe control."""
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(12)
        self._preview = ComparisonPreview()
        root.addWidget(self._preview, 1)
        wipe_row = QHBoxLayout()
        wipe_row.addWidget(_label("ORIGINAL WEBCAM", object_name="muted"))
        wipe = QSlider(Qt.Orientation.Horizontal)
        wipe.setRange(0, 100)
        wipe.setValue(52)
        wipe.setAccessibleName("Comparison wipe")
        wipe.valueChanged.connect(self._preview.set_wipe)
        wipe_row.addWidget(wipe, 1)
        wipe_row.addWidget(_label("BETTER BACKGROUNDS", object_name="stageActive"))
        root.addLayout(wipe_row)

    def set_room(self, room: str) -> None:
        """Update the room named by the comparison."""
        self.setAccessibleDescription(f"Compare output for {room}")
