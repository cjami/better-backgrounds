"""Qt Widgets pages for the tabbed desktop application."""

from __future__ import annotations

from collections import Counter
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
from better_backgrounds.scene import CropRegion, SceneReference, Viewpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from better_backgrounds.build_session import VideoSelection
    from better_backgrounds.input_camera import InputCamera

STAGE_ORDER = (
    ("validation", "Validating video"),
    ("frame_selection", "Selecting frames"),
    ("camera_estimation", "Estimating camera poses"),
    ("scene_training", "Training spatial scene"),
    ("runtime_conversion", "Preparing runtime scene"),
)
COMPLETE_PROGRESS = 100


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
    input_camera_selected = Signal(str)
    sample_install_requested = Signal()

    def __init__(
        self,
        rooms: Sequence[str],
        preview_factory: Callable[[], QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Create the room picker, feed preview, and camera control."""
        super().__init__(parent)
        self._camera_active = False
        self._sample_room = ""
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
        self._preview = preview_factory()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        feed_stack.addWidget(self._preview)

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
        sidebar_layout.addWidget(_label("Input camera", object_name="section"))
        self._input_camera = QComboBox()
        self._input_camera.setObjectName("inputCameraSelector")
        self._input_camera.setAccessibleName("Input camera feed")
        self._input_camera.setToolTip("Camera used as the foreground video input")
        self._input_camera.currentIndexChanged.connect(self._emit_input_camera)
        sidebar_layout.addWidget(self._input_camera)
        sidebar_layout.addWidget(
            _label(
                "Select the webcam that will provide the foreground feed.",
                object_name="muted",
                word_wrap=True,
            ),
        )
        sidebar_layout.addSpacing(4)
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

        self._sample_panel = QFrame()
        self._sample_panel.setObjectName("card")
        sample_layout = QVBoxLayout(self._sample_panel)
        sample_layout.setContentsMargins(12, 10, 12, 10)
        sample_layout.setSpacing(6)
        self._sample_status = _label("Sample is not installed", object_name="muted", word_wrap=True)
        sample_layout.addWidget(self._sample_status)
        self._sample_progress = QProgressBar()
        self._sample_progress.setRange(0, 100)
        self._sample_progress.setTextVisible(False)
        self._sample_progress.setAccessibleName("Sample scene download progress")
        self._sample_progress.hide()
        sample_layout.addWidget(self._sample_progress)
        self._sample_attribution = _label("", object_name="muted", word_wrap=True)
        sample_layout.addWidget(self._sample_attribution)
        self._sample_install = QPushButton("Download sample")
        self._sample_install.setObjectName("primary")
        self._sample_install.clicked.connect(self.sample_install_requested)
        sample_layout.addWidget(self._sample_install)
        self._sample_panel.hide()
        sidebar_layout.addWidget(self._sample_panel)
        root.addWidget(sidebar)

        self.set_rooms(rooms)

    @property
    def camera_active(self) -> bool:
        """Return whether the placeholder virtual camera is active."""
        return self._camera_active

    @property
    def current_input_camera_id(self) -> str | None:
        """Return the stable identifier selected in the input-camera control."""
        value = self._input_camera.currentData()
        return value if isinstance(value, str) else None

    def set_input_cameras(
        self,
        cameras: Sequence[InputCamera],
        selected_device_id: str | None,
    ) -> None:
        """Replace the device list while preserving the effective selection."""
        descriptions = Counter(camera.description for camera in cameras)
        occurrences: Counter[str] = Counter()
        self._input_camera.blockSignals(True)  # noqa: FBT003
        self._input_camera.clear()
        for camera in cameras:
            occurrences[camera.description] += 1
            label = camera.description
            if descriptions[camera.description] > 1:
                label = f"{label} ({occurrences[camera.description]})"
            if camera.is_default:
                label = f"{label} · Default"
            self._input_camera.addItem(label, camera.device_id)
        if not cameras:
            self._input_camera.addItem("No camera detected")
            self._input_camera.setEnabled(False)
        else:
            self._input_camera.setEnabled(True)
            selected_index = self._input_camera.findData(selected_device_id)
            self._input_camera.setCurrentIndex(max(0, selected_index))
        self._input_camera.blockSignals(False)  # noqa: FBT003

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
                self._update_sample_panel(room)
                return

    def _emit_room(self, current: QListWidgetItem | None) -> None:
        if current is not None:
            room = str(current.data(Qt.ItemDataRole.UserRole))
            self._feed_title.setText(f"VIRTUAL CAMERA  ·  {room}")
            self._update_sample_panel(room)
            self.room_selected.emit(room)

    def _emit_input_camera(self, _index: int) -> None:
        device_id = self.current_input_camera_id
        if device_id is not None:
            self.input_camera_selected.emit(device_id)

    def configure_sample(
        self,
        room: str,
        *,
        size: int,
        attribution: str,
        installed: bool,
    ) -> None:
        """Describe the prepared sample and its explicit local install action."""
        self._sample_room = room
        self._sample_attribution.setText(f"{attribution} · CC BY 4.0")
        size_megabytes = size / 1024 / 1024
        self._sample_install.setText(f"Download sample ({size_megabytes:.1f} MB)")
        self.set_sample_ready(ready=installed)
        self._update_sample_panel(self.current_room)

    def set_sample_downloading(self, completed: int, total: int) -> None:
        """Show determinate progress for the complete verified asset set."""
        self._sample_progress.show()
        self._sample_progress.setValue(round(completed / total * 100))
        self._sample_status.setText("Downloading and verifying sample…")
        self._sample_install.setEnabled(False)

    def set_sample_ready(self, *, ready: bool) -> None:
        """Show whether the sample is available offline."""
        self._sample_progress.hide()
        self._sample_status.setText("Ready offline" if ready else "Sample is not installed")
        self._sample_install.setVisible(not ready)
        self._sample_install.setEnabled(not ready)

    def set_sample_error(self, message: str) -> None:
        """Show a recoverable sample install failure."""
        self._sample_progress.hide()
        self._sample_status.setText(message)
        self._sample_install.setText("Retry sample download")
        self._sample_install.setEnabled(True)
        self._sample_install.show()

    def set_preview_image(self, path: object | None) -> None:
        """Forward a verified native preview to capable feed widgets."""
        setter = getattr(self._preview, "set_scene_image", None)
        if callable(setter):
            setter(path)

    def _update_sample_panel(self, room: str) -> None:
        self._sample_panel.setVisible(bool(self._sample_room) and room == self._sample_room)

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

    viewpoint_saved = Signal(str, object)

    def __init__(
        self,
        renderer_factory: Callable[[], QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Create the renderer and Python-owned adjustment controls."""
        super().__init__(parent)
        self._room_id = ""
        self._viewpoint = Viewpoint()
        self._default_viewpoint = Viewpoint()
        self._drafts: dict[str, Viewpoint] = {}
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._loaded_scene_id = ""
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(18, 18, 18, 18)
        scene_layout.setSpacing(12)
        overlays = QHBoxLayout()
        for name in ("Depth", "Confidence", "Coverage", "Subject region"):
            chip = QPushButton(name)
            chip.setObjectName("overlayChip")
            chip.setCheckable(True)
            chip.setChecked(name == "Subject region")
            if name != "Subject region":
                chip.setEnabled(False)
                chip.setToolTip("Unavailable until a reliable scene-buffer pass is selected")
            overlays.addWidget(chip)
        overlays.addStretch()
        overlays.addWidget(_label("RECONSTRUCTED VIEWPOINT  ·  SPLAT", object_name="feedBadge"))
        scene_layout.addLayout(overlays)
        self._renderer = renderer_factory()
        self._renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scene_layout.addWidget(self._renderer, 1)
        renderer_progress = getattr(self._renderer, "scene_progressed", None)
        if renderer_progress is not None:
            renderer_progress.connect(self._set_scene_progress)
        renderer_error = getattr(self._renderer, "scene_failed", None)
        if renderer_error is not None:
            renderer_error.connect(self._set_scene_error)
        renderer_viewpoint = getattr(self._renderer, "viewpoint_changed", None)
        if renderer_viewpoint is not None:
            renderer_viewpoint.connect(self._accept_renderer_viewpoint)
        self._scene_status = _label("Select an installed spatial room", object_name="muted")
        scene_layout.addWidget(self._scene_status)
        actions = QHBoxLayout()
        self._attribution = _label("", object_name="muted", word_wrap=True)
        actions.addWidget(self._attribution, 1)
        actions.addStretch()
        reset = QPushButton("Reset view")
        reset.setObjectName("quietAction")
        reset.clicked.connect(self._reset_viewpoint)
        actions.addWidget(reset)
        save = QPushButton("Save settings")
        save.setObjectName("primary")
        save.clicked.connect(self._save_viewpoint)
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
            preset.clicked.connect(
                lambda _checked=False, preset_name=name: self._apply_preset(preset_name),
            )
            presets.addWidget(preset)
        controls.addLayout(presets)
        self._add_slider(controls, "Field of view", 24, 90, 42)
        self._add_slider(controls, "Horizon", -100, 100, -15)
        self._add_slider(controls, "Output crop", 0, 20, 0)
        aspect_row = QHBoxLayout()
        aspect_row.addWidget(_label("Output aspect", object_name="muted"))
        self._aspect = QComboBox()
        self._aspect.setAccessibleName("Output aspect ratio")
        self._aspect.addItem("16:9", 16 / 9)
        self._aspect.addItem("4:3", 4 / 3)
        self._aspect.addItem("1:1", 1.0)
        self._aspect.currentIndexChanged.connect(self._aspect_changed)
        aspect_row.addWidget(self._aspect)
        controls.addLayout(aspect_row)
        controls.addSpacing(6)
        controls.addWidget(_label("Subject", object_name="section"))
        self._add_slider(controls, "Depth in scene", 5, 50, 24)
        focus = self._add_slider(controls, "Virtual focus", 5, 50, 26)
        focus.setEnabled(False)
        focus.setToolTip("Unavailable because this scene has no reliable depth proxy")
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

    def set_room(
        self,
        room: str,
        scene: SceneReference | None = None,
        *,
        installed: bool = False,
        viewpoint: Viewpoint | None = None,
    ) -> None:
        """Restore one room draft and load its managed scene at most once."""
        if self._room_id:
            self._drafts[self._room_id] = self._viewpoint
        self._room_id = room
        self.setAccessibleDescription(f"Adjust settings for {room}")
        self._default_viewpoint = scene.default_viewpoint if scene is not None else Viewpoint()
        self._viewpoint = self._drafts.get(room, viewpoint or self._default_viewpoint)
        if scene is not None:
            self._viewpoint = self._viewpoint.model_copy(
                update={"scene_transform": self._default_viewpoint.scene_transform},
            )
        self._sync_controls()

        if scene is None:
            self._renderer.show()
            self._scene_status.setText("No spatial scene is registered for this room yet")
            self._attribution.clear()
            return
        self._attribution.setText(f"{scene.attribution} · {scene.license_name}")
        if not installed:
            self._renderer.hide()
            self._scene_status.setText("Install this sample in Show to explore its spatial scene")
            return
        self._renderer.show()
        self._scene_status.setText("Loading spatial scene…")
        method_name = "set_viewpoint" if self._loaded_scene_id == scene.asset_id else "set_scene"
        setter = getattr(self._renderer, method_name, None)
        if callable(setter):
            if method_name == "set_scene":
                setter(scene, self._viewpoint)
                self._loaded_scene_id = scene.asset_id
            else:
                setter(self._viewpoint)

    def _set_scene_progress(self, loaded: int, total: int) -> None:
        progress = round(loaded / total * 100)
        message = "Scene ready" if progress == COMPLETE_PROGRESS else f"Loading scene… {progress}%"
        self._scene_status.setText(message)

    def _set_scene_error(self, message: str) -> None:
        self._loaded_scene_id = ""
        self._scene_status.setText(f"Scene unavailable: {message}. Re-select the room to retry.")

    def _accept_renderer_viewpoint(self, viewpoint: object) -> None:
        if isinstance(viewpoint, Viewpoint):
            self._viewpoint = viewpoint
            self._drafts[self._room_id] = viewpoint
            self._sync_controls()

    def _reset_viewpoint(self) -> None:
        self._viewpoint = self._default_viewpoint
        self._drafts[self._room_id] = self._viewpoint
        self._sync_controls()
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(self._viewpoint)
        self._scene_status.setText("View reset to the safe prepared camera")

    def _save_viewpoint(self) -> None:
        if self._room_id:
            self.viewpoint_saved.emit(self._room_id, self._viewpoint)
            self._scene_status.setText("Viewpoint saved for this room")

    def _apply_preset(self, name: str) -> None:
        default = self._default_viewpoint
        if name == "Eye level":
            viewpoint = default
        elif name == "Low":
            position = default.position.model_copy(
                update={"y": max(default.safe_camera_region.minimum.y, default.position.y - 0.6)},
            )
            viewpoint = default.model_copy(update={"position": position})
        elif name == "High":
            position = default.position.model_copy(
                update={"y": min(default.safe_camera_region.maximum.y, default.position.y + 0.6)},
            )
            viewpoint = default.model_copy(update={"position": position})
        else:
            viewpoint = default.model_copy(update={"field_of_view": 90.0})
        self._viewpoint = viewpoint
        self._drafts[self._room_id] = viewpoint
        self._sync_controls()
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(viewpoint)

    def _add_slider(
        self,
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
        self._slider_labels[title] = value_label
        row.addWidget(value_label)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setAccessibleName(title)
        slider.valueChanged.connect(
            lambda current, name=title: self._control_changed(name, current),
        )
        self._sliders[title] = slider
        layout.addWidget(slider)
        return slider

    def _control_changed(self, title: str, value: int) -> None:
        self._slider_labels[title].setText(self._format_slider(title, value))
        if title == "Field of view":
            update: dict[str, object] = {"field_of_view": float(value)}
        elif title == "Horizon":
            update = {"horizon": value / 10}
        elif title == "Depth in scene":
            update = {"subject_depth": value / 10}
        elif title == "Virtual focus":
            update = {"focus_depth": value / 10}
        elif title == "Output crop":
            inset = value / 100
            update = {"crop": CropRegion(left=inset, top=inset, right=1 - inset, bottom=1 - inset)}
        else:
            return
        self._viewpoint = self._viewpoint.model_copy(update=update)
        self._drafts[self._room_id] = self._viewpoint
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(self._viewpoint)

    def _aspect_changed(self, _index: int) -> None:
        self._viewpoint = self._viewpoint.model_copy(
            update={"aspect_ratio": float(self._aspect.currentData())},
        )
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(self._viewpoint)

    def _sync_controls(self) -> None:
        values = {
            "Field of view": round(self._viewpoint.field_of_view),
            "Horizon": round(self._viewpoint.horizon * 10),
            "Output crop": round(self._viewpoint.crop.left * 100),
            "Depth in scene": round(self._viewpoint.subject_depth * 10),
            "Virtual focus": round(self._viewpoint.focus_depth * 10),
        }
        for title, value in values.items():
            slider = self._sliders[title]
            slider.blockSignals(True)  # noqa: FBT003
            slider.setValue(value)
            slider.blockSignals(False)  # noqa: FBT003
        index = min(
            range(self._aspect.count()),
            key=lambda item: abs(float(self._aspect.itemData(item)) - self._viewpoint.aspect_ratio),
        )
        self._aspect.blockSignals(True)  # noqa: FBT003
        self._aspect.setCurrentIndex(index)
        self._aspect.blockSignals(False)  # noqa: FBT003

    @staticmethod
    def _format_slider(title: str, value: int) -> str:
        if title == "Field of view":
            return f"{value}°"
        if title == "Horizon":
            return f"{value / 10:.1f}°"
        if title == "Output crop":
            return f"{value}%"
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
