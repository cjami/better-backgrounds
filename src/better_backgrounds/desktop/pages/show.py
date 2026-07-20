"""Show-page room selection and live-preview controls."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.pages.common import AspectRatioContainer
from better_backgrounds.desktop.pages.common import label as _label
from better_backgrounds.harmonization import HarmonizationSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from better_backgrounds.desktop.camera import InputCamera


class ShowPage(QWidget):
    """Preview and control the local composite for the selected room."""

    room_selected = Signal(str)
    build_requested = Signal()
    camera_changed = Signal(bool)
    preview_restart_requested = Signal()
    input_camera_selected = Signal(str)
    sample_install_requested = Signal()
    seed_confirmed = Signal()
    seed_retry_requested = Signal()
    reseed_requested = Signal()
    person_candidate_selected = Signal(int)
    mirroring_changed = Signal(bool)
    harmonization_changed = Signal(object)

    def __init__(
        self,
        rooms: Sequence[str],
        preview_factory: Callable[[], QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Create the room picker, feed preview, and camera control."""
        super().__init__(parent)
        self._camera_active = False
        self._preview_active = False
        self._sample_room = ""
        self._room_id = ""
        self._harmonization_drafts: dict[str, HarmonizationSettings] = {}
        self._harmonization = HarmonizationSettings(global_harmonization=True)
        self._thumbnail_paths: dict[str, Path] = {}
        self._room_thumbnails: dict[str, QLabel] = {}
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        feed_column = QWidget()
        feed_layout = QVBoxLayout(feed_column)
        feed_layout.setContentsMargins(22, 22, 22, 22)
        feed_layout.setSpacing(14)
        feed_surface = QFrame()
        feed_surface.setObjectName("feedSurface")
        self._feed_stack = QStackedLayout(feed_surface)
        self._feed_stack.setContentsMargins(0, 0, 0, 0)
        self._feed_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._preview = preview_factory()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._feed_stack.addWidget(self._preview)

        overlay = QWidget()
        overlay.setObjectName("feedOverlay")
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setSpacing(6)
        overlay_layout.addStretch()
        self._preview_note = _label(
            "Camera is off",
            object_name="previewNote",
        )
        overlay_layout.addWidget(
            self._preview_note,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self._preview_hint = _label(
            "Choose a camera to begin",
            object_name="muted",
        )
        overlay_layout.addWidget(
            self._preview_hint,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        overlay_layout.addStretch()
        self._overlay = overlay
        self._feed_stack.addWidget(overlay)
        self._feed_stack.setCurrentWidget(overlay)
        self._aspect_preview = AspectRatioContainer(feed_surface)
        self._aspect_preview.setObjectName("showAspectPreview")
        feed_layout.addWidget(self._aspect_preview, 1)
        self._camera = QPushButton("Start virtual camera")
        self._camera.setObjectName("cameraToggle")
        self._camera.setCheckable(True)
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
        camera_header = QHBoxLayout()
        camera_header.addWidget(_label("Camera", object_name="section"))
        camera_header.addStretch()
        restart_preview = QPushButton("Restart")
        restart_preview.setObjectName("railAction")
        restart_preview.setToolTip("Restart the webcam preview")
        restart_preview.clicked.connect(self.preview_restart_requested)
        camera_header.addWidget(restart_preview)
        sidebar_layout.addLayout(camera_header)
        self._input_camera = QComboBox()
        self._input_camera.setObjectName("inputCameraSelector")
        self._input_camera.setAccessibleName("Input camera feed")
        self._input_camera.setToolTip("Camera used as the foreground video input")
        self._input_camera.currentIndexChanged.connect(self._emit_input_camera)
        sidebar_layout.addWidget(self._input_camera)

        self._mirrored = QCheckBox("Mirror webcam")
        self._mirrored.setObjectName("mirrorWebcam")
        self._mirrored.setChecked(True)
        self._mirrored.setToolTip("Mirror only your webcam image; the room stays unchanged")
        self._mirrored.toggled.connect(self.mirroring_changed)
        sidebar_layout.addWidget(self._mirrored)

        self._harmonise_subject = QCheckBox("Harmonise subject")
        self._harmonise_subject.setObjectName("harmonization-global-harmonization")
        self._harmonise_subject.setChecked(True)
        self._harmonise_subject.setToolTip(
            "Match your webcam lighting and colour to the selected room",
        )
        self._harmonise_subject.toggled.connect(self._harmonization_control_changed)
        sidebar_layout.addWidget(self._harmonise_subject)
        self._seed_controls = QWidget()
        seed_layout = QHBoxLayout(self._seed_controls)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(7)
        self._confirm_seed = QPushButton("Confirm person")
        self._confirm_seed.setObjectName("primary")
        self._confirm_seed.clicked.connect(self.seed_confirmed)
        seed_layout.addWidget(self._confirm_seed)
        self._retry_seed = QPushButton("Retry")
        self._retry_seed.setObjectName("quietAction")
        self._retry_seed.clicked.connect(self.seed_retry_requested)
        seed_layout.addWidget(self._retry_seed)
        self._seed_controls.hide()
        sidebar_layout.addWidget(self._seed_controls)
        self._candidate_controls = QWidget()
        candidate_layout = QHBoxLayout(self._candidate_controls)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.setSpacing(7)
        self._candidate_buttons: list[QPushButton] = []
        for candidate_id in range(1, 5):
            button = QPushButton(f"Person {candidate_id}")
            button.setObjectName("quietAction")
            button.setAccessibleName(f"Select person {candidate_id}")
            button.clicked.connect(
                lambda _checked=False, selected=candidate_id: self.person_candidate_selected.emit(
                    selected,
                ),
            )
            candidate_layout.addWidget(button)
            self._candidate_buttons.append(button)
        self._candidate_controls.hide()
        sidebar_layout.addWidget(self._candidate_controls)
        self._reselect_person = QPushButton("Re-select person")
        self._reselect_person.setObjectName("quietAction")
        self._reselect_person.clicked.connect(self.reseed_requested)
        self._reselect_person.hide()
        sidebar_layout.addWidget(self._reselect_person)
        divider = QFrame()
        divider.setObjectName("settingsDivider")
        divider.setFixedHeight(1)
        sidebar_layout.addWidget(divider)
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
        """Return whether virtual-camera publication has been requested."""
        return self._camera_active

    @property
    def preview_active(self) -> bool:
        """Return whether the local webcam preview is active."""
        return self._preview_active

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
            if not self._camera_active:
                self._camera.setEnabled(False)
        else:
            self._input_camera.setEnabled(True)
            self._camera.setEnabled(True)
            selected_index = self._input_camera.findData(selected_device_id)
            self._input_camera.setCurrentIndex(max(0, selected_index))
        self._input_camera.blockSignals(False)  # noqa: FBT003

    def set_rooms(self, rooms: Sequence[str], selected: str | None = None) -> None:
        """Replace the room list while preserving the requested selection."""
        target = selected or self.current_room
        self._rooms.blockSignals(True)  # noqa: FBT003
        self._rooms.clear()
        self._room_thumbnails.clear()
        for room in rooms:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, room)
            item.setSizeHint(QSize(0, 66))
            self._rooms.addItem(item)
            self._rooms.setItemWidget(item, self._room_card(room))
        self._rooms.blockSignals(False)  # noqa: FBT003
        self.set_room(target or (rooms[0] if rooms else ""))

    def set_room_thumbnails(self, thumbnails: Mapping[str, Path | None]) -> None:
        """Show verified room images in the room picker."""
        for room, path in thumbnails.items():
            self.set_room_thumbnail(room, path)

    def set_room_thumbnail(self, room: str, path: Path | None) -> None:
        """Update one room card when its preview or saved view becomes available."""
        if path is None:
            self._thumbnail_paths.pop(room, None)
        else:
            self._thumbnail_paths[room] = path
        self._apply_room_thumbnail(room)

    def set_output_aspect_ratio(self, aspect_ratio: float) -> None:
        """Match the visible feed viewport to the selected room output."""
        self._aspect_preview.set_aspect_ratio(aspect_ratio)

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
                self._update_sample_panel(room)
                self._set_live_room(room)
                return

    def _emit_room(self, current: QListWidgetItem | None) -> None:
        if current is not None:
            room = str(current.data(Qt.ItemDataRole.UserRole))
            self._update_sample_panel(room)
            self._set_live_room(room)
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

    def set_camera_state(self, state: str, message: str) -> None:
        """Reflect the native preview and one-time target-selection lifecycle."""
        self._preview_active = state in {
            "starting",
            "preparing",
            "seeding",
            "seed-error",
            "seed-ready",
            "choose-person",
            "initializing",
            "live",
            "lost",
        }
        self._preview_note.setText("" if state == "live" else message)
        self._preview_note.setVisible(state != "live")
        self._preview_hint.setVisible(state != "live")
        self._seed_controls.setVisible(state in {"seed-ready", "seed-error"})
        self._confirm_seed.setVisible(state == "seed-ready")
        self._candidate_controls.setVisible(state == "choose-person")
        self._reselect_person.setVisible(state in {"live", "lost"})
        self._reselect_person.setText(
            "Find person again" if state == "lost" else "Re-select person",
        )
        if state == "live":
            self._preview_hint.clear()
        elif state in {"preparing", "seeding", "seed-ready", "choose-person", "initializing"}:
            self._preview_hint.setText("Webcam frames stay on this device")
        elif state == "seed-error":
            self._preview_hint.setText("Move into a clearer position, then retry person selection")
        elif state == "lost":
            self._preview_hint.setText("Choose yourself again to continue")
        elif state == "error":
            self._preview_hint.setText("Check permission or device availability, then restart")
        else:
            self._preview_hint.setText("Choose a camera to begin")

    def set_person_candidates(self, count: int) -> None:
        """Expose accessible alternatives to selecting an outlined person."""
        for index, button in enumerate(self._candidate_buttons, start=1):
            button.setVisible(index <= count)

    def _update_sample_panel(self, room: str) -> None:
        self._sample_panel.setVisible(bool(self._sample_room) and room == self._sample_room)

    def _toggle_camera(self) -> None:
        requested = self._camera.isChecked()
        self._camera_active = requested
        self._camera.setText(
            "Stop virtual camera" if requested else "Start virtual camera",
        )
        self._camera.setAccessibleName(
            "Stop virtual camera" if requested else "Start virtual camera",
        )
        self._camera.setProperty("active", requested)
        self._camera.style().unpolish(self._camera)
        self._camera.style().polish(self._camera)
        self.camera_changed.emit(requested)

    def set_live_preferences(self, *, mirrored: bool) -> None:
        """Restore foreground presentation without emitting changes."""
        self._mirrored.blockSignals(True)  # noqa: FBT003
        self._mirrored.setChecked(mirrored)
        self._mirrored.blockSignals(False)  # noqa: FBT003

    def _set_live_room(self, room: str) -> None:
        if self._room_id and self._room_id != room:
            self._harmonization_drafts[self._room_id] = self._harmonization
        self._room_id = room
        self._harmonization = self._harmonization_drafts.get(
            room,
            HarmonizationSettings(global_harmonization=True),
        )
        self._harmonise_subject.blockSignals(True)  # noqa: FBT003
        self._harmonise_subject.setChecked(self._harmonization.global_harmonization)
        self._harmonise_subject.blockSignals(False)  # noqa: FBT003
        self.harmonization_changed.emit(self._harmonization)

    def _harmonization_control_changed(self, enabled: bool) -> None:  # noqa: FBT001
        self._harmonization = HarmonizationSettings(global_harmonization=enabled)
        if self._room_id:
            self._harmonization_drafts[self._room_id] = self._harmonization
        self.harmonization_changed.emit(self._harmonization)

    def _room_card(self, room: str) -> QWidget:
        card = QWidget()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(10)
        thumbnail = QLabel()
        thumbnail.setObjectName("roomThumbnail")
        thumbnail.setFixedSize(70, 48)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail.setAccessibleName(f"{room} thumbnail")
        self._room_thumbnails[room] = thumbnail
        self._apply_room_thumbnail(room)
        layout.addWidget(thumbnail)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addStretch()
        text.addWidget(_label(room, object_name="roomName"))
        text.addStretch()
        layout.addLayout(text, 1)
        return card

    def _apply_room_thumbnail(self, room: str) -> None:
        thumbnail = self._room_thumbnails.get(room)
        if thumbnail is None:
            return
        path = self._thumbnail_paths.get(room)
        pixmap = QPixmap() if path is None else QPixmap(str(path))
        if pixmap.isNull():
            thumbnail.clear()
            thumbnail.setText("⌂")
            return
        scaled = pixmap.scaled(
            thumbnail.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - thumbnail.width()) // 2)
        y = max(0, (scaled.height() - thumbnail.height()) // 2)
        thumbnail.setText("")
        thumbnail.setPixmap(scaled.copy(x, y, thumbnail.width(), thumbnail.height()))
