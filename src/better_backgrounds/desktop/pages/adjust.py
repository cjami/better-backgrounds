"""Adjust-page room viewpoint and presentation controls."""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.pages.common import AspectRatioContainer
from better_backgrounds.desktop.pages.common import label as _label
from better_backgrounds.scene import CropRegion, SceneReference, Viewpoint

if TYPE_CHECKING:
    from collections.abc import Callable

COMPLETE_PROGRESS = 100
AUTOSAVE_DELAY_MS = 500
AUTOSAVE_RETRY_MS = 2_000


class AdjustPage(QWidget):
    """Adjust the current room's viewpoint and presentation settings."""

    viewpoint_saved = Signal(str, object)
    viewpoint_previewed = Signal(object)
    snapshot_generated = Signal(str, str, str, object, object)

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
        self._scene: SceneReference | None = None
        self._installed = False
        self._resource_active = True
        self._scene_ready = False
        self._dirty_rooms: set[str] = set()
        self._retried_rooms: set[str] = set()
        self._snapshot_requests: dict[str, tuple[str, str, Viewpoint]] = {}
        self._latest_snapshot_by_room: dict[str, str] = {}
        self._spatial_depth_available = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._flush_current_autosave)
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._flush_current_autosave)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(18, 18, 18, 18)
        scene_layout.setSpacing(12)
        scene_layout.addWidget(_label("Room view", object_name="section"))
        self._renderer = renderer_factory()
        self._renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._aspect_preview = AspectRatioContainer(self._renderer)
        self._aspect_preview.setObjectName("adjustAspectPreview")
        scene_layout.addWidget(self._aspect_preview, 1)
        renderer_progress = getattr(self._renderer, "scene_progressed", None)
        if renderer_progress is not None:
            renderer_progress.connect(self._set_scene_progress)
        renderer_error = getattr(self._renderer, "scene_failed", None)
        if renderer_error is not None:
            renderer_error.connect(self._set_scene_error)
        renderer_viewpoint = getattr(self._renderer, "viewpoint_changed", None)
        if renderer_viewpoint is not None:
            renderer_viewpoint.connect(self._accept_renderer_viewpoint)
        renderer_snapshot = getattr(self._renderer, "snapshot_ready", None)
        if renderer_snapshot is not None:
            renderer_snapshot.connect(self._accept_renderer_snapshot)
        self._scene_status = _label("Select an installed spatial room", object_name="muted")
        scene_layout.addWidget(self._scene_status)
        actions = QHBoxLayout()
        self._attribution = _label("", object_name="muted", word_wrap=True)
        actions.addWidget(self._attribution, 1)
        self._save_status = _label("Changes save automatically", object_name="muted")
        self._save_status.setAccessibleName("Automatic save status")
        actions.addWidget(self._save_status)
        actions.addStretch()
        reset = QPushButton("Reset view")
        reset.setObjectName("quietAction")
        reset.clicked.connect(self._reset_viewpoint)
        actions.addWidget(reset)
        scene_layout.addLayout(actions)
        root.addWidget(scene, 1)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setFixedWidth(310)
        controls = QVBoxLayout(inspector)
        controls.setContentsMargins(16, 18, 16, 16)
        controls.setSpacing(7)
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
        controls.addWidget(_label("Depth of field", object_name="section"))
        self._blur_strength = self._add_slider(controls, "Background blur", 0, 100, 0)
        self._blur_strength.setToolTip("Soften the room while keeping you in focus")
        controls.addStretch()
        root.addWidget(inspector)

    def discard_viewpoint(self, room_id: str) -> None:
        """Forget an in-memory camera draft when scene framing is rebuilt."""
        self._drafts.pop(room_id, None)
        self._dirty_rooms.discard(room_id)
        self._retried_rooms.discard(room_id)
        self._latest_snapshot_by_room.pop(room_id, None)
        if self._room_id == room_id:
            self._room_id = ""
            self._loaded_scene_id = ""
            self._scene_ready = False

    def set_resource_active(self, active: bool) -> None:  # noqa: FBT001
        """Load spatial geometry only while Adjust is the active product tab."""
        if active == self._resource_active:
            return
        resource_setter = getattr(self._renderer, "set_resource_active", None)
        if not active:
            self._autosave_timer.stop()
            self._retry_timer.stop()
            self._flush_current_autosave()
            self._resource_active = False
            clearer = getattr(self._renderer, "clear_scene", None)
            if callable(clearer):
                clearer()
            self._loaded_scene_id = ""
            self._scene_ready = False
            if callable(resource_setter):
                resource_setter(active=False)
            return
        self._resource_active = True
        self._retried_rooms.discard(self._room_id)
        if callable(resource_setter):
            resource_setter(active=True)
        if self._room_id:
            self.set_room(
                self._room_id,
                self._scene,
                installed=self._installed,
                viewpoint=self._viewpoint,
            )

    def set_room(
        self,
        room: str,
        scene: SceneReference | None = None,
        *,
        installed: bool = False,
        viewpoint: Viewpoint | None = None,
    ) -> None:
        """Restore one room draft and load its managed scene at most once."""
        if self._room_id and self._room_id != room:
            self._autosave_timer.stop()
            self._retry_timer.stop()
            self._flush_current_autosave()
        if self._room_id:
            self._drafts[self._room_id] = self._viewpoint
        self._room_id = room
        self._scene = scene
        self._installed = installed
        self.setAccessibleDescription(f"Adjust settings for {room}")
        self._default_viewpoint = scene.default_viewpoint if scene is not None else Viewpoint()
        self._spatial_depth_available = scene is not None
        self._viewpoint = self._drafts.get(room, viewpoint or self._default_viewpoint)
        if scene is not None:
            self._viewpoint = self._viewpoint.model_copy(
                update={"scene_transform": self._default_viewpoint.scene_transform},
            )
        self._sync_controls()
        self._sync_depth_controls()

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
        self._scene_ready = self._scene_ready and self._loaded_scene_id == scene.asset_id
        self._scene_status.setText("Loading spatial scene…")
        if not self._resource_active:
            self._scene_status.setText("Open Adjust to load the spatial scene")
            return
        method_name = "set_viewpoint" if self._loaded_scene_id == scene.asset_id else "set_scene"
        setter = getattr(self._renderer, method_name, None)
        if callable(setter):
            if method_name == "set_scene":
                self._scene_ready = False
                setter(scene, self._viewpoint)
                self._loaded_scene_id = scene.asset_id
            else:
                setter(self._viewpoint)

    def _set_scene_progress(self, loaded: int, total: int) -> None:
        progress = round(loaded / total * 100)
        self._scene_ready = progress == COMPLETE_PROGRESS
        message = "Scene ready" if progress == COMPLETE_PROGRESS else f"Loading scene… {progress}%"
        self._scene_status.setText(message)
        if self._scene_ready and self._room_id in self._dirty_rooms:
            self._autosave_timer.start(AUTOSAVE_DELAY_MS)

    def _set_scene_error(self, message: str) -> None:
        pending_requests = [
            request_id
            for request_id, request in self._snapshot_requests.items()
            if request[0] == self._room_id
        ]
        if pending_requests:
            for request_id in pending_requests:
                self._fail_snapshot(request_id)
            return
        self._loaded_scene_id = ""
        self._scene_ready = False
        self._scene_status.setText(f"Scene unavailable: {message}. Re-select the room to retry.")

    def _accept_renderer_viewpoint(self, viewpoint: object) -> None:
        if isinstance(viewpoint, Viewpoint):
            self._viewpoint = viewpoint
            self._sync_controls()
            self._publish_viewpoint_change(apply_to_renderer=False)

    def _reset_viewpoint(self) -> None:
        self._viewpoint = self._default_viewpoint
        self._sync_controls()
        self._publish_viewpoint_change()
        self._scene_status.setText("View reset to the safe prepared camera")

    @Slot()
    def _flush_current_autosave(self) -> None:
        room_id = self._room_id
        scene = self._scene
        if room_id not in self._dirty_rooms:
            return
        if (
            not room_id
            or scene is None
            or not self._installed
            or not self._scene_ready
            or self._loaded_scene_id != scene.asset_id
            or not self._resource_active
        ):
            self._save_status.setText("Changes will save when the scene is ready")
            return
        viewpoint = self._viewpoint
        self._dirty_rooms.discard(room_id)
        self.viewpoint_saved.emit(room_id, viewpoint)
        if room_id in self._dirty_rooms:
            return
        requester = getattr(self._renderer, "request_snapshot", None)
        if not callable(requester):
            self._retried_rooms.discard(room_id)
            self._save_status.setText("Changes saved automatically")
            return
        request_id = uuid4().hex
        self._snapshot_requests[request_id] = (room_id, scene.asset_id, viewpoint)
        self._latest_snapshot_by_room[room_id] = request_id
        requester(request_id)
        self._save_status.setText("Saving changes automatically…")

    def _accept_renderer_snapshot(
        self,
        asset_id: str,
        _revision: int,
        kind: str,
        request_id: str,
        payload: str,
    ) -> None:
        request = self._snapshot_requests.get(request_id)
        if (
            request is None
            or asset_id != request[1]
            or kind != "background"
            or self._latest_snapshot_by_room.get(request[0]) != request_id
        ):
            self._snapshot_requests.pop(request_id, None)
            return
        try:
            background = base64.b64decode(payload.encode("ascii"), validate=True)
        except UnicodeEncodeError, binascii.Error, ValueError:
            self._fail_snapshot(request_id)
            return
        room_id, requested_asset_id, viewpoint = request
        self.snapshot_generated.emit(
            request_id,
            room_id,
            requested_asset_id,
            viewpoint,
            background,
        )
        if request_id not in self._snapshot_requests:
            return
        self._snapshot_requests.pop(request_id, None)
        self._latest_snapshot_by_room.pop(room_id, None)
        self._retried_rooms.discard(room_id)
        if room_id == self._room_id and room_id not in self._dirty_rooms:
            self._save_status.setText("Changes saved automatically")

    def report_viewpoint_save_error(self, room_id: str, viewpoint: Viewpoint) -> None:
        """Keep a failed viewpoint write pending for automatic retry."""
        if self._drafts.get(room_id, viewpoint) != viewpoint:
            return
        self._dirty_rooms.add(room_id)
        self._latest_snapshot_by_room.pop(room_id, None)
        self._schedule_retry(room_id)

    def report_snapshot_save_error(self, request_id: str) -> None:
        """Keep a failed derived-background write pending for automatic retry."""
        self._fail_snapshot(request_id)

    def _publish_viewpoint_change(self, *, apply_to_renderer: bool = True) -> None:
        if not self._room_id:
            return
        viewpoint = self._viewpoint
        self._drafts[self._room_id] = viewpoint
        if apply_to_renderer:
            setter = getattr(self._renderer, "set_viewpoint", None)
            if callable(setter):
                setter(viewpoint)
        self.viewpoint_previewed.emit(viewpoint)
        self._dirty_rooms.add(self._room_id)
        self._retried_rooms.discard(self._room_id)
        self._latest_snapshot_by_room.pop(self._room_id, None)
        self._save_status.setText("Saving changes automatically…")
        self._retry_timer.stop()
        if self._scene_ready and self._resource_active:
            self._autosave_timer.start(AUTOSAVE_DELAY_MS)

    def _fail_snapshot(self, request_id: str) -> None:
        request = self._snapshot_requests.pop(request_id, None)
        if request is None:
            return
        room_id, _asset_id, viewpoint = request
        if self._latest_snapshot_by_room.get(room_id) == request_id:
            self._latest_snapshot_by_room.pop(room_id, None)
        if self._drafts.get(room_id, viewpoint) == viewpoint:
            self._dirty_rooms.add(room_id)
            self._schedule_retry(room_id)

    def _schedule_retry(self, room_id: str) -> None:
        can_retry = not (
            room_id != self._room_id
            or not self._resource_active
            or not self._scene_ready
            or room_id in self._retried_rooms
        )
        if room_id == self._room_id:
            message = (
                "Couldn't save automatically; retrying…"
                if can_retry
                else "Couldn't save automatically; will retry later"
            )
            self._save_status.setText(message)
        if not can_retry:
            return
        self._retried_rooms.add(room_id)
        self._retry_timer.start(AUTOSAVE_RETRY_MS)

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
        elif title == "Background blur":
            depth_of_field = self._viewpoint.depth_of_field
            update = {
                "depth_of_field": depth_of_field.model_copy(
                    update={"blur_strength": value / 100},
                ),
            }
        elif title == "Output crop":
            inset = value / 100
            update = {
                "crop": CropRegion(left=inset, top=inset, right=1 - inset, bottom=1 - inset),
            }
        else:
            return
        self._viewpoint = self._viewpoint.model_copy(update=update)
        self._publish_viewpoint_change()

    def _aspect_changed(self, _index: int) -> None:
        self._viewpoint = self._viewpoint.model_copy(
            update={"aspect_ratio": float(self._aspect.currentData())},
        )
        self._aspect_preview.set_aspect_ratio(self._viewpoint.aspect_ratio)
        self._publish_viewpoint_change()

    def _sync_controls(self) -> None:
        values = {
            "Field of view": round(self._viewpoint.field_of_view),
            "Horizon": round(self._viewpoint.horizon * 10),
            "Output crop": round(self._viewpoint.crop.left * 100),
            "Background blur": round(
                self._viewpoint.depth_of_field.blur_strength * 100,
            ),
        }
        for title, value in values.items():
            slider = self._sliders[title]
            slider.blockSignals(True)  # noqa: FBT003
            slider.setValue(value)
            slider.blockSignals(False)  # noqa: FBT003
            self._slider_labels[title].setText(self._format_slider(title, value))
        index = min(
            range(self._aspect.count()),
            key=lambda item: abs(float(self._aspect.itemData(item)) - self._viewpoint.aspect_ratio),
        )
        self._aspect.blockSignals(True)  # noqa: FBT003
        self._aspect.setCurrentIndex(index)
        self._aspect.blockSignals(False)  # noqa: FBT003
        self._aspect_preview.set_aspect_ratio(self._viewpoint.aspect_ratio)
        self._sync_depth_controls()

    def _sync_depth_controls(self) -> None:
        tooltip = "" if self._spatial_depth_available else "Available when a spatial room is loaded"
        self._blur_strength.setEnabled(self._spatial_depth_available)
        self._blur_strength.setToolTip(tooltip)

    @staticmethod
    def _format_slider(title: str, value: int) -> str:
        if title == "Field of view":
            return f"{value}°"
        if title == "Horizon":
            return f"{value / 10:.1f}°"
        return f"{value}%"
