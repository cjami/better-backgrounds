"""Adjust-page room viewpoint and presentation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
from better_backgrounds.harmonization import HarmonizationSettings
from better_backgrounds.scene import CropRegion, SceneReference, Viewpoint

if TYPE_CHECKING:
    from collections.abc import Callable

COMPLETE_PROGRESS = 100


class AdjustPage(QWidget):
    """Adjust the current room's viewpoint and presentation settings."""

    viewpoint_saved = Signal(str, object)
    viewpoint_previewed = Signal(object)
    mirroring_changed = Signal(bool)
    harmonization_changed = Signal(object)

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
        self._harmonization_drafts: dict[str, HarmonizationSettings] = {}
        self._harmonization = HarmonizationSettings(global_harmonization=True)
        self._harmonization_controls: dict[str, QCheckBox] = {}
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._loaded_scene_id = ""
        self._spatial_depth_available = False
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
        controls.addWidget(_label("Depth of field", object_name="section"))
        controls.addWidget(
            _label(
                "You stay in focus. Increase this to progressively soften room details by depth.",
                object_name="muted",
                word_wrap=True,
            ),
        )
        self._blur_strength = self._add_slider(controls, "Depth-of-field blur", 0, 100, 0)
        controls.addSpacing(6)
        controls.addWidget(_label("Foreground", object_name="section"))
        self._mirrored = QCheckBox("Mirror my preview")
        self._mirrored.setChecked(True)
        self._mirrored.setToolTip("Mirrors the webcam foreground only; the room is never mirrored")
        self._mirrored.toggled.connect(self.mirroring_changed)
        controls.addWidget(self._mirrored)
        controls.addWidget(
            _label(
                "MatAnyone 2 owns the live matte and temporal memory. "
                "Use Re-select person in Show if tracking is lost.",
                object_name="muted",
                word_wrap=True,
            ),
        )
        controls.addSpacing(6)
        controls.addWidget(_label("Harmonisation", object_name="section"))
        controls.addWidget(
            _label(
                "Harmonizer uses an external non-commercial checkpoint. Global adjustments are "
                "smoothed over time; edge cleanup and fallback are automatic.",
                object_name="muted",
                word_wrap=True,
            ),
        )
        harmonization_components = (("global_harmonization", "Global harmonization"),)
        for key, title in harmonization_components:
            component = QCheckBox(title)
            component.setObjectName(f"harmonization-{key.replace('_', '-')}")
            component.toggled.connect(self._harmonization_control_changed)
            self._harmonization_controls[key] = component
            controls.addWidget(component)
        controls.addWidget(
            _label(
                "Compare retains the standard exact-frame composite as its baseline.",
                object_name="muted",
                word_wrap=True,
            ),
        )
        controls.addStretch()
        root.addWidget(inspector)

    def set_live_preferences(self, *, mirrored: bool) -> None:
        """Restore foreground presentation without emitting changes."""
        self._mirrored.blockSignals(True)  # noqa: FBT003
        self._mirrored.setChecked(mirrored)
        self._mirrored.blockSignals(False)  # noqa: FBT003

    def discard_viewpoint(self, room_id: str) -> None:
        """Forget an in-memory camera draft when scene framing is rebuilt."""
        self._drafts.pop(room_id, None)
        if self._room_id == room_id:
            self._room_id = ""
            self._loaded_scene_id = ""

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
            self._harmonization_drafts[self._room_id] = self._harmonization
        self._room_id = room
        self.setAccessibleDescription(f"Adjust settings for {room}")
        self._default_viewpoint = scene.default_viewpoint if scene is not None else Viewpoint()
        self._spatial_depth_available = scene is not None
        self._viewpoint = self._drafts.get(room, viewpoint or self._default_viewpoint)
        self._harmonization = self._harmonization_drafts.get(
            room,
            HarmonizationSettings(global_harmonization=True),
        )
        if scene is not None:
            self._viewpoint = self._viewpoint.model_copy(
                update={"scene_transform": self._default_viewpoint.scene_transform},
            )
        self._sync_controls()
        self._sync_depth_controls()
        self._sync_harmonization_controls()
        self.harmonization_changed.emit(self._harmonization)

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
            self.viewpoint_previewed.emit(viewpoint)

    def _reset_viewpoint(self) -> None:
        self._viewpoint = self._default_viewpoint
        self._drafts[self._room_id] = self._viewpoint
        self._sync_controls()
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(self._viewpoint)
        self.viewpoint_previewed.emit(self._viewpoint)
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
        self.viewpoint_previewed.emit(viewpoint)

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
        elif title == "Depth-of-field blur":
            depth_of_field = self._viewpoint.depth_of_field
            update = {
                "depth_of_field": depth_of_field.model_copy(
                    update={"blur_strength": value / 100},
                ),
            }
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
        self.viewpoint_previewed.emit(self._viewpoint)

    def _aspect_changed(self, _index: int) -> None:
        self._viewpoint = self._viewpoint.model_copy(
            update={"aspect_ratio": float(self._aspect.currentData())},
        )
        self._drafts[self._room_id] = self._viewpoint
        self._aspect_preview.set_aspect_ratio(self._viewpoint.aspect_ratio)
        setter = getattr(self._renderer, "set_viewpoint", None)
        if callable(setter):
            setter(self._viewpoint)
        self.viewpoint_previewed.emit(self._viewpoint)

    def _harmonization_control_changed(self) -> None:
        values = {key: control.isChecked() for key, control in self._harmonization_controls.items()}
        self._harmonization = HarmonizationSettings.model_validate(values)
        if self._room_id:
            self._harmonization_drafts[self._room_id] = self._harmonization
        self.harmonization_changed.emit(self._harmonization)

    def _sync_harmonization_controls(self) -> None:
        for key, control in self._harmonization_controls.items():
            control.blockSignals(True)  # noqa: FBT003
            control.setChecked(bool(getattr(self._harmonization, key)))
            control.blockSignals(False)  # noqa: FBT003

    def _sync_controls(self) -> None:
        values = {
            "Field of view": round(self._viewpoint.field_of_view),
            "Horizon": round(self._viewpoint.horizon * 10),
            "Output crop": round(self._viewpoint.crop.left * 100),
            "Depth-of-field blur": round(
                self._viewpoint.depth_of_field.blur_strength * 100,
            ),
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
        if title == "Output crop":
            return f"{value}%"
        return f"{value}%"
