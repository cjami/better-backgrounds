"""Feature-first: Headless tests for the Python-owned desktop boundary."""

import base64
import sys
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QUrl, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QPushButton,
    QSlider,
    QStackedLayout,
    QStackedWidget,
)

from better_backgrounds.desktop.app import (
    packaged_sharp_command,
    packaged_splat_command,
    packaged_worker_command,
)
from better_backgrounds.desktop.bridge import LiveRendererBridge, RendererBridge
from better_backgrounds.desktop.camera import InputCamera, InputCameraSource
from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.main_window import MainWindow
from better_backgrounds.desktop.pages import AdjustPage
from better_backgrounds.desktop.pages.common import AspectRatioContainer
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.desktop.webview import navigation_is_allowed
from better_backgrounds.jobs.build_session import IdleBuild
from better_backgrounds.scene import (
    Quaternion,
    SceneReference,
    SceneTransform,
    Viewpoint,
    load_sample_manifest,
)

TAB_COUNT = 4
BUILD_TAB = 1
COMPARE_TAB = 3


class TrackingRenderer(ScenePreview):
    """Record scene commands without creating native WebEngine state."""

    def __init__(self) -> None:
        """Create an empty command log."""
        super().__init__()
        self.scenes: list[tuple[SceneReference, Viewpoint]] = []
        self.viewpoints: list[Viewpoint] = []
        self.clears = 0
        self.resource_states: list[bool] = []

    def set_scene(self, scene: SceneReference, viewpoint: Viewpoint) -> None:
        """Record one managed scene load."""
        self.scenes.append((scene, viewpoint))

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Record one camera-only update."""
        self.viewpoints.append(viewpoint)

    def clear_scene(self) -> None:
        """Record release of retained spatial geometry."""
        self.clears += 1

    def set_resource_active(self, active: bool) -> None:  # noqa: FBT001
        """Record WebGL frame-loop suspension."""
        self.resource_states.append(active)


class SnapshotTrackingRenderer(TrackingRenderer):
    """Expose current-frame snapshot signals for Adjust-page tests."""

    snapshot_ready = Signal(str, int, str, str)
    scene_progressed = Signal(int, int)

    def __init__(self) -> None:
        """Record requested output dimensions."""
        super().__init__()
        self.snapshot_requests = 0

    def request_snapshot(self) -> None:
        """Record one explicit current-frame export."""
        self.snapshot_requests += 1


def application() -> QApplication:
    """Return the one application allowed by Qt per process."""
    existing = cast("QApplication | None", QApplication.instance())
    return existing or QApplication([])


def test_main_window_contains_four_independent_product_tabs() -> None:
    """Construct the complete tabbed shell without native media behavior."""
    app = application()
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
    )

    stack = window.findChild(QStackedWidget, "tabPages")
    tabs = window.findChildren(QPushButton, "tab")

    assert app.applicationName() is not None
    assert stack is not None
    assert stack.count() == TAB_COUNT
    layout = stack.layout()
    assert isinstance(layout, QStackedLayout)
    assert layout.stackingMode() is QStackedLayout.StackingMode.StackAll
    assert [tab.text() for tab in tabs] == ["Show", "Build", "Adjust", "Compare"]
    assert all(tab.isEnabled() for tab in tabs)
    assert window.active_tab == 0
    assert isinstance(window.build_session.state, IdleBuild)
    device = window.findChild(QComboBox, "sharpDevice")
    assert device is not None
    assert device.currentData() == "auto"
    window.close()


def test_application_icon_loads_from_package_data() -> None:
    """Keep the shared vector mark available to source and packaged builds."""
    application()

    assert not application_icon().isNull()


def test_tabs_can_be_opened_in_any_order() -> None:
    """Keep navigation separate from build-session state."""
    window = MainWindow(command_factory=lambda _job_id, _outcome: [], renderer_factory=ScenePreview)

    window.select_tab(COMPARE_TAB)
    assert window.active_tab == COMPARE_TAB
    window.select_tab(BUILD_TAB)
    assert window.active_tab == BUILD_TAB
    assert isinstance(window.build_session.state, IdleBuild)
    window.close()


def test_adjust_renderer_remains_mapped_behind_other_tabs(tmp_path: Path) -> None:
    """Avoid remapping Chromium's GPU surface during tab switches."""
    renderer = TrackingRenderer()
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=lambda: renderer,
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    window.show()
    renderer.show()

    window.select_tab(BUILD_TAB)
    application().processEvents()

    assert renderer.isVisibleTo(window)
    window.close()


def test_sample_is_a_stable_room_without_forcing_navigation(tmp_path: Path) -> None:
    """Share the sample identifier while leaving the active product tab alone."""
    renderers: list[TrackingRenderer] = []

    def renderer_factory() -> TrackingRenderer:
        renderer = TrackingRenderer()
        renderers.append(renderer)
        return renderer

    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=renderer_factory,
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    sample_downloads = [
        button
        for button in window.findChildren(QPushButton)
        if button.text().startswith("Download sample")
    ]

    assert window.selected_room == "Table Tennis Room — Sample"
    assert window.selected_room_id == "table-tennis-room-v1"
    assert window.active_tab == 0
    assert len(renderers) == 1
    assert len(sample_downloads) == 1
    window.select_tab(2)
    window.select_tab(0)
    assert len(renderers) == 1
    window.close()


def test_last_selected_room_is_restored_on_relaunch(tmp_path: Path) -> None:
    """Open the previous room directly so its cached render can be presented."""
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        scene_cache_root=cache_root,
        data_root=data_root,
    )
    window.select_room("Living room")
    window.close()

    restored = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        scene_cache_root=cache_root,
        data_root=data_root,
    )

    assert restored.selected_room == "Living room"
    restored.close()


def test_adjust_reuses_scene_and_keeps_its_room_draft() -> None:
    """Avoid a scene reload when a room is revisited in the retained page."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]

    page.set_room(scene.asset_id, scene, installed=True)
    page.set_room("another-room")
    page.set_room(scene.asset_id, scene, installed=True)

    assert len(renderer.scenes) == 1
    assert renderer.viewpoints[-1] == scene.default_viewpoint
    page.close()


def test_adjust_defers_spatial_scene_load_until_the_tab_is_active() -> None:
    """Room selection alone must not load a splat for Show or Compare."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]
    page.set_resource_active(False)

    page.set_room(scene.asset_id, scene, installed=True)
    assert renderer.scenes == []

    page.set_resource_active(True)
    assert renderer.scenes == [(scene, scene.default_viewpoint)]
    page.close()


def test_adjust_unloads_scene_and_suspends_rendering_when_hidden() -> None:
    """Release splat memory and GPU time before Show resumes webcam inference."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]
    page.set_room(scene.asset_id, scene, installed=True)

    page.set_resource_active(False)

    assert renderer.clears == 1
    assert renderer.resource_states == [False]

    page.set_resource_active(True)
    assert renderer.resource_states == [False, True]
    assert renderer.scenes == [
        (scene, scene.default_viewpoint),
        (scene, scene.default_viewpoint),
    ]
    page.close()


def test_adjust_saves_the_current_renderer_frame_immediately() -> None:
    """Use the visible framebuffer as both presentation and room evidence."""
    application()
    renderer = SnapshotTrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]
    generated: list[tuple[object, ...]] = []
    page.snapshot_generated.connect(lambda *values: generated.append(values))
    page.set_room(scene.asset_id, scene, installed=True)
    renderer.scene_progressed.emit(100, 100)
    save = next(
        button
        for button in page.findChildren(QPushButton)
        if button.text() == "Save view & background"
    )

    save.click()
    renderer.snapshot_ready.emit(
        scene.asset_id,
        4,
        "background",
        base64.b64encode(b"background").decode(),
    )

    assert renderer.snapshot_requests == 1
    assert generated == [
        (scene.asset_id, scene.asset_id, scene.default_viewpoint, b"background"),
    ]
    page.close()


def test_adjust_reloads_a_reimported_scene_from_its_new_default() -> None:
    """Discard stale camera and renderer state when a stable scene ID is rebuilt."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]
    sliders = {slider.accessibleName(): slider for slider in page.findChildren(QSlider)}

    page.set_room(scene.asset_id, scene, installed=True)
    sliders["Field of view"].setValue(60)
    page.discard_viewpoint(scene.asset_id)
    page.set_room(scene.asset_id, scene, installed=True)

    assert len(renderer.scenes) == 2
    assert renderer.scenes[-1][1] == scene.default_viewpoint
    page.close()


def test_adjust_keeps_asset_normalization_when_restoring_a_saved_camera() -> None:
    """Do not let an older room preference undo the scene's import transform."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    scene = load_sample_manifest().scenes[0]
    transform = SceneTransform(orientation=Quaternion(z=1.0, w=0.0))
    scene = scene.model_copy(
        update={
            "default_viewpoint": scene.default_viewpoint.model_copy(
                update={"scene_transform": transform},
            ),
        },
    )

    page.set_room(scene.asset_id, scene, installed=True, viewpoint=Viewpoint())

    assert renderer.scenes[-1][1].scene_transform == transform
    page.close()


def test_adjust_enables_depth_of_field_for_every_spatial_scene() -> None:
    """Expose zero-default depth blur for any loaded spatial room."""
    application()
    renderer = TrackingRenderer()
    page = AdjustPage(lambda: renderer)
    sample = load_sample_manifest().scenes[0]
    sliders = {slider.accessibleName(): slider for slider in page.findChildren(QSlider)}
    checkboxes = {checkbox.text(): checkbox for checkbox in page.findChildren(QCheckBox)}

    page.set_room(sample.asset_id, sample, installed=True)

    assert sliders["Depth-of-field blur"].isEnabled()
    assert sliders["Depth-of-field blur"].value() == 0
    assert "Depth of field" not in checkboxes
    assert "Depth in scene" not in sliders
    assert "Focus band" not in sliders
    assert "Depth-aware occlusion" not in checkboxes
    assert "Subject size" not in sliders

    sliders["Depth-of-field blur"].setValue(70)
    assert renderer.viewpoints[-1].depth_of_field.blur_strength == 0.7
    page.close()


def test_adjust_and_show_follow_the_same_output_aspect(tmp_path: Path) -> None:
    """Keep room framing identical while moving between adjustment and presentation."""
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    containers = {
        container.objectName(): container for container in window.findChildren(AspectRatioContainer)
    }

    assert containers["showAspectPreview"].aspect_ratio == 16 / 9
    assert containers["adjustAspectPreview"].aspect_ratio == 16 / 9
    aspects = [
        combo
        for combo in window.findChildren(QComboBox)
        if combo.accessibleName() == "Output aspect ratio"
    ]
    assert len(aspects) == 1
    aspects[0].setCurrentIndex(1)
    assert containers["showAspectPreview"].aspect_ratio == 4 / 3
    assert containers["adjustAspectPreview"].aspect_ratio == 4 / 3
    window.close()


def test_show_tab_has_a_clear_camera_toggle() -> None:
    """Expose explicit start and stop actions for local webcam capture."""
    camera_source = InputCameraSource(
        lambda: (InputCamera(device_id="camera-a", description="Desk camera"),),
    )
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        camera_source=camera_source,
    )
    camera = window.findChild(QPushButton, "cameraToggle")

    assert camera is not None
    assert camera.text() == "●  Start virtual camera"
    camera.click()
    assert camera.text() == "■  Stop virtual camera"
    camera.click()
    assert camera.text() == "●  Start virtual camera"
    window.close()


def test_show_selects_and_persists_an_input_camera(tmp_path: Path) -> None:
    """Keep webcam selection in Python-owned application state."""
    cameras = (
        InputCamera(device_id="camera-a", description="Desk camera"),
        InputCamera(device_id="camera-b", description="Monitor camera", is_default=True),
    )
    data_root = tmp_path / "data"
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        camera_source=InputCameraSource(lambda: cameras),
        scene_cache_root=tmp_path / "cache",
        data_root=data_root,
    )
    selector = window.findChild(QComboBox, "inputCameraSelector")

    assert selector is not None
    assert selector.currentData() == "camera-b"
    selector.setCurrentIndex(0)
    assert window.selected_input_camera_id == "camera-a"
    window.close()

    restored = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        camera_source=InputCameraSource(lambda: cameras),
        scene_cache_root=tmp_path / "cache",
        data_root=data_root,
    )
    restored_selector = restored.findChild(QComboBox, "inputCameraSelector")
    assert restored_selector is not None
    assert restored_selector.currentData() == "camera-a"
    restored.close()


def test_input_camera_hotplug_falls_back_without_losing_preference(tmp_path: Path) -> None:
    """Temporarily use an available camera and restore the preferred device on return."""
    cameras = [
        InputCamera(device_id="camera-a", description="Desk camera"),
        InputCamera(device_id="camera-b", description="Monitor camera", is_default=True),
    ]
    source = InputCameraSource(lambda: tuple(cameras))
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        camera_source=source,
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    selector = window.findChild(QComboBox, "inputCameraSelector")
    assert selector is not None
    selector.setCurrentIndex(0)

    cameras.pop(0)
    source.refresh()
    assert window.selected_input_camera_id == "camera-b"

    cameras.insert(0, InputCamera(device_id="camera-a", description="Desk camera"))
    source.refresh()
    assert window.selected_input_camera_id == "camera-a"
    window.close()


def test_show_reports_when_no_input_camera_is_available(tmp_path: Path) -> None:
    """Keep the empty-device state explicit and non-interactive."""
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        camera_source=InputCameraSource(lambda: ()),
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    selector = window.findChild(QComboBox, "inputCameraSelector")

    assert selector is not None
    assert not selector.isEnabled()
    assert selector.currentText() == "No camera detected"
    assert window.selected_input_camera_id is None
    window.close()


def test_renderer_bridge_rejects_invalid_viewpoint() -> None:
    """Expose a task-specific validated method instead of general Python access."""
    bridge = RendererBridge()

    assert not bridge.submit_viewpoint('{"field_of_view":200}')
    assert bridge.submit_viewpoint(
        '{"field_of_view":42,"horizon":-1.5,"subject_depth":2.4,"focus_depth":2.6}',
    )


def test_renderer_bridge_rejects_invalid_scene_status() -> None:
    """Keep progress and errors bounded before they reach Python UI state."""
    bridge = RendererBridge()

    assert bridge.report_scene_progress("sample-room", 50, 100)
    assert not bridge.report_scene_progress("sample-room", 101, 100)
    assert bridge.report_scene_error("sample-room", "gpu_unavailable", "No GPU renderer")
    assert not bridge.report_scene_error("sample-room", "bad code!", "No GPU renderer")
    assert bridge.report_snapshot_ready("sample-room", 3, "background", "cG5n")
    assert bridge.report_snapshot_ready("sample-room", 3, "harmonization", "cG5n")
    assert not bridge.report_snapshot_ready("sample-room", 3, "occlusion", "cG5n")
    assert not bridge.report_snapshot_ready("sample-room", -1, "background", "cG5n")
    assert not bridge.report_snapshot_ready("sample-room", 3, "depth", "cG5n")
    assert not bridge.report_snapshot_ready("sample-room", 3, "background", "")


def test_renderer_bridge_bounds_fixed_output_size() -> None:
    """Keep output framebuffer requests inside the trusted renderer limit."""
    bridge = RendererBridge()
    requested: list[tuple[int, int]] = []
    bridge.output_size_requested.connect(lambda width, height: requested.append((width, height)))

    bridge.request_output_size(1920, 1080)

    assert requested == [(1920, 1080)]
    with pytest.raises(ValueError, match="between 1 and 8192"):
        bridge.request_output_size(0, 1080)


def test_live_bridge_validates_camera_state_and_diagnostics() -> None:
    """Keep browser lifecycle and performance messages inside bounded schemas."""
    bridge = LiveRendererBridge()

    assert bridge.report_camera_state("live", "Live · Desk camera")
    assert not bridge.report_camera_state("recording", "Unexpected state")
    assert bridge.report_diagnostics(
        '{"display_fps":30,"mask_fps":18,"mask_age_ms":42,'
        '"dropped_frames":2,"worker_time_ms":31,'
        '"processing_width":256,"processing_height":144}',
    )
    assert not bridge.report_diagnostics('{"display_fps":-1}')


def test_navigation_is_restricted_to_synthetic_origin() -> None:
    """Block filesystem and network navigation from the embedded renderer."""
    assert navigation_is_allowed(QUrl("https://app.better-backgrounds.invalid/viewer.html"))
    assert not navigation_is_allowed(QUrl("file:///private/room.sog"))
    assert not navigation_is_allowed(QUrl("https://example.com/"))


def test_frozen_application_reuses_its_executable_for_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the fake process contract operational in a standalone package."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = packaged_worker_command("job-1", "success")

    assert command[:2] == [str(Path(sys.argv[0]).resolve()), "--fake-worker"]


def test_frozen_application_reuses_its_executable_for_sharp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep SHARP inference inside the packaged worker dispatch."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = packaged_sharp_command(
        "job-1",
        Path("room image.jpg"),
        "cuda",
        "upload",
    )

    assert command[:2] == [str(Path(sys.argv[0]).resolve()), "--sharp-worker"]
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--source-kind") + 1] == "upload"


def test_frozen_application_reuses_its_executable_for_splat_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep direct PLY ingestion inside the packaged worker dispatch."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = packaged_splat_command("job-1", Path("room scene.ply"))

    assert command[:2] == [str(Path(sys.argv[0]).resolve()), "--splat-worker"]
    assert command[command.index("--source") + 1] == "room scene.ply"

    streamed = packaged_splat_command("job-2", Path("whole room.ssog"))
    assert streamed[:2] == [str(Path(sys.argv[0]).resolve()), "--splat-worker"]
    assert streamed[streamed.index("--source") + 1] == "whole room.ssog"
