"""Feature-first: Headless tests for the Python-owned desktop boundary."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QPushButton,
    QStackedLayout,
    QStackedWidget,
)

from better_backgrounds.desktop.app import (
    packaged_sharp_command,
    packaged_worker_command,
)
from better_backgrounds.desktop.bridge import LiveRendererBridge, RendererBridge
from better_backgrounds.desktop.camera import InputCamera, InputCameraSource
from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.main_window import MainWindow
from better_backgrounds.desktop.pages import AdjustPage
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

if TYPE_CHECKING:
    import pytest

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

    def set_scene(self, scene: SceneReference, viewpoint: Viewpoint) -> None:
        """Record one managed scene load."""
        self.scenes.append((scene, viewpoint))

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Record one camera-only update."""
        self.viewpoints.append(viewpoint)


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
