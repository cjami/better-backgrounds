"""Feature-first: Tests for the retained Show live pipeline."""

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from better_backgrounds.desktop.camera import InputCamera, InputCameraSource
from better_backgrounds.desktop.main_window import MainWindow
from better_backgrounds.desktop.pages import ShowPage
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.harmonization import HarmonizationSettings

if TYPE_CHECKING:
    from pathlib import Path

_APPLICATION: QApplication | None = None


class TrackingLiveRenderer(ScenePreview):
    """Record live pipeline commands without camera or browser access."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)
    composite_frame_ready = Signal(object, float)

    def __init__(self) -> None:
        """Create empty lifecycle and presentation logs."""
        super().__init__()
        self.starts: list[tuple[str, bool]] = []
        self.stops = 0
        self.mirroring: list[bool] = []
        self.matting: list[str] = []
        self.harmonization: list[HarmonizationSettings] = []
        self.resource_states: list[bool] = []

    def start_camera(self, label: str, *, mirrored: bool) -> None:
        """Record one explicit camera start."""
        self.starts.append((label, mirrored))

    def stop_camera(self) -> None:
        """Record prompt stream teardown."""
        self.stops += 1

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Record foreground-only mirroring changes."""
        self.mirroring.append(mirrored)

    def set_matting_settings(self, payload: str) -> None:
        """Record worker refinement changes."""
        self.matting.append(payload)

    def set_harmonization(self, settings: HarmonizationSettings) -> None:
        """Record the room-scoped global appearance switch."""
        self.harmonization.append(settings)

    def set_resource_active(self, active: bool) -> None:  # noqa: FBT001
        """Record whether hidden live resources should remain active."""
        self.resource_states.append(active)


class PassiveVirtualCamera:
    """Keep a deterministic virtual output active until the controller closes it."""

    def send(self, frame: object) -> None:
        """Accept a frame without touching an operating-system camera."""
        _ = frame

    def sleep_until_next_frame(self) -> None:
        """Yield briefly to model the backend cadence."""
        time.sleep(0.002)

    def close(self) -> None:
        """Release the fake output."""


def wait_until(predicate, *, timeout: float = 1.0) -> None:  # noqa: ANN001
    """Process queued Qt events until a controller transition is visible."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application().processEvents()
        if predicate():
            return
        time.sleep(0.002)
    message = "condition was not met before timeout"
    raise AssertionError(message)


def application() -> QApplication:
    """Return the one QApplication allowed by Qt."""
    global _APPLICATION  # noqa: PLW0603
    if _APPLICATION is None:
        instance = QApplication.instance()
        _APPLICATION = instance if isinstance(instance, QApplication) else QApplication([])
    return _APPLICATION


def test_failed_seed_waits_for_an_explicit_retry() -> None:
    """Avoid repeatedly loading MediaPipe when no valid person seed is found."""
    application()
    page = ShowPage([], ScenePreview)
    retry = next(button for button in page.findChildren(QPushButton) if button.text() == "Retry")
    confirm = next(
        button for button in page.findChildren(QPushButton) if button.text() == "Confirm person"
    )

    page.set_camera_state("seed-error", "No person was found")

    assert not retry.isHidden()
    assert confirm.isHidden()
    page.close()


def test_tracking_loss_offers_explicit_person_reselection() -> None:
    """Pause on a lost target without silently restarting person confirmation."""
    application()
    page = ShowPage([], ScenePreview)
    reselect = next(
        button for button in page.findChildren(QPushButton) if button.text() == "Re-select person"
    )
    requests: list[bool] = []
    page.reseed_requested.connect(lambda: requests.append(True))

    page.set_camera_state("lost", "Tracking paused")
    reselect.click()

    assert not reselect.isHidden()
    assert requests == [True]
    page.close()


def create_window(tmp_path: Path, pipeline: TrackingLiveRenderer) -> MainWindow:
    """Create a deterministic desktop with one available camera."""
    application()
    cameras = (InputCamera(device_id="camera-a", description="Desk camera", is_default=True),)
    return MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        live_renderer_factory=lambda: pipeline,
        camera_source=InputCameraSource(lambda: cameras),
        virtual_camera_sink_factory=lambda _profile: PassiveVirtualCamera(),
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )


def test_adjust_suspends_live_resources_without_restarting_camera(tmp_path: Path) -> None:
    """Yield live resources to Adjust and restore them on the other tabs."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    camera = window.findChild(QPushButton, "cameraToggle")
    assert camera is not None
    virtual_states: list[bool] = []
    window.virtual_camera_changed.connect(virtual_states.append)

    assert pipeline.starts == [("camera-a", True)]
    assert pipeline.resource_states == [True]
    camera.click()
    wait_until(lambda: camera.text() == "Stop virtual camera")
    window.select_tab(2)
    window.select_tab(1)
    window.select_tab(0)

    assert pipeline.starts == [("camera-a", True)]
    assert pipeline.stops == 0
    assert pipeline.resource_states == [True, False, True, True]
    camera.click()
    wait_until(lambda: camera.text() == "Start virtual camera")
    assert pipeline.stops == 0
    assert virtual_states == [True, False]
    window.close()
    assert pipeline.stops == 1


def test_application_close_releases_an_active_camera(tmp_path: Path) -> None:
    """Stop the retained stream during application shutdown."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    camera = window.findChild(QPushButton, "cameraToggle")
    assert camera is not None

    window.close()

    assert pipeline.stops == 1


def test_show_persists_and_applies_foreground_only_mirroring(tmp_path: Path) -> None:
    """Keep the webcam mirror setting with the live presentation controls."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    show_page = window.findChild(ShowPage)
    assert show_page is not None
    mirror = next(
        checkbox
        for checkbox in show_page.findChildren(QCheckBox)
        if checkbox.text() == "Mirror webcam"
    )

    mirror.setChecked(False)

    assert pipeline.mirroring == [False]
    window.close()

    restored_pipeline = TrackingLiveRenderer()
    restored = create_window(tmp_path, restored_pipeline)
    restored_show_page = restored.findChild(ShowPage)
    assert restored_show_page is not None
    restored_mirror = next(
        checkbox
        for checkbox in restored_show_page.findChildren(QCheckBox)
        if checkbox.text() == "Mirror webcam"
    )
    assert not restored_mirror.isChecked()
    restored.close()


def test_harmonise_subject_is_on_by_default_in_show_and_can_be_disabled(tmp_path: Path) -> None:
    """Keep the appearance setting beside the live webcam controls."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    show_page = window.findChild(ShowPage)
    assert show_page is not None
    controls = {checkbox.text(): checkbox for checkbox in show_page.findChildren(QCheckBox)}

    expected = {"Harmonise subject"}
    assert expected <= controls.keys()
    assert all(controls[title].isChecked() for title in expected)
    assert "Depth-dependent effects" not in controls
    assert pipeline.harmonization[-1] == HarmonizationSettings(global_harmonization=True)

    controls["Harmonise subject"].setChecked(False)

    assert pipeline.harmonization[-1] == HarmonizationSettings(global_harmonization=False)
    window.close()
