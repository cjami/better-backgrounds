"""Tests for the one retained Show and Compare live pipeline."""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from better_backgrounds.desktop.main_window import MainWindow
from better_backgrounds.desktop.pages import ShowPage
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.input_camera import InputCamera, InputCameraSource

if TYPE_CHECKING:
    from pathlib import Path

_APPLICATION: QApplication | None = None


class TrackingLiveRenderer(ScenePreview):
    """Record live pipeline commands without camera or browser access."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)

    def __init__(self) -> None:
        """Create empty lifecycle and presentation logs."""
        super().__init__()
        self.starts: list[tuple[str, bool]] = []
        self.stops = 0
        self.presentations: list[tuple[str, int]] = []
        self.mirroring: list[bool] = []
        self.matting: list[str] = []

    def start_camera(self, label: str, *, mirrored: bool) -> None:
        """Record one explicit camera start."""
        self.starts.append((label, mirrored))

    def stop_camera(self) -> None:
        """Record prompt stream teardown."""
        self.stops += 1

    def set_presentation(self, mode: str, wipe: int) -> None:
        """Record presentation-only changes."""
        self.presentations.append((mode, wipe))

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Record foreground-only mirroring changes."""
        self.mirroring.append(mirrored)

    def set_matting_settings(self, payload: str) -> None:
        """Record worker refinement changes."""
        self.matting.append(payload)


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


def create_window(tmp_path: Path, pipeline: TrackingLiveRenderer) -> MainWindow:
    """Create a deterministic desktop with one available camera."""
    application()
    cameras = (InputCamera(device_id="camera-a", description="Desk camera", is_default=True),)
    return MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
        live_renderer_factory=lambda: pipeline,
        camera_source=InputCameraSource(lambda: cameras),
        scene_cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )


def test_preview_starts_before_virtual_output_and_survives_tab_changes(tmp_path: Path) -> None:
    """Keep local preview capture independent from virtual-camera publication."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    camera = window.findChild(QPushButton, "cameraToggle")
    assert camera is not None
    virtual_states: list[bool] = []
    window.virtual_camera_changed.connect(virtual_states.append)

    assert pipeline.starts == [("camera-a", True)]
    camera.click()
    window.select_tab(2)
    window.select_tab(3)
    window.select_tab(0)

    assert pipeline.starts == [("camera-a", True)]
    assert pipeline.stops == 0
    assert ("compare", 52) in pipeline.presentations
    camera.click()
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


def test_adjust_persists_and_applies_foreground_only_mirroring(tmp_path: Path) -> None:
    """Keep mirroring in Adjust and leave the room renderer unchanged."""
    application()
    pipeline = TrackingLiveRenderer()
    window = create_window(tmp_path, pipeline)
    mirror = next(
        checkbox
        for checkbox in window.findChildren(QCheckBox)
        if checkbox.text() == "Mirror my preview"
    )

    mirror.setChecked(False)

    assert pipeline.mirroring == [False]
    window.close()

    restored_pipeline = TrackingLiveRenderer()
    restored = create_window(tmp_path, restored_pipeline)
    restored_mirror = next(
        checkbox
        for checkbox in restored.findChildren(QCheckBox)
        if checkbox.text() == "Mirror my preview"
    )
    assert not restored_mirror.isChecked()
    restored.close()
