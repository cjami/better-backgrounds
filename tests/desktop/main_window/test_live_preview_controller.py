"""Focused tests for retained camera-preview controller ownership."""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.camera import InputCamera, InputCameraSource
from better_backgrounds.desktop.main_window import LivePreviewController
from better_backgrounds.desktop.pages import ShowPage

if TYPE_CHECKING:
    from pathlib import Path


class RecordingPreview(QWidget):
    """Record camera lifecycle and preference application."""

    def __init__(self) -> None:
        """Create empty lifecycle logs."""
        super().__init__()
        self.starts: list[tuple[str, bool, str]] = []
        self.stops = 0
        self.mirroring: list[bool] = []

    def start_camera(
        self,
        device_id: str,
        *,
        mirrored: bool,
        input_resolution: str,
    ) -> None:
        """Record one camera start."""
        self.starts.append((device_id, mirrored, input_resolution))

    def stop_camera(self) -> None:
        """Record one preview teardown."""
        self.stops += 1

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Record a live preference update."""
        self.mirroring.append(mirrored)


def test_live_controller_restores_hotplug_selection_preferences_and_shutdown(
    tmp_path: Path,
) -> None:
    """Keep one preferred camera across fallback, restart, and application close."""
    QApplication.instance() or QApplication([])
    cameras = [
        InputCamera(device_id="camera-a", description="Desk camera"),
        InputCamera(device_id="camera-b", description="Monitor camera", is_default=True),
    ]
    source = InputCameraSource(lambda: tuple(cameras))
    preview = RecordingPreview()
    show = ShowPage([], lambda: preview)
    parent = QWidget()
    controller = LivePreviewController(
        parent,
        show,
        preview,
        source,
        tmp_path,
    )

    controller.start()
    controller.select_input_camera("camera-a")
    cameras.pop(0)
    source.refresh()
    cameras.insert(0, InputCamera(device_id="camera-a", description="Desk camera"))
    source.refresh()
    controller.change_mirroring(mirrored=False)
    controller.shutdown()

    assert controller.selected_camera_id == "camera-a"
    assert preview.starts == [
        ("camera-b", True, "1080p"),
        ("camera-a", True, "1080p"),
        ("camera-b", True, "1080p"),
        ("camera-a", True, "1080p"),
    ]
    assert preview.mirroring == [False]
    assert preview.stops == 4

    restored_preview = RecordingPreview()
    restored_parent = QWidget()
    restored = LivePreviewController(
        restored_parent,
        ShowPage([], lambda: restored_preview),
        restored_preview,
        source,
        tmp_path,
    )
    restored.start()
    assert restored.selected_camera_id == "camera-a"
    assert restored.mirrored is False
    restored.shutdown()


def test_live_controller_persists_and_immediately_applies_input_resolution(
    tmp_path: Path,
) -> None:
    """Restart capture at 720p only after an explicit user selection."""
    QApplication.instance() or QApplication([])
    source = InputCameraSource(
        lambda: (InputCamera(device_id="camera-a", description="Desk camera"),),
    )
    preview = RecordingPreview()
    parent = QWidget()
    controller = LivePreviewController(
        parent,
        ShowPage([], lambda: preview),
        preview,
        source,
        tmp_path,
    )
    controller.start()

    controller.change_input_resolution("720p")

    assert controller.input_resolution == "720p"
    assert preview.starts == [
        ("camera-a", True, "1080p"),
        ("camera-a", True, "720p"),
    ]
    assert preview.stops == 1

    restored_preview = RecordingPreview()
    restored_parent = QWidget()
    restored = LivePreviewController(
        restored_parent,
        ShowPage([], lambda: restored_preview),
        restored_preview,
        source,
        tmp_path,
    )
    restored.start()

    assert restored.input_resolution == "720p"
    assert restored_preview.starts == [("camera-a", True, "720p")]
    controller.shutdown()
    restored.shutdown()
