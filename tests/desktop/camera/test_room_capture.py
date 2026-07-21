"""Countdown webcam room-capture controller tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.camera import InputCameraSource, RoomCaptureController
from better_backgrounds.desktop.camera.devices import InputCamera
from better_backgrounds.desktop.camera.room_capture import COUNTDOWN_SECONDS


class FakeCapture(QObject):
    """Stand in for QtCameraCapture without opening a device."""

    frame_captured = Signal(object, float)
    profile_changed = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        """Record start and stop calls."""
        super().__init__()
        self.started: list[str] = []
        self.stops = 0
        self.result = True

    def start(self, device_id: str) -> bool:
        """Record the opened device and report success."""
        self.started.append(device_id)
        return self.result

    def stop(self) -> None:
        """Record a release."""
        self.stops += 1


class FakePage(QObject):
    """Provide the Build-page capture surface used by the controller."""

    capture_requested = Signal()
    capture_now_requested = Signal()
    capture_cancelled = Signal()

    def __init__(self) -> None:
        """Record the capture-surface updates."""
        super().__init__()
        self.showed_capture = 0
        self.showed_upload = 0
        self.frames: list[object] = []
        self.countdowns: list[int] = []
        self.errors: list[str] = []

    def show_capture(self) -> None:
        """Record entry to the capture state."""
        self.showed_capture += 1

    def show_upload(self) -> None:
        """Record the return to upload."""
        self.showed_upload += 1

    def set_capture_frame(self, image: object) -> None:
        """Record one live preview frame."""
        self.frames.append(image)

    def set_countdown(self, seconds: int) -> None:
        """Record one countdown update."""
        self.countdowns.append(seconds)

    def set_capture_error(self, message: str) -> None:
        """Record a capture failure."""
        self.errors.append(message)


def _asymmetric_frame() -> np.ndarray:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 3:] = 255
    return frame


def _make_controller(
    tmp_path: Path,
    *,
    cameras: tuple[InputCamera, ...] = (InputCamera("cam", "Cam", is_default=True),),
    selected: str | None = "cam",
) -> tuple[RoomCaptureController, FakePage, FakeCapture, list[bool]]:
    QApplication.instance() or QApplication([])
    page = FakePage()
    capture = FakeCapture()
    controller = RoomCaptureController(
        page,
        InputCameraSource(provider=lambda: cameras),
        lambda: selected,
        tmp_path,
        capture=capture,
    )
    active: list[bool] = []
    controller.active.connect(active.append)
    return controller, page, capture, active


def test_capture_now_saves_mirrored_frame(tmp_path: Path) -> None:
    """Save the room with the same familiar mirroring shown in the preview."""
    controller, page, capture, active = _make_controller(tmp_path)
    captured: list[object] = []
    controller.captured.connect(captured.append)

    controller.start()
    assert active == [True]
    assert capture.started == ["cam"]
    assert page.showed_capture == 1
    assert page.countdowns[0] == COUNTDOWN_SECONDS

    frame = _asymmetric_frame()
    capture.frame_captured.emit(frame, 0.0)
    controller.capture_now()

    assert active == [True, False]
    assert capture.stops >= 1
    assert len(captured) == 1
    saved = captured[0]
    assert isinstance(saved, Path)
    with Image.open(saved) as opened:
        stored = np.asarray(opened.convert("RGB"))
    assert np.array_equal(stored, np.flip(frame, axis=1))


def test_countdown_reaching_zero_captures(tmp_path: Path) -> None:
    """Auto-capture once the countdown ticks down to zero."""
    controller, _page, capture, _active = _make_controller(tmp_path)
    captured: list[object] = []
    controller.captured.connect(captured.append)

    controller.start()
    capture.frame_captured.emit(_asymmetric_frame(), 0.0)
    for _ in range(COUNTDOWN_SECONDS):
        controller._tick()  # noqa: SLF001

    assert len(captured) == 1


def test_cancel_releases_camera_without_capture(tmp_path: Path) -> None:
    """Cancelling stops the camera and returns to upload without a capture."""
    controller, page, capture, active = _make_controller(tmp_path)
    captured: list[object] = []
    controller.captured.connect(captured.append)

    controller.start()
    controller.cancel()

    assert active == [True, False]
    assert capture.stops >= 1
    assert page.showed_upload == 1
    assert captured == []


def test_missing_camera_reports_error(tmp_path: Path) -> None:
    """Report a friendly error and release when no camera is available."""
    controller, page, capture, active = _make_controller(tmp_path, cameras=(), selected=None)

    controller.start()

    assert capture.started == []
    assert page.errors
    assert active == [True, False]
