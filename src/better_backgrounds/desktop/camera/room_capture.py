"""Countdown-driven webcam capture of the empty room for the Build page."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from better_backgrounds.desktop.camera.capture import QtCameraCapture

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray
    from PySide6.QtCore import SignalInstance

    from better_backgrounds.desktop.camera.devices import InputCameraSource

COUNTDOWN_SECONDS = 10
RGB_CHANNELS = 3


class _CapturePage(Protocol):
    @property
    def capture_requested(self) -> SignalInstance: ...

    @property
    def capture_now_requested(self) -> SignalInstance: ...

    @property
    def capture_cancelled(self) -> SignalInstance: ...

    def show_capture(self) -> None: ...

    def show_upload(self) -> None: ...

    def set_capture_frame(self, image: QImage) -> None: ...

    def set_countdown(self, seconds: int) -> None: ...

    def set_capture_error(self, message: str) -> None: ...


class _CameraCapture(Protocol):
    @property
    def frame_captured(self) -> SignalInstance: ...

    @property
    def failed(self) -> SignalInstance: ...

    def start(self, device_id: str) -> bool: ...

    def stop(self) -> None: ...


class RoomCaptureController(QObject):
    """Own the webcam handle, countdown, and empty-room frame capture."""

    captured = Signal(object)
    active = Signal(bool)

    def __init__(
        self,
        page: _CapturePage,
        camera_source: InputCameraSource,
        selected_camera_id: Callable[[], str | None],
        capture_root: Path,
        *,
        capture: _CameraCapture | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Connect Build-page capture controls to an isolated camera session."""
        super().__init__(parent)
        self._page = page
        self._camera_source = camera_source
        self._selected_camera_id = selected_camera_id
        self._capture_root = capture_root
        self._capture = capture or QtCameraCapture(self)
        self._capture.frame_captured.connect(self._frame_captured)
        self._capture.failed.connect(self._camera_failed)
        self._timer = QTimer(self)
        self._timer.setInterval(1_000)
        self._timer.timeout.connect(self._tick)
        self._latest_frame: NDArray[np.uint8] | None = None
        self._remaining = 0
        self._running = False
        self._awaiting_capture = False
        page.capture_requested.connect(self.start)
        page.capture_now_requested.connect(self.capture_now)
        page.capture_cancelled.connect(self.cancel)

    @Slot()
    def start(self) -> None:
        """Open the selected camera and begin the step-away countdown."""
        if self._running:
            return
        device_id = self._resolve_device_id()
        self._running = True
        self._latest_frame = None
        self._awaiting_capture = False
        self.active.emit(True)  # noqa: FBT003
        self._page.show_capture()
        if device_id is None or not self._capture.start(device_id):
            self._page.set_capture_error(
                "No camera is available to capture the room. Connect a webcam and try again.",
            )
            self._finish()
            return
        self._remaining = COUNTDOWN_SECONDS
        self._page.set_countdown(self._remaining)
        self._timer.start()

    @Slot()
    def capture_now(self) -> None:
        """Skip the remaining countdown and capture the next available frame."""
        if not self._running:
            return
        self._timer.stop()
        self._perform_capture()

    @Slot()
    def cancel(self) -> None:
        """Stop the camera and countdown and return to the upload surface."""
        if not self._running:
            return
        self._finish()
        self._page.show_upload()

    @Slot()
    def _tick(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            self._timer.stop()
            self._perform_capture()
        else:
            self._page.set_countdown(self._remaining)

    @Slot(object, float)
    def _frame_captured(self, frame: object, _captured_at: float) -> None:
        if not isinstance(frame, np.ndarray):
            return
        source = cast("NDArray[np.uint8]", frame)
        self._latest_frame = source
        mirrored = np.ascontiguousarray(np.flip(source, axis=1))
        height, width = mirrored.shape[:2]
        image = QImage(
            mirrored.data,
            width,
            height,
            mirrored.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._page.set_capture_frame(image)
        if self._awaiting_capture:
            self._perform_capture()

    @Slot(str)
    def _camera_failed(self, message: str) -> None:
        if not self._running:
            return
        self._page.set_capture_error(message)
        self._finish()

    def _perform_capture(self) -> None:
        frame = self._latest_frame
        if frame is None:
            self._awaiting_capture = True
            return
        self._awaiting_capture = False
        try:
            path = self._save_capture(frame)
        except OSError as error:
            self._page.set_capture_error(f"Could not save the captured room: {str(error)[:200]}")
            self._finish()
            return
        self._finish()
        self.captured.emit(path)

    def _save_capture(self, frame: NDArray[np.uint8]) -> Path:
        self._capture_root.mkdir(parents=True, exist_ok=True)
        path = self._capture_root / f"{uuid4().hex}.png"
        Image.fromarray(frame, mode="RGB").save(path, format="PNG")
        self._prune_previous(keep=path)
        return path

    def _prune_previous(self, *, keep: Path) -> None:
        for previous in self._capture_root.glob("*.png"):
            if previous == keep:
                continue
            try:
                previous.unlink()
            except OSError:
                continue

    def _resolve_device_id(self) -> str | None:
        selected = self._selected_camera_id()
        cameras = self._camera_source.cameras()
        available = {camera.device_id for camera in cameras}
        if selected in available:
            return selected
        default = next((camera for camera in cameras if camera.is_default), None)
        if default is not None:
            return default.device_id
        return cameras[0].device_id if cameras else None

    def _finish(self) -> None:
        self._timer.stop()
        self._capture.stop()
        self._latest_frame = None
        self._awaiting_capture = False
        if self._running:
            self._running = False
            self.active.emit(False)  # noqa: FBT003

    def shutdown(self) -> None:
        """Release the camera if a capture is still in flight."""
        self._finish()
