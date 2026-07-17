"""Qt Multimedia webcam capture isolated from the live-session UI."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import (
    QCamera,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoSink,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

RGB_CHANNELS = 3
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FRAME_RATE = 30.0


def qimage_to_rgb(image: QImage) -> NDArray[np.uint8]:
    """Copy one Qt image into tightly packed RGB pixels."""
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    width = converted.width()
    height = converted.height()
    rows = np.frombuffer(converted.bits(), dtype=np.uint8, count=converted.sizeInBytes()).reshape(
        height,
        converted.bytesPerLine(),
    )
    return rows[:, : width * RGB_CHANNELS].reshape(height, width, RGB_CHANNELS).copy()


class QtCameraCapture(QObject):
    """Own one QCamera and publish copied RGB frames with capture timestamps."""

    frame_captured = Signal(object, float)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a retained capture session without opening a device."""
        super().__init__(parent)
        self._session = QMediaCaptureSession(self)
        self._sink = QVideoSink(self)
        self._sink.videoFrameChanged.connect(self._video_frame_changed)
        self._session.setVideoOutput(self._sink)
        self._camera: QCamera | None = None

    def start(self, device_id: str) -> bool:
        """Open one current Qt device identifier at the closest 720p format."""
        self.stop()
        device = next(
            (
                candidate
                for candidate in QMediaDevices.videoInputs()
                if _camera_identifier(candidate) == device_id
            ),
            None,
        )
        if device is None:
            self.failed.emit("Selected camera is no longer available")
            return False
        camera = QCamera(device, self)
        formats = device.videoFormats()
        if formats:
            selected_format = min(
                formats,
                key=lambda item: (
                    abs(item.resolution().width() - TARGET_WIDTH)
                    + abs(item.resolution().height() - TARGET_HEIGHT),
                    abs(min(item.maxFrameRate(), TARGET_FRAME_RATE) - TARGET_FRAME_RATE) * 10,
                ),
            )
            camera.setCameraFormat(selected_format)
        camera.errorOccurred.connect(self._camera_failed)
        self._session.setCamera(camera)
        self._camera = camera
        camera.start()
        return True

    def stop(self) -> None:
        """Stop and release the current native camera handle."""
        if self._camera is None:
            return
        self._camera.stop()
        self._camera.deleteLater()
        self._camera = None

    @Slot(QVideoFrame)
    def _video_frame_changed(self, frame: QVideoFrame) -> None:
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        self.frame_captured.emit(qimage_to_rgb(image), time.monotonic() * 1000.0)

    @Slot(QCamera.Error, str)
    def _camera_failed(self, _error: QCamera.Error, message: str) -> None:
        self.failed.emit(f"Camera failed: {message[:240]}")


def _camera_identifier(device) -> str:  # noqa: ANN001
    return bytes(device.id().toHex().data()).decode("ascii")
