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


class FrameRateLimiter:
    """Sample an overproducing capture backend at a stable target cadence."""

    def __init__(self, *, target_frame_rate: float) -> None:
        """Create a monotonic deadline sequence for accepted frames."""
        if target_frame_rate <= 0:
            msg = "target frame rate must be positive"
            raise ValueError(msg)
        self._interval_ms = 1_000.0 / target_frame_rate
        self._next_frame_at_ms: float | None = None

    def allows(self, captured_at_ms: float) -> bool:
        """Accept the nearest available frame at each target deadline."""
        deadline = self._next_frame_at_ms
        if deadline is None:
            self._next_frame_at_ms = captured_at_ms + self._interval_ms
            return True
        if captured_at_ms < deadline:
            return False
        next_deadline = deadline + self._interval_ms
        self._next_frame_at_ms = (
            captured_at_ms + self._interval_ms
            if next_deadline <= captured_at_ms - self._interval_ms
            else next_deadline
        )
        return True

    def reset(self) -> None:
        """Forget timing from the previous camera session."""
        self._next_frame_at_ms = None


def camera_format_score(
    width: int,
    height: int,
    minimum_frame_rate: float,
    maximum_frame_rate: float,
) -> tuple[int, float, float, int]:
    """Rank capture formats by 720p fidelity and a real 30 fps delivery rate."""
    resolution_distance = abs(width - TARGET_WIDTH) + abs(height - TARGET_HEIGHT)
    below_quality_target = int(width < TARGET_WIDTH or height < TARGET_HEIGHT)
    frame_rate_distance = (
        0.0
        if minimum_frame_rate <= TARGET_FRAME_RATE <= maximum_frame_rate
        else min(
            abs(minimum_frame_rate - TARGET_FRAME_RATE),
            abs(maximum_frame_rate - TARGET_FRAME_RATE),
        )
    )
    return (
        below_quality_target,
        frame_rate_distance,
        abs(maximum_frame_rate - TARGET_FRAME_RATE),
        resolution_distance,
    )


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
        self._rate_limiter = FrameRateLimiter(target_frame_rate=TARGET_FRAME_RATE)

    def start(self, device_id: str) -> bool:
        """Open one current Qt device identifier at the closest 720p format."""
        self.stop()
        self._rate_limiter.reset()
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
                key=lambda item: camera_format_score(
                    item.resolution().width(),
                    item.resolution().height(),
                    item.minFrameRate(),
                    item.maxFrameRate(),
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
            self._rate_limiter.reset()
            return
        self._camera.stop()
        self._camera.deleteLater()
        self._camera = None
        self._rate_limiter.reset()

    @Slot(QVideoFrame)
    def _video_frame_changed(self, frame: QVideoFrame) -> None:
        if not frame.isValid():
            return
        captured_at = time.monotonic() * 1000.0
        if not self._rate_limiter.allows(captured_at):
            return
        image = frame.toImage()
        if image.isNull():
            return
        self.frame_captured.emit(qimage_to_rgb(image), captured_at)

    @Slot(QCamera.Error, str)
    def _camera_failed(self, _error: QCamera.Error, message: str) -> None:
        self.failed.emit(f"Camera failed: {message[:240]}")


def _camera_identifier(device) -> str:  # noqa: ANN001
    return bytes(device.id().toHex().data()).decode("ascii")
