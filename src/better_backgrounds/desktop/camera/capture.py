"""Qt Multimedia webcam capture isolated from the live-session UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
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

from better_backgrounds.desktop.camera.devices import (
    DEFAULT_INPUT_RESOLUTION,
    InputResolution,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
MINIMUM_ASPECT_RATIO = 0.5
MAXIMUM_ASPECT_RATIO = 4.0
FULL_HD_WIDTH = 1920
FULL_HD_HEIGHT = 1080
HD_WIDTH = 1280
HD_HEIGHT = 720
TARGET_FRAME_RATE = 30.0
CAPTURE_JITTER_TOLERANCE = 0.1
MINIMUM_TARGET_FRAME_RATE = TARGET_FRAME_RATE * 0.95


@dataclass(frozen=True, slots=True)
class OutputGeometry:
    """Describe one aspect-correct output canvas at the camera's quality tier."""

    width: int
    height: int
    aspect_ratio: float

    def __post_init__(self) -> None:
        """Reject invalid geometry before it reaches shared-memory allocation."""
        if (
            self.width <= 0
            or self.height <= 0
            or not MINIMUM_ASPECT_RATIO <= self.aspect_ratio <= MAXIMUM_ASPECT_RATIO
        ):
            msg = "output geometry must have positive dimensions and a supported aspect ratio"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CaptureProfile:
    """Record native capture and normalized live-processing dimensions."""

    native_width: int
    native_height: int
    minimum_frame_rate: float
    maximum_frame_rate: float
    processing_width: int
    processing_height: int

    def output_geometry(self, aspect_ratio: float) -> OutputGeometry:
        """Create the requested output aspect at this profile's vertical tier."""
        width = max(1, round(self.processing_height * aspect_ratio))
        return OutputGeometry(width=width, height=self.processing_height, aspect_ratio=aspect_ratio)


def capture_profile(
    width: int,
    height: int,
    minimum_frame_rate: float,
    maximum_frame_rate: float,
    resolution: InputResolution = DEFAULT_INPUT_RESOLUTION,
) -> CaptureProfile:
    """Normalize capture to the user-selected tier without upscaling."""
    if width <= 0 or height <= 0:
        msg = "capture dimensions must be positive"
        raise ValueError(msg)
    target_height = FULL_HD_HEIGHT if resolution == "1080p" else HD_HEIGHT
    if height >= target_height:
        processing_height = target_height
        processing_width = round(width * processing_height / height)
    else:
        processing_width, processing_height = width, height
    return CaptureProfile(
        native_width=width,
        native_height=height,
        minimum_frame_rate=minimum_frame_rate,
        maximum_frame_rate=maximum_frame_rate,
        processing_width=processing_width,
        processing_height=processing_height,
    )


def normalize_capture_frame(
    frame: NDArray[np.uint8],
    profile: CaptureProfile,
) -> NDArray[np.uint8]:
    """Downsample higher-resolution capture once without upscaling smaller feeds."""
    expected = (profile.native_height, profile.native_width, RGB_CHANNELS)
    if frame.dtype != np.uint8 or frame.shape != expected:
        msg = f"captured frame must be {expected} uint8 RGB"
        raise ValueError(msg)
    target = (profile.processing_width, profile.processing_height)
    if target == (profile.native_width, profile.native_height):
        return frame
    return cast(
        "NDArray[np.uint8]",
        cv2.resize(frame, target, interpolation=cv2.INTER_AREA),
    )


def fit_frame_to_output(
    source: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    geometry: OutputGeometry,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Crop or background-pad matching source and alpha without stretching."""
    if source.dtype != np.uint8 or source.ndim != RGB_DIMENSIONS or source.shape[2] != RGB_CHANNELS:
        msg = "output source must be uint8 RGB"
        raise ValueError(msg)
    if alpha.dtype != np.uint8 or alpha.shape != source.shape[:2]:
        msg = "output alpha must be uint8 and match source"
        raise ValueError(msg)
    height, width = source.shape[:2]
    target_width, target_height = geometry.width, geometry.height
    if height > target_height:
        scale = target_height / height
        resized_width = max(1, round(width * scale))
        source = cast(
            "NDArray[np.uint8]",
            cv2.resize(source, (resized_width, target_height), interpolation=cv2.INTER_AREA),
        )
        alpha = cast(
            "NDArray[np.uint8]",
            cv2.resize(alpha, (resized_width, target_height), interpolation=cv2.INTER_AREA),
        )
        height, width = source.shape[:2]
    if height < target_height:
        target_height = height
        target_width = max(1, round(target_height * geometry.aspect_ratio))
    if width >= target_width:
        left = (width - target_width) // 2
        return (
            np.ascontiguousarray(source[:, left : left + target_width]),
            np.ascontiguousarray(alpha[:, left : left + target_width]),
        )
    fitted_source = np.zeros((target_height, target_width, RGB_CHANNELS), dtype=np.uint8)
    fitted_alpha = np.zeros((target_height, target_width), dtype=np.uint8)
    left = (target_width - width) // 2
    fitted_source[:, left : left + width] = source
    fitted_alpha[:, left : left + width] = alpha
    return fitted_source, fitted_alpha


class FrameRateLimiter:
    """Sample an overproducing capture backend at a stable target cadence."""

    def __init__(self, *, target_frame_rate: float) -> None:
        """Create a monotonic deadline sequence for accepted frames."""
        if target_frame_rate <= 0:
            msg = "target frame rate must be positive"
            raise ValueError(msg)
        self._interval_ms = 1_000.0 / target_frame_rate
        self._tolerance_ms = self._interval_ms * CAPTURE_JITTER_TOLERANCE
        self._last_frame_at_ms: float | None = None

    def allows(self, captured_at_ms: float) -> bool:
        """Accept the nearest available frame at each target deadline."""
        previous = self._last_frame_at_ms
        if previous is None:
            self._last_frame_at_ms = captured_at_ms
            return True
        if captured_at_ms - previous + self._tolerance_ms < self._interval_ms:
            return False
        self._last_frame_at_ms = captured_at_ms
        return True

    def reset(self) -> None:
        """Forget timing from the previous camera session."""
        self._last_frame_at_ms = None


def camera_format_score(
    width: int,
    height: int,
    minimum_frame_rate: float,
    maximum_frame_rate: float,
    resolution: InputResolution = DEFAULT_INPUT_RESOLUTION,
) -> tuple[int, int, int, int, float, float]:
    """Rank formats by the requested resolution tier, then frame rate."""
    if resolution == "1080p":
        target_width, target_height = FULL_HD_WIDTH, FULL_HD_HEIGHT
    else:
        target_width, target_height = HD_WIDTH, HD_HEIGHT
    height_distance = abs(height - target_height)
    resolution_distance = abs(width - target_width) + abs(height - target_height)
    below_rate_target = int(maximum_frame_rate < MINIMUM_TARGET_FRAME_RATE)
    misses_exact_rate = int(
        not minimum_frame_rate <= TARGET_FRAME_RATE <= maximum_frame_rate,
    )
    frame_rate_distance = (
        0.0
        if minimum_frame_rate <= TARGET_FRAME_RATE <= maximum_frame_rate
        else min(
            abs(minimum_frame_rate - TARGET_FRAME_RATE),
            abs(maximum_frame_rate - TARGET_FRAME_RATE),
        )
    )
    return (
        height_distance,
        resolution_distance,
        below_rate_target,
        misses_exact_rate,
        frame_rate_distance,
        abs(maximum_frame_rate - TARGET_FRAME_RATE),
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
    profile_changed = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a retained capture session without opening a device."""
        super().__init__(parent)
        self._session = QMediaCaptureSession(self)
        self._sink = QVideoSink(self)
        self._sink.videoFrameChanged.connect(self._video_frame_changed)
        self._session.setVideoOutput(self._sink)
        self._camera: QCamera | None = None
        self._profile: CaptureProfile | None = None
        self._rate_limiter = FrameRateLimiter(target_frame_rate=TARGET_FRAME_RATE)

    @property
    def active_profile(self) -> CaptureProfile | None:
        """Return the selected native and processing dimensions."""
        return self._profile

    def start(
        self,
        device_id: str,
        resolution: InputResolution = DEFAULT_INPUT_RESOLUTION,
    ) -> bool:
        """Open one current Qt device identifier at the selected quality tier."""
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
                    resolution,
                ),
            )
            camera.setCameraFormat(selected_format)
            selected_resolution = selected_format.resolution()
            self._profile = capture_profile(
                selected_resolution.width(),
                selected_resolution.height(),
                selected_format.minFrameRate(),
                selected_format.maxFrameRate(),
                resolution,
            )
            self.profile_changed.emit(self._profile)
        else:
            self._profile = None
        camera.errorOccurred.connect(self._camera_failed)
        self._session.setCamera(camera)
        self._camera = camera
        camera.start()
        return True

    def stop(self) -> None:
        """Stop and release the current native camera handle."""
        if self._camera is None:
            self._profile = None
            self._rate_limiter.reset()
            return
        self._camera.stop()
        self._camera.deleteLater()
        self._camera = None
        self._profile = None
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
        source = qimage_to_rgb(image)
        profile = self._profile
        if profile is not None and source.shape[:2] == (
            profile.native_height,
            profile.native_width,
        ):
            source = normalize_capture_frame(source, profile)
        self.frame_captured.emit(source, captured_at)

    @Slot(QCamera.Error, str)
    def _camera_failed(self, _error: QCamera.Error, message: str) -> None:
        self.failed.emit(f"Camera failed: {message[:240]}")


def _camera_identifier(device) -> str:  # noqa: ANN001
    return bytes(device.id().toHex().data()).decode("ascii")
