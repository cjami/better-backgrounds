"""OBS-backed virtual-camera publication for the desktop application."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    from numpy.typing import NDArray

OUTPUT_FRAME_RATE = 30.0
STALE_FRAME_SECONDS = 0.5
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
BACKGROUND_RGB = (14, 15, 18)
ACCENT_RGB = (224, 163, 74)
TEXT_RGB = (231, 232, 234)
VirtualCameraPhase = Literal[
    "unavailable",
    "inactive",
    "starting",
    "active",
    "stopping",
    "failed",
]


@dataclass(frozen=True, slots=True)
class VirtualCameraProfile:
    """Describe one fixed OBS output mode selected before publication."""

    profile_id: str
    label: str
    width: int
    height: int
    frame_rate: float = OUTPUT_FRAME_RATE


FULL_HD_PROFILE = VirtualCameraProfile("1080p", "1080p", 1920, 1080)
HD_PROFILE = VirtualCameraProfile("720p", "720p", 1280, 720)
VIRTUAL_CAMERA_PROFILES = {
    FULL_HD_PROFILE.profile_id: FULL_HD_PROFILE,
    HD_PROFILE.profile_id: HD_PROFILE,
}


class VirtualCameraSink(Protocol):
    """Accept fixed-size RGB frames for one virtual-camera session."""

    def send(self, frame: NDArray[np.uint8]) -> None:
        """Publish one RGB frame."""

    def sleep_until_next_frame(self) -> None:
        """Wait until the next output deadline."""

    def close(self) -> None:
        """Release the virtual-camera producer."""


VirtualCameraSinkFactory = Callable[[VirtualCameraProfile], VirtualCameraSink]


@dataclass(frozen=True, slots=True)
class VirtualCameraState:
    """Describe the confirmed virtual-camera lifecycle for presentation."""

    phase: VirtualCameraPhase
    message: str = ""

    @property
    def active(self) -> bool:
        """Return whether frames are currently being published."""
        return self.phase == "active"


def fit_virtual_camera_frame(
    frame: NDArray[np.uint8],
    profile: VirtualCameraProfile = FULL_HD_PROFILE,
) -> NDArray[np.uint8]:
    """Fit one RGB composite into the selected canvas without stretching."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != RGB_DIMENSIONS
        or frame.shape[2] != RGB_CHANNELS
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
    ):
        msg = "virtual-camera frames must be non-empty uint8 RGB images"
        raise ValueError(msg)
    source_height, source_width = frame.shape[:2]
    scale = min(profile.width / source_width, profile.height / source_height)
    target_width = max(1, min(profile.width, round(source_width * scale)))
    target_height = max(1, min(profile.height, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
    output = np.full(
        (profile.height, profile.width, RGB_CHANNELS),
        BACKGROUND_RGB,
        dtype=np.uint8,
    )
    left = (profile.width - target_width) // 2
    top = (profile.height - target_height) // 2
    output[top : top + target_height, left : left + target_width] = resized
    return np.ascontiguousarray(output)


def create_waiting_frame(
    profile: VirtualCameraProfile = FULL_HD_PROFILE,
) -> NDArray[np.uint8]:
    """Create the privacy-safe frame shown while no fresh composite is available."""
    frame = np.full(
        (profile.height, profile.width, RGB_CHANNELS),
        BACKGROUND_RGB,
        dtype=np.uint8,
    )
    title = "BETTER BACKGROUNDS"
    subtitle = "Waiting for live preview"
    title_font = cv2.FONT_HERSHEY_DUPLEX
    subtitle_font = cv2.FONT_HERSHEY_SIMPLEX
    size_scale = profile.height / FULL_HD_PROFILE.height
    title_scale = 1.6 * size_scale
    subtitle_scale = 0.85 * size_scale
    title_thickness = max(1, round(2 * size_scale))
    subtitle_thickness = max(1, round(2 * size_scale))
    title_size = cv2.getTextSize(title, title_font, title_scale, title_thickness)[0]
    subtitle_size = cv2.getTextSize(
        subtitle,
        subtitle_font,
        subtitle_scale,
        subtitle_thickness,
    )[0]
    centre_y = profile.height // 2
    cv2.putText(
        frame,
        title,
        ((profile.width - title_size[0]) // 2, centre_y - round(22 * size_scale)),
        title_font,
        title_scale,
        ACCENT_RGB,
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        ((profile.width - subtitle_size[0]) // 2, centre_y + round(42 * size_scale)),
        subtitle_font,
        subtitle_scale,
        TEXT_RGB,
        subtitle_thickness,
        cv2.LINE_AA,
    )
    return frame


def create_obs_virtual_camera(profile: VirtualCameraProfile) -> VirtualCameraSink:
    """Open the single OBS Virtual Camera producer using its Python backend."""
    import pyvirtualcam  # noqa: PLC0415

    return pyvirtualcam.Camera(
        width=profile.width,
        height=profile.height,
        fps=profile.frame_rate,
        fmt=pyvirtualcam.PixelFormat.RGB,
        device="OBS Virtual Camera",
        backend="obs",
    )


def _startup_error_message(error: BaseException) -> str:
    detail = str(error).strip().lower()
    occupied_markers = (
        "already",
        "busy",
        "in use",
        "occupied",
        "0x800700aa",
        "virtual camera output could not be started",
    )
    if any(fragment in detail for fragment in occupied_markers):
        return (
            "OBS Virtual Camera is already in use. Stop it in OBS or another application, "
            "then retry."
        )
    if sys.platform == "darwin":
        return (
            "OBS Virtual Camera is unavailable. Install OBS Studio 30+, start and stop its "
            "Virtual Camera once, approve it in System Settings, then retry."
        )
    return (
        "OBS Virtual Camera is unavailable. Install OBS Studio 30+ with its Virtual Camera "
        "component, or stop another application using it, then retry."
    )


class VirtualCameraController(QObject):
    """Publish the latest composite at a stable cadence without blocking Qt."""

    state_changed = Signal(object)
    active_changed = Signal(bool)
    _worker_started = Signal(int)
    _worker_stopped = Signal(int)
    _worker_failed = Signal(int, str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        sink_factory: VirtualCameraSinkFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an inactive publisher and injectable OBS boundary."""
        super().__init__(parent)
        self._supported = sink_factory is not None or sys.platform in {"win32", "darwin"}
        self._sink_factory = sink_factory or create_obs_virtual_camera
        self._clock = clock
        self._lock = threading.Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_received_at = 0.0
        self._latest_revision = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._generation = 0
        self._profile = FULL_HD_PROFILE
        self._state = VirtualCameraState(
            "inactive" if self._supported else "unavailable",
            "" if self._supported else "Virtual camera is supported on Windows and macOS.",
        )
        self._worker_started.connect(self._accept_started)
        self._worker_stopped.connect(self._accept_stopped)
        self._worker_failed.connect(self._accept_failed)

    @property
    def state(self) -> VirtualCameraState:
        """Return the last confirmed publisher state."""
        return self._state

    @property
    def profile(self) -> VirtualCameraProfile:
        """Return the output mode used for the next publication session."""
        return self._profile

    @Slot(str)
    def select_profile(self, profile_id: str) -> None:
        """Select a fixed OBS output mode while publication is inactive."""
        profile = VIRTUAL_CAMERA_PROFILES.get(profile_id)
        if profile is not None and self._state.phase in {"inactive", "failed"}:
            self._profile = profile

    @Slot(bool)
    def request_active(self, active: bool) -> None:  # noqa: FBT001
        """Start or stop publication in response to an explicit UI request."""
        if active:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        """Open OBS Virtual Camera on a dedicated publication thread."""
        if not self._supported:
            self._set_state(self._state)
            return
        thread = self._thread
        if thread is not None and thread.is_alive() and self._state.phase in {"inactive", "failed"}:
            thread.join(timeout=0.05)
        if self._state.phase in {"starting", "active", "stopping"} or (
            thread is not None and thread.is_alive()
        ):
            return
        self._generation += 1
        generation = self._generation
        profile = self._profile
        self._stop_event = threading.Event()
        self._set_state(VirtualCameraState("starting", "Connecting to OBS Virtual Camera…"))
        self._thread = threading.Thread(
            target=self._run,
            args=(generation, profile, self._stop_event),
            name="virtual-camera-output",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """Request output teardown, optionally waiting during application shutdown."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            if self._state.phase != "unavailable":
                self._set_state(VirtualCameraState("inactive"))
            return
        if self._state.phase != "stopping":
            self._set_state(VirtualCameraState("stopping", "Stopping virtual camera…"))
        self._stop_event.set()
        if wait and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    @Slot(object, float)
    def publish_frame(self, frame: object, captured_at: float) -> None:
        """Replace the pending composite without performing output-sized work on Qt."""
        _ = captured_at
        if not isinstance(frame, np.ndarray):
            return
        if (
            frame.dtype != np.uint8
            or frame.ndim != RGB_DIMENSIONS
            or frame.shape[2] != RGB_CHANNELS
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
        ):
            return
        typed_frame = cast("NDArray[np.uint8]", frame)
        with self._lock:
            self._latest_frame = typed_frame
            self._latest_received_at = self._clock()
            self._latest_revision += 1

    def shutdown(self) -> None:
        """Stop publication before the desktop and its Qt objects are destroyed."""
        self.stop(wait=True)

    def _run(
        self,
        generation: int,
        profile: VirtualCameraProfile,
        stop_event: threading.Event,
    ) -> None:
        sink: VirtualCameraSink | None = None
        try:
            sink = self._sink_factory(profile)
        except Exception as error:  # noqa: BLE001
            self._worker_failed.emit(generation, _startup_error_message(error))
            return
        self._worker_started.emit(generation)
        waiting = create_waiting_frame(profile)
        output = waiting
        prepared_revision = -1
        failure_message: str | None = None
        try:
            while not stop_event.is_set():
                with self._lock:
                    latest = self._latest_frame
                    received_at = self._latest_received_at
                    revision = self._latest_revision
                if latest is None or self._clock() - received_at > STALE_FRAME_SECONDS:
                    output = waiting
                elif revision != prepared_revision:
                    output = fit_virtual_camera_frame(latest, profile)
                    prepared_revision = revision
                sink.send(output)
                sink.sleep_until_next_frame()
        except Exception as error:  # noqa: BLE001
            failure_message = f"Virtual camera stopped: {str(error)[:240]}"
        finally:
            with suppress(Exception):  # Backend teardown must not escape its worker.
                sink.close()
        if failure_message is None:
            self._worker_stopped.emit(generation)
        else:
            self._worker_failed.emit(generation, failure_message)

    @Slot(int)
    def _accept_started(self, generation: int) -> None:
        if generation != self._generation or self._state.phase != "starting":
            return
        self._set_state(
            VirtualCameraState(
                "active",
                f"Publishing {self._profile.label} via OBS Virtual Camera",
            ),
        )
        self.active_changed.emit(True)  # noqa: FBT003

    @Slot(int)
    def _accept_stopped(self, generation: int) -> None:
        if generation != self._generation:
            return
        was_active = self._state.phase in {"active", "stopping"}
        self._thread = None
        self._set_state(VirtualCameraState("inactive"))
        if was_active:
            self.active_changed.emit(False)  # noqa: FBT003

    @Slot(int, str)
    def _accept_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        was_active = self._state.phase in {"active", "stopping"}
        self._thread = None
        self._set_state(VirtualCameraState("failed", message))
        if was_active:
            self.active_changed.emit(False)  # noqa: FBT003

    def _set_state(self, state: VirtualCameraState) -> None:
        self._state = state
        self.state_changed.emit(state)
