"""Tests for bounded OBS virtual-camera publication."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Never

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.virtual_camera import (
    BACKGROUND_RGB,
    FULL_HD_PROFILE,
    HD_PROFILE,
    VirtualCameraController,
    VirtualCameraProfile,
    create_obs_virtual_camera,
    fit_virtual_camera_frame,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RecordingSink:
    """Record publication calls while maintaining a short deterministic cadence."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        """Create a sink that optionally disconnects after a frame count."""
        self.frames: list[NDArray[np.uint8]] = []
        self.closed = False
        self._fail_after = fail_after

    def send(self, frame: NDArray[np.uint8]) -> None:
        """Record one independently owned output frame."""
        if self._fail_after is not None and len(self.frames) >= self._fail_after:
            msg = "backend disconnected"
            raise RuntimeError(msg)
        self.frames.append(frame.copy())

    def sleep_until_next_frame(self) -> None:
        """Model the backend cadence without slowing the test suite."""
        time.sleep(0.002)

    def close(self) -> None:
        """Record bounded backend teardown."""
        self.closed = True


def _application() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def _wait_until(predicate, *, timeout: float = 1.0) -> None:  # noqa: ANN001
    application = _application()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.002)
    msg = "condition was not met before timeout"
    raise AssertionError(msg)


def test_fits_supported_aspects_inside_a_fixed_1080p_canvas() -> None:
    """Preserve room aspect ratios while delivering one stable OBS media type."""
    wide = fit_virtual_camera_frame(np.full((720, 1280, 3), 10, dtype=np.uint8))
    classic = fit_virtual_camera_frame(np.full((480, 640, 3), 20, dtype=np.uint8))
    square = fit_virtual_camera_frame(np.full((720, 720, 3), 30, dtype=np.uint8))

    assert wide.shape == (FULL_HD_PROFILE.height, FULL_HD_PROFILE.width, 3)
    assert wide.flags.c_contiguous
    assert np.all(wide == 10)
    assert np.all(classic[:, :240] == BACKGROUND_RGB)
    assert np.all(classic[:, 240:1680] == 20)
    assert np.all(classic[:, 1680:] == BACKGROUND_RGB)
    assert np.all(square[:, :420] == BACKGROUND_RGB)
    assert np.all(square[:, 420:1500] == 30)
    assert np.all(square[:, 1500:] == BACKGROUND_RGB)

    hd = fit_virtual_camera_frame(np.full((480, 640, 3), 40, dtype=np.uint8), HD_PROFILE)
    assert hd.shape == (HD_PROFILE.height, HD_PROFILE.width, 3)
    assert np.all(hd[:, :160] == BACKGROUND_RGB)
    assert np.all(hd[:, 160:1120] == 40)
    assert np.all(hd[:, 1120:] == BACKGROUND_RGB)


def test_rejects_invalid_virtual_camera_frames() -> None:
    """Keep corrupt arrays away from the fixed OBS publication contract."""
    invalid = np.zeros((1080, 1920), dtype=np.uint8)

    with pytest.raises(ValueError, match="uint8 RGB"):
        fit_virtual_camera_frame(invalid)


def test_obs_sink_uses_selected_exact_rgb_media_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the selected profile and pinned OBS backend contract to pyvirtualcam."""
    sink = RecordingSink()
    arguments: dict[str, object] = {}
    rgb_format = object()

    def create_camera(**keywords: object) -> RecordingSink:
        arguments.update(keywords)
        return sink

    module = SimpleNamespace(
        Camera=create_camera,
        PixelFormat=SimpleNamespace(RGB=rgb_format),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)

    assert create_obs_virtual_camera(HD_PROFILE) is sink
    assert arguments == {
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "fmt": rgb_format,
        "device": "OBS Virtual Camera",
        "backend": "obs",
    }


def test_controller_confirms_start_publishes_latest_frame_and_stops() -> None:
    """Publish on a worker and expose only confirmed active transitions."""
    _application()
    sink = RecordingSink()
    selected_profiles = []
    controller = VirtualCameraController(
        sink_factory=lambda profile: selected_profiles.append(profile) or sink,
    )
    active: list[bool] = []
    controller.active_changed.connect(active.append)

    controller.start()
    _wait_until(lambda: controller.state.phase == "active")
    controller.start()
    controller.publish_frame(np.full((720, 1280, 3), 77, dtype=np.uint8), 0.0)
    _wait_until(lambda: any(np.all(frame == 77) for frame in sink.frames))
    controller.stop()
    _wait_until(lambda: controller.state.phase == "inactive")
    controller.stop()

    assert active == [True, False]
    assert sink.closed
    assert selected_profiles == [FULL_HD_PROFILE]


def test_controller_uses_waiting_frame_when_latest_composite_is_stale() -> None:
    """Replace an old webcam image rather than freezing it indefinitely."""
    _application()
    now = [10.0]
    sink = RecordingSink()
    controller = VirtualCameraController(
        sink_factory=lambda _profile: sink,
        clock=lambda: now[0],
    )
    controller.publish_frame(np.full((1080, 1920, 3), 91, dtype=np.uint8), 0.0)
    controller.start()
    _wait_until(lambda: any(np.all(frame == 91) for frame in sink.frames))

    before = len(sink.frames)
    now[0] += 0.6
    _wait_until(
        lambda: (
            len(sink.frames) > before
            and not np.all(sink.frames[-1] == 91)
            and tuple(sink.frames[-1][0, 0]) == BACKGROUND_RGB
        ),
    )
    controller.shutdown()


def test_controller_reports_startup_and_midstream_failures() -> None:
    """Restore a retryable state when OBS cannot open or later disconnects."""
    _application()

    def unavailable(_profile: VirtualCameraProfile) -> Never:
        msg = "device already in use"
        raise RuntimeError(msg)

    startup = VirtualCameraController(sink_factory=unavailable)
    startup.start()
    _wait_until(lambda: startup.state.phase == "failed")
    assert startup.state.message == (
        "OBS Virtual Camera is already in use. Stop it in OBS or another application, then retry."
    )

    sink = RecordingSink(fail_after=1)
    streaming = VirtualCameraController(sink_factory=lambda _profile: sink)
    states: list[bool] = []
    streaming.active_changed.connect(states.append)
    streaming.start()
    _wait_until(lambda: streaming.state.phase == "failed")

    assert streaming.state.message == "Virtual camera stopped: backend disconnected"
    assert states == [True, False]
    assert sink.closed


def test_controller_locks_selected_profile_for_each_active_session() -> None:
    """Offer 720p and 1080p without changing media type under a consumer."""
    _application()
    selected_profiles = []
    sinks: list[RecordingSink] = []

    def create(profile: VirtualCameraProfile) -> RecordingSink:
        selected_profiles.append(profile)
        sink = RecordingSink()
        sinks.append(sink)
        return sink

    controller = VirtualCameraController(sink_factory=create)
    controller.select_profile("720p")
    controller.start()
    _wait_until(lambda: controller.state.phase == "active")
    controller.select_profile("1080p")
    assert controller.profile == HD_PROFILE
    controller.stop()
    _wait_until(lambda: controller.state.phase == "inactive")

    controller.select_profile("1080p")
    controller.start()
    _wait_until(lambda: controller.state.phase == "active")
    controller.shutdown()

    assert selected_profiles == [HD_PROFILE, FULL_HD_PROFILE]
