"""Qt camera discovery, persistence, and native frame capture."""

from better_backgrounds.desktop.camera.capture import CaptureProfile, OutputGeometry
from better_backgrounds.desktop.camera.devices import (
    DEFAULT_INPUT_RESOLUTION,
    InputCamera,
    InputCameraSelectionStore,
    InputCameraSource,
    InputResolution,
    InputResolutionStore,
)
from better_backgrounds.desktop.camera.room_capture import RoomCaptureController

__all__ = [
    "DEFAULT_INPUT_RESOLUTION",
    "CaptureProfile",
    "InputCamera",
    "InputCameraSelectionStore",
    "InputCameraSource",
    "InputResolution",
    "InputResolutionStore",
    "OutputGeometry",
    "RoomCaptureController",
]
