"""Qt camera discovery, persistence, and native frame capture."""

from better_backgrounds.desktop.camera.capture import CaptureProfile, OutputGeometry
from better_backgrounds.desktop.camera.devices import (
    InputCamera,
    InputCameraSelectionStore,
    InputCameraSource,
)
from better_backgrounds.desktop.camera.room_capture import RoomCaptureController

__all__ = [
    "CaptureProfile",
    "InputCamera",
    "InputCameraSelectionStore",
    "InputCameraSource",
    "OutputGeometry",
    "RoomCaptureController",
]
