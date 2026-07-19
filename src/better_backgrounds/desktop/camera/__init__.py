"""Qt camera discovery, persistence, and native frame capture."""

from better_backgrounds.desktop.camera.capture import CaptureProfile, OutputGeometry
from better_backgrounds.desktop.camera.devices import (
    InputCamera,
    InputCameraSelectionStore,
    InputCameraSource,
)

__all__ = [
    "CaptureProfile",
    "InputCamera",
    "InputCameraSelectionStore",
    "InputCameraSource",
    "OutputGeometry",
]
