"""Python-owned input-camera discovery and selection persistence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtMultimedia import QMediaDevices

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class InputCamera:
    """Describe one selectable video input without retaining native handles."""

    device_id: str
    description: str
    is_default: bool = False


CameraProvider = Callable[[], Sequence[InputCamera]]


class InputCameraSource(QObject):
    """Expose current Qt video inputs through a small injectable boundary."""

    cameras_changed = Signal()

    def __init__(
        self,
        provider: CameraProvider | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Use system discovery by default and permit deterministic test providers."""
        super().__init__(parent)
        self._media_devices: QMediaDevices | None = None
        if provider is None:
            self._media_devices = QMediaDevices(self)
            self._media_devices.videoInputsChanged.connect(self.refresh)
            self._provider = self._system_cameras
        else:
            self._provider = provider

    def cameras(self) -> tuple[InputCamera, ...]:
        """Return an immutable snapshot of the currently available inputs."""
        return tuple(self._provider())

    @Slot()
    def refresh(self) -> None:
        """Notify consumers after a device hot-plug event."""
        self.cameras_changed.emit()

    @staticmethod
    def _system_cameras() -> tuple[InputCamera, ...]:
        cameras = []
        for device in QMediaDevices.videoInputs():
            identifier = bytes(device.id().toHex().data()).decode("ascii")
            cameras.append(
                InputCamera(
                    device_id=identifier,
                    description=device.description().strip() or "Camera",
                    is_default=device.isDefault(),
                ),
            )
        return tuple(cameras)


class InputCameraSelection(BaseModel):
    """Validate the versioned camera preference document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    device_id: str = Field(min_length=1, max_length=4_096)


class InputCameraSelectionStore:
    """Persist the preferred input camera as an atomic application-data file."""

    def __init__(self, path: Path) -> None:
        """Use the provided Python-owned application-data path."""
        self.path = path

    def load(self) -> str | None:
        """Return a valid preferred device identifier when one has been saved."""
        try:
            document = InputCameraSelection.model_validate_json(
                self.path.read_text(encoding="utf-8"),
            )
        except OSError, ValidationError:
            return None
        return document.device_id

    def save(self, device_id: str) -> None:
        """Atomically replace the preferred device identifier."""
        document = InputCameraSelection(device_id=device_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
