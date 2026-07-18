"""Typed SHARP build, checkpoint, and output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import Field, HttpUrl

from better_backgrounds.scene.models import StrictModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from better_backgrounds.reconstruction.images import SceneImageSelection
    from better_backgrounds.reconstruction.sharp.runtime import SharpDevice, SharpDeviceRequest


class SharpCancelledError(RuntimeError):
    """Raised when a SHARP preparation or build cooperatively stops."""


@dataclass(frozen=True, slots=True)
class SharpBuildConfig:
    """Configure one isolated SHARP build."""

    device: SharpDeviceRequest
    checkpoint_path: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class SharpCapabilities:
    """Describe the selected official prediction device."""

    device_type: SharpDevice
    accelerated: bool


@dataclass(frozen=True, slots=True)
class SceneBuildRequest:
    """Bind one image selection to a stable worker job."""

    job_id: str
    selection: SceneImageSelection
    config: SharpBuildConfig


@dataclass(frozen=True, slots=True)
class SharpPlyMetadata:
    """Return validated SHARP camera metadata needed by the renderer."""

    gaussian_count: int
    image_size: tuple[int, int]
    focal_length_px: float
    field_of_view: float


class SharpCheckpointManifest(StrictModel):
    """Pin the official research checkpoint and its accepted license identity."""

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1, max_length=80)
    filename: str = Field(pattern=r"^[a-zA-Z0-9_.-]+\.pt$")
    url: HttpUrl
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    builder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_name: str = Field(min_length=1, max_length=100)
    license_url: HttpUrl


class SharpPredictor(Protocol):
    """Allow official inference to be replaced only at test boundaries."""

    def __call__(
        self,
        image_path: Path,
        focal_length_px: float,
        checkpoint_path: Path,
        device: SharpDevice,
        output_path: Path,
        model_loaded: Callable[[], None],
    ) -> float:
        """Write one SHARP PLY and return accelerator-synchronized inference time."""
        ...
