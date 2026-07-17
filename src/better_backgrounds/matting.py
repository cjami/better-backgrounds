"""Versioned local settings for webcam mask refinement and presentation."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator

if TYPE_CHECKING:
    from pathlib import Path


class MattingSettings(BaseModel):
    """Bound confidence shaping controls sent to the local worker."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.55, ge=0.05, le=0.95, allow_inf_nan=False)
    temporal: float = Field(default=0.65, ge=0.0, le=0.95, allow_inf_nan=False)
    feather: float = Field(default=0.06, ge=0.01, le=0.5, allow_inf_nan=False)
    edge_radius: int = Field(default=0, ge=0, le=4)

    def worker_payload(self) -> str:
        """Serialize with the JavaScript worker's camel-case edge key."""
        return self.model_dump_json().replace('"edge_radius"', '"edgeRadius"')


class PackagedMattingAsset(BaseModel):
    """Identify one immutable runtime file by safe path and digest."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=200)
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        """Reject absolute and traversing package paths."""
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or any(part in {".", ".."} for part in path.parts)
        ):
            msg = "matting asset path must be normalized and relative"
            raise ValueError(msg)
        return value


class MattingRuntimeMetadata(BaseModel):
    """Record the exact locally bundled MediaPipe runtime."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    source: HttpUrl
    license: Literal["Apache-2.0"]
    assets: tuple[PackagedMattingAsset, ...]


class MattingModelMetadata(BaseModel):
    """Record the locally bundled person-segmentation model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    path: Literal["selfie_segmenter_landscape.tflite"]
    source: HttpUrl
    model_card: HttpUrl
    license: Literal["Apache-2.0"]
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MattingAssetManifest(BaseModel):
    """Validate source, license, version, size, and checksum metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    runtime: MattingRuntimeMetadata
    model: MattingModelMetadata


def load_matting_asset_manifest() -> MattingAssetManifest:
    """Load the versioned package-data manifest."""
    content = (
        files("better_backgrounds.desktop")
        .joinpath("assets/matting/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    return MattingAssetManifest.model_validate_json(content)


def verify_packaged_matting_assets() -> MattingAssetManifest:
    """Refuse altered or incomplete offline runtime and model assets."""
    manifest = load_matting_asset_manifest()
    root = files("better_backgrounds.desktop").joinpath("assets/matting")
    expected = [
        *manifest.runtime.assets,
        PackagedMattingAsset(
            path=manifest.model.path,
            size=manifest.model.size,
            sha256=manifest.model.sha256,
        ),
    ]
    for asset in expected:
        content = root.joinpath(asset.path).read_bytes()
        if len(content) != asset.size or hashlib.sha256(content).hexdigest() != asset.sha256:
            msg = f"Packaged matting asset failed verification: {asset.path}"
            raise ValueError(msg)
    return manifest


class LivePreferences(BaseModel):
    """Persist user-owned foreground presentation settings."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    mirrored: bool = True
    matting: MattingSettings = Field(default_factory=MattingSettings)


class LivePreferencesStore:
    """Load and atomically save local camera-presentation preferences."""

    def __init__(self, path: Path) -> None:
        """Use one application-data document owned by Python."""
        self.path = path

    def load(self) -> LivePreferences:
        """Return defaults when persisted settings are absent or invalid."""
        try:
            return LivePreferences.model_validate_json(self.path.read_text(encoding="utf-8"))
        except OSError, ValidationError:
            return LivePreferences()

    def save(self, preferences: LivePreferences) -> None:
        """Atomically replace the complete live preference document."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(preferences.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
