"""Versioned presentation preferences and the one-shot seed model asset."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


class MattingSettings(BaseModel):
    """Accept legacy refinement preferences during the MatAnyone 2 migration."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.55, ge=0.05, le=0.95, allow_inf_nan=False)
    temporal: float = Field(default=0.65, ge=0.0, le=0.95, allow_inf_nan=False)
    feather: float = Field(default=0.06, ge=0.01, le=0.5, allow_inf_nan=False)
    edge_radius: int = Field(default=0, ge=0, le=4)


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
    """Validate the one-shot MediaPipe seed model metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
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
    """Refuse an altered or incomplete one-shot MediaPipe seed model."""
    manifest = load_matting_asset_manifest()
    model = files("better_backgrounds.desktop").joinpath(
        "assets",
        "matting",
        manifest.model.path,
    )
    content = model.read_bytes()
    if (
        len(content) != manifest.model.size
        or hashlib.sha256(content).hexdigest() != manifest.model.sha256
    ):
        msg = f"Packaged seed model failed verification: {manifest.model.path}"
        raise ValueError(msg)
    return manifest


def packaged_seed_model_path() -> Path:
    """Return the verified MediaPipe model used only for first-frame seeding."""
    manifest = verify_packaged_matting_assets()
    resource = files("better_backgrounds.desktop").joinpath(
        "assets",
        "matting",
        manifest.model.path,
    )
    return Path(str(resource))


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
