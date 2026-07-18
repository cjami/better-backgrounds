"""Validated scene geometry, resources, provenance, and camera state."""

from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from PySide6.QtCore import QUrl

SCENE_SCHEME = "bbscene"
APP_SCHEME = "bbapp"
SOURCE_SIZE_DIMENSIONS = 2


class StrictModel(BaseModel):
    """Reject unknown fields at persisted and downloaded trust boundaries."""

    model_config = ConfigDict(extra="forbid")


class Vector3(StrictModel):
    """Represent a finite point or direction in scene coordinates."""

    x: float = Field(default=0.0, allow_inf_nan=False)
    y: float = Field(default=0.0, allow_inf_nan=False)
    z: float = Field(default=0.0, allow_inf_nan=False)


class Quaternion(StrictModel):
    """Represent a normalized scene or camera orientation."""

    x: float = Field(default=0.0, allow_inf_nan=False)
    y: float = Field(default=0.0, allow_inf_nan=False)
    z: float = Field(default=0.0, allow_inf_nan=False)
    w: float = Field(default=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def normalized(self) -> Self:
        """Reject rotations that do not form a usable unit quaternion."""
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if not math.isclose(length, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            msg = "orientation must be a normalized quaternion"
            raise ValueError(msg)
        return self


class CropRegion(StrictModel):
    """Describe a normalized output crop."""

    left: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    top: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    right: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    bottom: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def has_area(self) -> Self:
        """Require a non-empty crop rectangle."""
        if self.left >= self.right or self.top >= self.bottom:
            msg = "crop must have positive area"
            raise ValueError(msg)
        return self


class SubjectRegion(StrictModel):
    """Describe the normalized area reserved for the webcam subject."""

    x: float = Field(default=0.35, ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(default=0.2, ge=0.0, le=1.0, allow_inf_nan=False)
    width: float = Field(default=0.3, gt=0.0, le=1.0, allow_inf_nan=False)
    height: float = Field(default=0.7, gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def within_output(self) -> Self:
        """Keep the placement guide inside the rendered frame."""
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            msg = "subject region must remain inside the output"
            raise ValueError(msg)
        return self


class CameraBounds(StrictModel):
    """Bound camera movement to poses supported by the captured scene."""

    minimum: Vector3 = Field(default_factory=lambda: Vector3(x=-1.0, y=0.6, z=-2.5))
    maximum: Vector3 = Field(default_factory=lambda: Vector3(x=1.0, y=2.2, z=0.5))

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Require an ordered three-dimensional region."""
        if not (
            self.minimum.x < self.maximum.x
            and self.minimum.y < self.maximum.y
            and self.minimum.z < self.maximum.z
        ):
            msg = "camera bounds must have positive volume"
            raise ValueError(msg)
        return self

    def contains(self, position: Vector3) -> bool:
        """Return whether a position stays inside the supported capture region."""
        return (
            self.minimum.x <= position.x <= self.maximum.x
            and self.minimum.y <= position.y <= self.maximum.y
            and self.minimum.z <= position.z <= self.maximum.z
        )


class SceneTransform(StrictModel):
    """Keep imported-scene normalization separate from user camera movement."""

    translation: Vector3 = Field(default_factory=Vector3)
    orientation: Quaternion = Field(default_factory=Quaternion)
    scale: float = Field(default=1.0, ge=0.01, le=100.0, allow_inf_nan=False)


def colmap_scene_transform() -> SceneTransform:
    """Convert COLMAP/Brush splat coordinates to the PlayCanvas scene basis."""
    return SceneTransform(orientation=Quaternion(z=1.0, w=0.0))


def sharp_scene_transform() -> SceneTransform:
    """Convert SHARP's OpenCV basis to the PlayCanvas scene basis."""
    return SceneTransform(orientation=Quaternion(x=1.0, w=0.0))


class Viewpoint(StrictModel):
    """Persist every safe virtual-camera and subject-placement input."""

    schema_version: Literal[1] = 1
    position: Vector3 = Field(default_factory=lambda: Vector3(x=0.0, y=1.4, z=-2.1))
    orientation: Quaternion = Field(default_factory=Quaternion)
    orbit_target: Vector3 = Field(default_factory=lambda: Vector3(x=0.0, y=1.1, z=0.0))
    field_of_view: float = Field(default=42.0, ge=24.0, le=90.0, allow_inf_nan=False)
    horizon: float = Field(default=-1.5, ge=-10.0, le=10.0, allow_inf_nan=False)
    near_clip: float = Field(default=0.05, gt=0.0, le=10.0, allow_inf_nan=False)
    far_clip: float = Field(default=40.0, gt=0.0, le=1_000.0, allow_inf_nan=False)
    aspect_ratio: float = Field(default=16 / 9, ge=0.5, le=4.0, allow_inf_nan=False)
    crop: CropRegion = Field(default_factory=CropRegion)
    subject_region: SubjectRegion = Field(default_factory=SubjectRegion)
    scene_transform: SceneTransform = Field(default_factory=SceneTransform)
    safe_camera_region: CameraBounds = Field(default_factory=CameraBounds)
    nearest_source_frame: str | None = Field(default=None, max_length=160)
    subject_depth: float = Field(default=2.4, ge=0.5, le=5.0, allow_inf_nan=False)
    focus_depth: float = Field(default=2.6, ge=0.5, le=5.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_clipping_range(self) -> Self:
        """Require the far plane to remain beyond the near plane."""
        if self.far_clip <= self.near_clip:
            msg = "far clip must be greater than near clip"
            raise ValueError(msg)
        return self

    @property
    def is_inside_safe_region(self) -> bool:
        """Report whether the camera is supported by nearby captured poses."""
        return self.safe_camera_region.contains(self.position)

    @property
    def camera_fingerprint(self) -> str:
        """Return a stable fingerprint for renderer buffers and persistence."""
        payload = self.model_dump_json(round_trip=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class AssetResource(StrictModel):
    """Identify one checksummed file belonging to a managed scene."""

    path: str = Field(min_length=1, max_length=200)
    url: HttpUrl | None = None
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        """Reject absolute, ambiguous, and traversing resource names."""
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or "\\" in value
            or ":" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            msg = "resource path must be a normalized relative path"
            raise ValueError(msg)
        return path.as_posix()


class SceneProvenance(StrictModel):
    """Record reproducible local-scene build evidence without retaining input pixels."""

    source_kind: Literal["upload", "camera"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size: tuple[int, int]
    builder_name: str = Field(default="Apple SHARP", min_length=1, max_length=80)
    builder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: Literal["cuda", "mps", "cpu"]
    inference_ms: float = Field(ge=0.0, allow_inf_nan=False)
    license_name: str = Field(min_length=1, max_length=100)
    license_url: HttpUrl | None = None

    @field_validator("source_size")
    @classmethod
    def positive_source_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Require a positive width and height."""
        if len(value) != SOURCE_SIZE_DIMENSIONS or any(dimension <= 0 for dimension in value):
            msg = "source size must contain a positive width and height"
            raise ValueError(msg)
        return value


class SceneReference(StrictModel):
    """Describe an application-owned scene without exposing a local path."""

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    display_name: str = Field(min_length=1, max_length=100)
    format: Literal["sog", "ply"]
    entrypoint: str = Field(min_length=1, max_length=200)
    resources: tuple[AssetResource, ...] = Field(min_length=1)
    license_name: str = Field(min_length=1, max_length=80)
    license_url: HttpUrl | None = None
    attribution: str = Field(min_length=1, max_length=300)
    attribution_url: HttpUrl | None = None
    preview: str | None = Field(default=None, max_length=200)
    default_viewpoint: Viewpoint = Field(default_factory=Viewpoint)
    provenance: SceneProvenance | None = None

    @model_validator(mode="after")
    def valid_entrypoint(self) -> Self:
        """Require unique resource paths and a manifest-owned entrypoint."""
        paths = [resource.path for resource in self.resources]
        if len(paths) != len(set(paths)):
            msg = "scene resource paths must be unique"
            raise ValueError(msg)
        if self.entrypoint not in paths:
            msg = "scene entrypoint must name a managed resource"
            raise ValueError(msg)
        if self.preview is not None and self.preview not in paths:
            msg = "scene preview must name a managed resource"
            raise ValueError(msg)
        return self

    @property
    def expected_size(self) -> int:
        """Return the complete download size in bytes."""
        return sum(resource.size for resource in self.resources)

    @property
    def managed_url(self) -> QUrl:
        """Return the renderer-safe URL of the runtime entrypoint."""
        return QUrl(f"{SCENE_SCHEME}://{self.asset_id}/{self.entrypoint}")


def normalize_colmap_scene_reference(reference: SceneReference) -> SceneReference:
    """Attach the non-destructive renderer transform for a COLMAP-derived scene."""
    viewpoint = reference.default_viewpoint.model_copy(
        update={"scene_transform": colmap_scene_transform()},
    )
    return reference.model_copy(update={"default_viewpoint": viewpoint})
