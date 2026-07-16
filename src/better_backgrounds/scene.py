"""Managed scene assets and room-scoped virtual camera state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterable
from contextlib import closing
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, Self
from urllib.request import urlopen
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from PySide6.QtCore import QUrl

SCENE_SCHEME = "bbscene"
APP_SCHEME = "bbapp"
CHUNK_SIZE = 64 * 1024


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
    url: HttpUrl
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


class SceneReference(StrictModel):
    """Describe an application-owned scene without exposing a local path."""

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    display_name: str = Field(min_length=1, max_length=100)
    format: Literal["sog"]
    entrypoint: str = Field(min_length=1, max_length=200)
    resources: tuple[AssetResource, ...] = Field(min_length=1)
    license_name: str = Field(min_length=1, max_length=80)
    license_url: HttpUrl
    attribution: str = Field(min_length=1, max_length=300)
    attribution_url: HttpUrl
    preview: str | None = Field(default=None, max_length=200)
    default_viewpoint: Viewpoint = Field(default_factory=Viewpoint)

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


class SceneAssetManifest(StrictModel):
    """Validate the checked-in versioned sample catalogue."""

    schema_version: Literal[1] = 1
    scenes: tuple[SceneReference, ...]


def load_sample_manifest() -> SceneAssetManifest:
    """Load and validate the versioned sample manifest from package data."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/sample-scenes-v1.json")
        .read_text(encoding="utf-8")
    )
    return SceneAssetManifest.model_validate_json(content)


ProgressCallback = Callable[[int, int], None]
ResourceOpener = Callable[[str], BinaryIO]


def _urlopen(url: str) -> BinaryIO:
    return urlopen(url, timeout=30)  # noqa: S310


class AssetInstaller:
    """Download scenes into a verified cache with atomic publication."""

    def __init__(self, root: Path, *, opener: ResourceOpener = _urlopen) -> None:
        """Use one application-owned root and an injectable network boundary."""
        self.root = root
        self._opener = opener
        self._verified: dict[str, str] = {}

    def install(
        self,
        reference: SceneReference,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Return a complete cached scene, downloading and verifying if needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / reference.asset_id
        if self.is_ready(reference):
            return target
        if target.exists():
            self._verified.pop(reference.asset_id, None)
            shutil.rmtree(target)

        staging = self.root / f".{reference.asset_id}.{uuid4().hex}.part"
        completed = 0
        try:
            staging.mkdir()
            for resource in reference.resources:
                destination = staging.joinpath(*PurePosixPath(resource.path).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                completed += self._download(resource, destination, completed, reference, progress)
            marker = staging / ".complete.json"
            marker.write_text(
                json.dumps({"schema_version": 1, "manifest": self._digest(reference)}),
                encoding="utf-8",
            )
            staging.replace(target)
            self._verified[reference.asset_id] = self._digest(reference)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    def _download(
        self,
        resource: AssetResource,
        destination: Path,
        already_completed: int,
        reference: SceneReference,
        progress: ProgressCallback | None,
    ) -> int:
        digest = hashlib.sha256()
        written = 0
        with closing(self._opener(str(resource.url))) as source, destination.open("xb") as output:
            while chunk := source.read(CHUNK_SIZE):
                written += len(chunk)
                if written > resource.size:
                    msg = f"asset integrity failure for {resource.path}"
                    raise ValueError(msg)
                digest.update(chunk)
                output.write(chunk)
                if progress is not None:
                    progress(already_completed + written, reference.expected_size)
            output.flush()
            os.fsync(output.fileno())
        if written != resource.size or digest.hexdigest() != resource.sha256:
            msg = f"asset integrity failure for {resource.path}"
            raise ValueError(msg)
        return written

    def is_ready(self, reference: SceneReference) -> bool:
        """Return whether every cached resource matches the checked-in manifest."""
        target = self.root / reference.asset_id
        marker = target / ".complete.json"
        expected_digest = self._digest(reference)
        if self._verified.get(reference.asset_id) == expected_digest and target.is_dir():
            return True
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            return False
        if marker_value != {"schema_version": 1, "manifest": expected_digest}:
            return False
        for resource in reference.resources:
            path = target.joinpath(*PurePosixPath(resource.path).parts)
            if not path.is_file() or path.stat().st_size != resource.size:
                return False
            if self._file_digest(path) != resource.sha256:
                return False
        self._verified[reference.asset_id] = expected_digest
        return True

    def resource_path(self, reference: SceneReference, resource_path: str) -> Path | None:
        """Return a verified Python-owned resource path for native UI use."""
        if not self.is_ready(reference):
            return None
        if resource_path not in {resource.path for resource in reference.resources}:
            return None
        return self.root.joinpath(reference.asset_id, *PurePosixPath(resource_path).parts)

    @staticmethod
    def _digest(reference: SceneReference) -> str:
        value = reference.model_dump_json(round_trip=True)
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()


class ManagedSceneResolver:
    """Resolve controlled scene URLs to verified manifest-owned cache files."""

    def __init__(self, installer: AssetInstaller, references: Iterable[SceneReference]) -> None:
        """Index only validated application-owned scene identifiers."""
        self._installer = installer
        self._references = {reference.asset_id: reference for reference in references}

    def resolve(self, url: QUrl) -> Path | None:
        """Return a managed file or reject the entire untrusted URL."""
        if url.scheme() != SCENE_SCHEME or url.hasQuery() or url.hasFragment():
            return None
        reference = self._references.get(url.host())
        if reference is None or not self._installer.is_ready(reference):
            return None
        raw_path = url.path().lstrip("/")
        path = PurePosixPath(raw_path)
        if path.as_posix() not in {resource.path for resource in reference.resources}:
            return None
        candidate = self._installer.root.joinpath(reference.asset_id, *path.parts).resolve()
        root = (self._installer.root / reference.asset_id).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate


class ViewpointDocument(StrictModel):
    """Version all persisted room viewpoints together."""

    schema_version: Literal[1] = 1
    rooms: dict[str, Viewpoint] = Field(default_factory=dict)


class ViewpointStore:
    """Persist room-scoped viewpoints through an atomic JSON document."""

    def __init__(self, path: Path) -> None:
        """Store viewpoints at an application-data path owned by Python."""
        self.path = path

    def load(self, room_id: str) -> Viewpoint | None:
        """Return the saved viewpoint for one room, if the document is valid."""
        return self._read().rooms.get(room_id)

    def save(self, room_id: str, viewpoint: Viewpoint) -> None:
        """Atomically replace one room's saved viewpoint."""
        document = self._read()
        document.rooms[room_id] = viewpoint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self) -> ViewpointDocument:
        try:
            return ViewpointDocument.model_validate_json(self.path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return ViewpointDocument()
