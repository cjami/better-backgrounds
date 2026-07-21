"""Persist rendered room backgrounds as verified scene-derived assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from better_backgrounds.scene.models import StrictModel

if TYPE_CHECKING:
    from better_backgrounds.scene.models import SceneReference, Viewpoint

SNAPSHOT_RENDERER_SCHEMA = 1
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_DIMENSION = 8_192
BACKGROUND_FILE = "background.png"
MANIFEST_FILE = "manifest.json"


class SnapshotManifest(StrictModel):
    """Describe one complete current-view renderer output."""

    schema_version: Literal[1] = 1
    renderer_schema: Literal[1] = SNAPSHOT_RENDERER_SCHEMA
    asset_id: str
    scene_fingerprint: str
    viewpoint_fingerprint: str
    width: int
    height: int
    background_size: int
    background_sha256: str


@dataclass(frozen=True, slots=True)
class SnapshotPaths:
    """Expose the verified image consumed by the live compositor."""

    background: Path


class SnapshotStore:
    """Publish and resolve lossless snapshots without loading scene geometry."""

    def __init__(self, root: Path) -> None:
        """Store derived renderer assets below the application cache."""
        self._root = root

    def load(self, scene: SceneReference, viewpoint: Viewpoint) -> SnapshotPaths | None:
        """Return a matching frame, ignoring stale or corrupt cache data."""
        scene_fingerprint = _scene_fingerprint(scene)
        directory = self._directory(scene.asset_id, scene_fingerprint, viewpoint)
        try:
            manifest = SnapshotManifest.model_validate_json(
                directory.joinpath(MANIFEST_FILE).read_text(encoding="utf-8"),
            )
        except OSError, ValidationError:
            return None
        identity = (
            manifest.asset_id,
            manifest.scene_fingerprint,
            manifest.viewpoint_fingerprint,
        )
        expected_identity = (
            scene.asset_id,
            scene_fingerprint,
            viewpoint.camera_fingerprint,
        )
        if identity != expected_identity:
            return None
        background = directory / BACKGROUND_FILE
        if not _verified_file(
            background,
            manifest.background_size,
            manifest.background_sha256,
        ):
            return None
        return SnapshotPaths(background=background)

    def save(
        self,
        scene: SceneReference,
        viewpoint: Viewpoint,
        background: bytes,
    ) -> SnapshotPaths:
        """Atomically publish a validated current-view background."""
        width, height = _png_dimensions(background)
        scene_fingerprint = _scene_fingerprint(scene)
        directory = self._directory(scene.asset_id, scene_fingerprint, viewpoint)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = SnapshotManifest(
            asset_id=scene.asset_id,
            scene_fingerprint=scene_fingerprint,
            viewpoint_fingerprint=viewpoint.camera_fingerprint,
            width=width,
            height=height,
            background_size=len(background),
            background_sha256=_digest(background),
        )
        background_path = directory / BACKGROUND_FILE
        _atomic_write(background_path, background)
        _atomic_write(
            directory / MANIFEST_FILE,
            manifest.model_dump_json(indent=2).encode(),
        )
        return SnapshotPaths(background=background_path)

    def delete(self, asset_id: str) -> None:
        """Remove every cached snapshot for one scene when its room is deleted."""
        directory = self._root / asset_id
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)

    def _directory(
        self,
        asset_id: str,
        scene_fingerprint: str,
        viewpoint: Viewpoint,
    ) -> Path:
        identity = json.dumps(
            {
                "renderer_schema": SNAPSHOT_RENDERER_SCHEMA,
                "scene": scene_fingerprint,
                "viewpoint": viewpoint.camera_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self._root / asset_id / _digest(identity)


def _scene_fingerprint(scene: SceneReference) -> str:
    identity = {
        "asset_id": scene.asset_id,
        "entrypoint": scene.entrypoint,
        "format": scene.format,
        "metric_depth": scene.supports_metric_depth,
        "resources": [
            {"path": resource.path, "size": resource.size, "sha256": resource.sha256}
            for resource in scene.resources
        ],
    }
    return _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if not 1 <= len(payload) <= MAX_SNAPSHOT_BYTES:
        msg = "snapshot payload size is outside the supported range"
        raise ValueError(msg)
    try:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            if (
                image.format != "PNG"
                or not 1 <= width <= MAX_SNAPSHOT_DIMENSION
                or not 1 <= height <= MAX_SNAPSHOT_DIMENSION
            ):
                msg = "snapshot must be a PNG at supported framebuffer dimensions"
                raise ValueError(msg)
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        msg = "snapshot payload is not a valid PNG"
        raise ValueError(msg) from error
    return width, height


def _verified_file(path: Path, expected_size: int, expected_digest: str) -> bool:
    try:
        payload = path.read_bytes()
    except OSError:
        return False
    return len(payload) == expected_size and _digest(payload) == expected_digest


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
