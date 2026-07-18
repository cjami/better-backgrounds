"""Pinned SHARP checkpoint discovery, validation, and installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from contextlib import closing
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import TYPE_CHECKING, BinaryIO
from urllib.request import urlopen
from uuid import uuid4

from better_backgrounds.reconstruction.images import sha256_file
from better_backgrounds.reconstruction.sharp.contracts import (
    SharpCancelledError,
    SharpCapabilities,
    SharpCheckpointManifest,
)
from better_backgrounds.reconstruction.sharp.runtime import (
    SharpDeviceRequest,
    ensure_vendored_sharp,
    resolve_sharp_device,
)

if TYPE_CHECKING:
    from pathlib import Path

SHARP_BUILDER_REVISION = "1eaa046834b81852261262b41b0919f5c1efdd2e"
SHARP_DEPENDENCY_VERSIONS = {
    "plyfile": "1.1.2",
    "scipy": "1.16.2",
    "timm": "1.0.20",
}
FILE_CHUNK_SIZE = 4 * 1024 * 1024
CheckpointProgress = Callable[[int, int], None]
CancellationCheck = Callable[[], bool]
ResourceOpener = Callable[[str], BinaryIO]


def load_sharp_checkpoint_manifest() -> SharpCheckpointManifest:
    """Load the exact checked-in SHARP checkpoint identity."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/sharp/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = SharpCheckpointManifest.model_validate_json(content)
    if manifest.builder_revision != SHARP_BUILDER_REVISION:
        msg = "The SHARP checkpoint manifest does not match the vendored source revision"
        raise ValueError(msg)
    return manifest


def probe_sharp_capabilities(
    requested: SharpDeviceRequest = "auto",
) -> SharpCapabilities:
    """Verify the pinned inference imports and resolve one usable device."""
    for package, expected in SHARP_DEPENDENCY_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            msg = f"The pinned SHARP dependency {package} is unavailable"
            raise RuntimeError(msg) from error
        if actual != expected:
            msg = f"SHARP requires {package}=={expected}, found {actual}"
            raise RuntimeError(msg)
    ensure_vendored_sharp()
    device = resolve_sharp_device(requested)
    return SharpCapabilities(device_type=device, accelerated=device != "cpu")


def _urlopen(url: str) -> BinaryIO:
    return urlopen(url, timeout=30)  # noqa: S310


class SharpCheckpointInstaller:
    """Download and atomically publish the license-gated official checkpoint."""

    def __init__(
        self,
        root: Path,
        *,
        manifest: SharpCheckpointManifest | None = None,
        opener: ResourceOpener = _urlopen,
    ) -> None:
        """Use a dedicated model cache and injectable download boundary."""
        self.root = root
        self.manifest = manifest or load_sharp_checkpoint_manifest()
        self._opener = opener

    @property
    def checkpoint_path(self) -> Path:
        """Return the pinned managed checkpoint path."""
        return self.root / self.manifest.filename

    @property
    def marker_path(self) -> Path:
        """Return the integrity marker beside the managed checkpoint."""
        return self.root / ".complete.json"

    def is_ready(self) -> bool:
        """Check cached size, identity, and modification evidence without rehashing 2.8 GB."""
        try:
            stat = self.checkpoint_path.stat()
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                return False
            return bool(
                stat.st_size == self.manifest.size
                and marker.get("sha256") == self.manifest.sha256
                and marker.get("size") == self.manifest.size
                and marker.get("mtime_ns") == stat.st_mtime_ns
            )
        except OSError, TypeError, ValueError:
            return False

    def prepare(
        self,
        *,
        license_accepted: bool,
        progress: CheckpointProgress | None = None,
        is_cancelled: CancellationCheck = lambda: False,
    ) -> Path:
        """Download, checksum, and publish only after explicit license acceptance."""
        if not license_accepted:
            msg = "The SHARP research-model license must be accepted before download"
            raise PermissionError(msg)
        if self.is_ready():
            return self.checkpoint_path
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.manifest.size + 512 * 1024 * 1024:
            msg = "There is not enough free disk space for the SHARP checkpoint"
            raise OSError(msg)
        temporary = self.root / f".{self.manifest.filename}.{uuid4().hex}.part"
        digest = hashlib.sha256()
        completed = 0
        try:
            with (
                closing(self._opener(str(self.manifest.url))) as response,
                temporary.open("wb") as destination,
            ):
                while chunk := response.read(FILE_CHUNK_SIZE):
                    if is_cancelled():
                        msg = "SHARP checkpoint preparation was cancelled"
                        raise SharpCancelledError(msg)
                    destination.write(chunk)
                    digest.update(chunk)
                    completed += len(chunk)
                    if completed > self.manifest.size:
                        msg = "The SHARP checkpoint exceeded its pinned size"
                        raise ValueError(msg)
                    if progress is not None:
                        progress(completed, self.manifest.size)
                destination.flush()
                os.fsync(destination.fileno())
            if completed != self.manifest.size or digest.hexdigest() != self.manifest.sha256:
                msg = "The SHARP checkpoint failed SHA-256 validation"
                raise ValueError(msg)
            temporary.replace(self.checkpoint_path)
            self._write_marker()
        finally:
            temporary.unlink(missing_ok=True)
        return self.checkpoint_path

    def validate(self, path: Path) -> None:
        """Require the official checkpoint identity before loading pickle-backed weights."""
        if path.resolve() == self.checkpoint_path.resolve() and self.is_ready():
            return
        if not path.is_file() or path.stat().st_size != self.manifest.size:
            msg = "The pinned SHARP checkpoint is missing or has the wrong size"
            raise ValueError(msg)
        if sha256_file(path) != self.manifest.sha256:
            msg = "The SHARP checkpoint failed SHA-256 validation"
            raise ValueError(msg)

    def _write_marker(self) -> None:
        stat = self.checkpoint_path.stat()
        payload = {
            "schema_version": 1,
            "sha256": self.manifest.sha256,
            "size": self.manifest.size,
            "mtime_ns": stat.st_mtime_ns,
        }
        temporary = self.marker_path.with_name(f".{self.marker_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.marker_path)
        finally:
            temporary.unlink(missing_ok=True)
