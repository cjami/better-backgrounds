"""Shared managed-checkpoint download, verification, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO
from urllib.request import urlopen
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path

FILE_CHUNK_SIZE = 4 * 1024 * 1024
FREE_SPACE_HEADROOM = 512 * 1024 * 1024
CheckpointProgress = Callable[[int, int], None]
CancellationCheck = Callable[[], bool]
ResourceOpener = Callable[[str], BinaryIO]


def sha256_path(path: Path) -> str:
    """Hash a file in bounded chunks without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(FILE_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def open_managed_url(url: str) -> BinaryIO:
    """Open one pinned checkpoint URL behind an injectable download boundary."""
    return urlopen(url, timeout=30)  # noqa: S310


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Pin the exact bytes one managed checkpoint is allowed to have."""

    filename: str
    url: str
    size: int
    sha256: str


class ManagedCheckpointInstaller:
    """Stream, verify, and atomically publish one size- and SHA-256-pinned checkpoint."""

    cancelled_error: type[Exception] = RuntimeError
    label: str = "checkpoint"
    license_name: str = ""
    requires_license_acceptance: bool = False

    def __init__(
        self,
        root: Path,
        identity: CheckpointIdentity,
        *,
        opener: ResourceOpener = open_managed_url,
    ) -> None:
        """Use a dedicated model cache and an injectable download boundary."""
        self.root = root
        self.identity = identity
        self._opener = opener

    @property
    def checkpoint_path(self) -> Path:
        """Return the pinned managed checkpoint path."""
        return self.root / self.identity.filename

    @property
    def marker_path(self) -> Path:
        """Return the integrity marker beside the managed checkpoint."""
        return self.root / ".complete.json"

    def is_ready(self) -> bool:
        """Check cached size, identity, and modification evidence without rehashing."""
        try:
            stat = self.checkpoint_path.stat()
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                return False
            return bool(
                stat.st_size == self.identity.size
                and marker.get("sha256") == self.identity.sha256
                and marker.get("size") == self.identity.size
                and marker.get("mtime_ns") == stat.st_mtime_ns
            )
        except OSError, TypeError, ValueError:
            return False

    def prepare(
        self,
        *,
        license_accepted: bool = False,
        progress: CheckpointProgress | None = None,
        is_cancelled: CancellationCheck = lambda: False,
    ) -> Path:
        """Download, checksum, and publish the pinned checkpoint exactly once.

        Models whose terms require per-user consent set
        ``requires_license_acceptance`` and refuse to download without it.
        """
        if self.is_ready():
            return self.checkpoint_path
        if self.requires_license_acceptance and not license_accepted:
            msg = f"The {self.label} model license must be accepted before download"
            raise PermissionError(msg)
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.identity.size + FREE_SPACE_HEADROOM:
            msg = f"There is not enough free disk space for the {self.label} checkpoint"
            raise OSError(msg)
        temporary = self.root / f".{self.identity.filename}.{uuid4().hex}.part"
        digest = hashlib.sha256()
        completed = 0
        try:
            with (
                closing(self._opener(str(self.identity.url))) as response,
                temporary.open("wb") as destination,
            ):
                while chunk := response.read(FILE_CHUNK_SIZE):
                    if is_cancelled():
                        msg = f"{self.label} checkpoint preparation was cancelled"
                        raise self.cancelled_error(msg)
                    destination.write(chunk)
                    digest.update(chunk)
                    completed += len(chunk)
                    if completed > self.identity.size:
                        msg = f"The {self.label} checkpoint exceeded its pinned size"
                        raise ValueError(msg)
                    if progress is not None:
                        progress(completed, self.identity.size)
                destination.flush()
                os.fsync(destination.fileno())
            if completed != self.identity.size or digest.hexdigest() != self.identity.sha256:
                msg = f"The {self.label} checkpoint failed SHA-256 validation"
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
        if not path.is_file() or path.stat().st_size != self.identity.size:
            msg = f"The pinned {self.label} checkpoint is missing or has the wrong size"
            raise ValueError(msg)
        if sha256_path(path) != self.identity.sha256:
            msg = f"The {self.label} checkpoint failed SHA-256 validation"
            raise ValueError(msg)

    def _write_marker(self) -> None:
        stat = self.checkpoint_path.stat()
        payload = {
            "schema_version": 1,
            "sha256": self.identity.sha256,
            "size": self.identity.size,
            "mtime_ns": stat.st_mtime_ns,
        }
        temporary = self.marker_path.with_name(f".{self.marker_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.marker_path)
        finally:
            temporary.unlink(missing_ok=True)
