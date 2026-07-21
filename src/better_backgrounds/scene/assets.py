"""Managed sample-scene manifests and atomic asset installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from contextlib import closing, suppress
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal
from urllib.request import urlopen
from uuid import uuid4

from better_backgrounds.scene.models import AssetResource, SceneReference, StrictModel

CHUNK_SIZE = 64 * 1024


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
            self._write_marker(marker, self._digest(reference))
            staging.replace(target)
            self._verified[reference.asset_id] = self._digest(reference)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    def adopt(self, reference: SceneReference, source_root: Path) -> Path:
        """Atomically publish verified locally generated scene resources."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / reference.asset_id
        staging = self.root / f".{reference.asset_id}.{uuid4().hex}.part"
        try:
            staging.mkdir()
            for resource in reference.resources:
                self._copy_generated_resource(resource, source_root, staging)
            self._write_marker(staging / ".complete.json", self._digest(reference))
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
            self._verified[reference.asset_id] = self._digest(reference)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    def _copy_generated_resource(
        self,
        resource: AssetResource,
        source_root: Path,
        staging: Path,
    ) -> None:
        source = source_root.joinpath(*PurePosixPath(resource.path).parts).resolve()
        if not source.is_relative_to(source_root.resolve()) or not source.is_file():
            msg = f"generated scene is missing {resource.path}"
            raise ValueError(msg)
        if source.stat().st_size != resource.size or self._file_digest(source) != resource.sha256:
            msg = f"generated scene integrity failure for {resource.path}"
            raise ValueError(msg)
        destination = staging.joinpath(*PurePosixPath(resource.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def _download(
        self,
        resource: AssetResource,
        destination: Path,
        already_completed: int,
        reference: SceneReference,
        progress: ProgressCallback | None,
    ) -> int:
        if resource.url is None:
            msg = f"generated resource {resource.path} cannot be downloaded"
            raise ValueError(msg)
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

    def remove(self, reference: SceneReference) -> None:
        """Delete one cached scene directory and forget its verification."""
        self._verified.pop(reference.asset_id, None)
        target = self.root / reference.asset_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

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
        if not isinstance(marker_value, dict) or marker_value.get("schema_version") not in {1, 2}:
            return False
        for resource in reference.resources:
            path = target.joinpath(*PurePosixPath(resource.path).parts)
            if not path.is_file() or path.stat().st_size != resource.size:
                return False
            if self._file_digest(path) != resource.sha256:
                return False
        self._verified[reference.asset_id] = expected_digest
        if marker_value != {"schema_version": 2, "resources": expected_digest}:
            with suppress(OSError):
                self._write_marker(marker, expected_digest)
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
        resources = reference.model_dump(
            mode="json",
            include={"asset_id", "format", "entrypoint", "resources"},
        )
        value = json.dumps(resources, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _write_marker(path: Path, digest: str) -> None:
        path.write_text(
            json.dumps({"schema_version": 2, "resources": digest}),
            encoding="utf-8",
        )

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
