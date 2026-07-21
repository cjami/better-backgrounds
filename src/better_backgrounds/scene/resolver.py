"""Controlled URL resolution for verified scene resources."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from better_backgrounds.scene.models import SCENE_SCHEME, SceneReference

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PySide6.QtCore import QUrl

    from better_backgrounds.scene.assets import AssetInstaller


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

    def register(self, reference: SceneReference) -> None:
        """Add one validated locally generated scene to the controlled index."""
        self._references[reference.asset_id] = reference

    def unregister(self, asset_id: str) -> None:
        """Drop one scene from the controlled index after its room is deleted."""
        self._references.pop(asset_id, None)
