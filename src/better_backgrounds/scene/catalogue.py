"""Versioned persistence for locally reconstructed scenes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from better_backgrounds.scene.models import (
    SceneReference,
    StrictModel,
    normalize_colmap_scene_reference,
)

if TYPE_CHECKING:
    from pathlib import Path

SCENE_CATALOGUE_V2 = 2


class LegacySceneCatalogueDocument(StrictModel):
    """Read generated rooms written before coordinate normalization."""

    schema_version: Literal[1] = 1
    scenes: tuple[SceneReference, ...] = ()


class SceneCatalogueDocument(StrictModel):
    """Version locally generated rooms separately from bundled samples."""

    schema_version: Literal[3] = 3
    scenes: tuple[SceneReference, ...] = ()


class SceneCatalogueV2Document(StrictModel):
    """Read normalized generated rooms written before SHARP provenance."""

    schema_version: Literal[2] = 2
    scenes: tuple[SceneReference, ...] = ()


class SceneCatalogue:
    """Persist generated scene references through atomic replacement."""

    def __init__(self, path: Path) -> None:
        """Use one application-owned catalogue path."""
        self.path = path

    def scenes(self) -> tuple[SceneReference, ...]:
        """Return valid generated rooms, ignoring a corrupt document."""
        return self._read().scenes

    def find(self, asset_id: str) -> SceneReference | None:
        """Find a generated room by stable identifier."""
        return next((scene for scene in self.scenes() if scene.asset_id == asset_id), None)

    def save(self, reference: SceneReference) -> None:
        """Insert or replace one room without disturbing catalogue order."""
        existing = [scene for scene in self.scenes() if scene.asset_id != reference.asset_id]
        self._write(SceneCatalogueDocument(scenes=(reference, *existing)))

    def delete(self, asset_id: str) -> None:
        """Drop one generated room, leaving the remaining catalogue order intact."""
        remaining = tuple(scene for scene in self.scenes() if scene.asset_id != asset_id)
        self._write(SceneCatalogueDocument(scenes=remaining))

    def _write(self, document: SceneCatalogueDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self) -> SceneCatalogueDocument:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("schema_version") == 1:
                legacy = LegacySceneCatalogueDocument.model_validate(payload)
                return SceneCatalogueDocument(
                    scenes=tuple(
                        normalize_colmap_scene_reference(scene) for scene in legacy.scenes
                    ),
                )
            if isinstance(payload, dict) and payload.get("schema_version") == SCENE_CATALOGUE_V2:
                previous = SceneCatalogueV2Document.model_validate(payload)
                return SceneCatalogueDocument(scenes=previous.scenes)
            return SceneCatalogueDocument.model_validate(payload)
        except OSError, TypeError, ValueError:
            return SceneCatalogueDocument()
