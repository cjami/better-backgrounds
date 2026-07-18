"""Room-scoped virtual camera persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field

from better_backgrounds.scene.models import StrictModel, Viewpoint

if TYPE_CHECKING:
    from pathlib import Path


class ViewpointDocument(StrictModel):
    """Version all persisted room viewpoints together."""

    schema_version: Literal[4] = 4
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
        self._write(document)

    def delete(self, room_id: str) -> None:
        """Discard a saved viewpoint after its scene is rebuilt or re-imported."""
        document = self._read()
        if document.rooms.pop(room_id, None) is not None:
            self._write(document)

    def _write(self, document: ViewpointDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self) -> ViewpointDocument:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("schema_version") in {1, 2, 3}:
                payload = {**payload, "schema_version": 4}
            return ViewpointDocument.model_validate(payload)
        except OSError, TypeError, ValueError:
            return ViewpointDocument()
