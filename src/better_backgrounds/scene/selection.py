"""Persist the room opened by the desktop application."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from better_backgrounds.scene.models import StrictModel

if TYPE_CHECKING:
    from pathlib import Path


class RoomSelectionDocument(StrictModel):
    """Record one stable room identifier."""

    schema_version: Literal[1] = 1
    room_id: str = Field(min_length=1, max_length=64)


class RoomSelectionStore:
    """Load and atomically save the most recently selected room."""

    def __init__(self, path: Path) -> None:
        """Use an application-data document independent of scene assets."""
        self.path = path

    def load(self) -> str | None:
        """Return no preference when the document is absent or invalid."""
        try:
            document = RoomSelectionDocument.model_validate_json(
                self.path.read_text(encoding="utf-8"),
            )
        except OSError, ValidationError:
            return None
        return document.room_id

    def save(self, room_id: str) -> None:
        """Atomically replace the selected stable room identifier."""
        document = RoomSelectionDocument(room_id=room_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
