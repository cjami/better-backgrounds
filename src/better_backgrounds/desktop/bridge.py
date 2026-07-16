"""Narrow validated bridge exposed to the embedded renderer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from PySide6.QtCore import QObject, Signal, Slot


class ViewpointMessage(BaseModel):
    """Validate the only Phase 2 renderer-to-Python payload."""

    model_config = ConfigDict(extra="forbid")

    field_of_view: float = Field(ge=24.0, le=90.0)
    horizon: float = Field(ge=-10.0, le=10.0)
    subject_depth: float = Field(ge=0.5, le=5.0)
    focus_depth: float = Field(ge=0.5, le=5.0)


class RendererBridge(QObject):
    """Expose task-specific renderer capabilities, never generic system access."""

    ready = Signal()
    viewpoint_received = Signal(object)

    @Slot()
    def renderer_ready(self) -> None:
        """Record that the trusted renderer initialized."""
        self.ready.emit()

    @Slot(str, result=bool)
    def submit_viewpoint(self, payload: str) -> bool:
        """Validate a renderer viewpoint before publishing it to Python."""
        try:
            viewpoint = ViewpointMessage.model_validate_json(payload)
        except ValidationError:
            return False
        self.viewpoint_received.emit(viewpoint)
        return True
