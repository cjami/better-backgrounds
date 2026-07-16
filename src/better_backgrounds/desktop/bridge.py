"""Narrow validated bridge exposed to the embedded scene renderer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from PySide6.QtCore import QObject, Signal, Slot

from better_backgrounds.scene import Viewpoint

ViewpointMessage = Viewpoint


class SceneErrorMessage(BaseModel):
    """Validate a bounded renderer failure before exposing it to the desktop."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    code: str = Field(pattern=r"^[a-z0-9_]{2,80}$")
    message: str = Field(min_length=1, max_length=300)


class RendererBridge(QObject):
    """Expose task-specific renderer capabilities, never generic system access."""

    ready = Signal()
    viewpoint_received = Signal(object)
    scene_progressed = Signal(str, int, int)
    scene_failed = Signal(object)
    scene_requested = Signal(str, str, str)
    viewpoint_requested = Signal(str)
    reset_requested = Signal()

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

    @Slot(str, int, int, result=bool)
    def report_scene_progress(self, asset_id: str, loaded: int, total: int) -> bool:
        """Publish bounded monotonic load progress from the trusted page."""
        if not asset_id or loaded < 0 or total <= 0 or loaded > total:
            return False
        self.scene_progressed.emit(asset_id, loaded, total)
        return True

    @Slot(str, str, str, result=bool)
    def report_scene_error(self, asset_id: str, code: str, message: str) -> bool:
        """Publish a validated recoverable renderer error."""
        try:
            error = SceneErrorMessage(asset_id=asset_id, code=code, message=message)
        except ValidationError:
            return False
        self.scene_failed.emit(error)
        return True

    def request_scene(self, asset_id: str, url: str, viewpoint: Viewpoint) -> None:
        """Ask the renderer to load one managed asset and camera."""
        self.scene_requested.emit(asset_id, url, viewpoint.model_dump_json())

    def request_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Ask the renderer to apply validated Python-owned controls."""
        self.viewpoint_requested.emit(viewpoint.model_dump_json())

    def request_reset(self) -> None:
        """Ask the renderer to return to the room's usable preset."""
        self.reset_requested.emit()
