"""Narrow validated bridge exposed to the embedded scene renderer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from PySide6.QtCore import QObject, Signal, Slot

from better_backgrounds.scene import Viewpoint

ViewpointMessage = Viewpoint
MAX_CAMERA_STATUS_LENGTH = 300
MAX_SNAPSHOT_BASE64_LENGTH = 16 * 1024 * 1024
MAX_RENDERER_OUTPUT_SIZE = 8_192


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
    snapshot_ready = Signal(str, int, str, str)
    scene_requested = Signal(str, str, str, bool)
    viewpoint_requested = Signal(str)
    reset_requested = Signal()
    scene_cleared = Signal()
    output_size_requested = Signal(int, int)
    snapshot_requested = Signal()
    renderer_active_requested = Signal(bool)

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

    @Slot(str, int, str, str, result=bool)
    def report_snapshot_ready(
        self,
        asset_id: str,
        revision: int,
        kind: str,
        payload: str,
    ) -> bool:
        """Publish a settled, revision-tagged renderer output."""
        if (
            not asset_id
            or revision < 0
            or kind not in {"background", "harmonization"}
            or not 1 <= len(payload) <= MAX_SNAPSHOT_BASE64_LENGTH
        ):
            return False
        self.snapshot_ready.emit(asset_id, revision, kind, payload)
        return True

    def request_scene(
        self,
        asset_id: str,
        url: str,
        viewpoint: Viewpoint,
        *,
        metric_depth_available: bool = False,
    ) -> None:
        """Ask the renderer to load one managed asset and camera."""
        self.scene_requested.emit(
            asset_id,
            url,
            viewpoint.model_dump_json(),
            metric_depth_available,
        )

    def request_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Ask the renderer to apply validated Python-owned controls."""
        self.viewpoint_requested.emit(viewpoint.model_dump_json())

    def request_reset(self) -> None:
        """Ask the renderer to return to the room's usable preset."""
        self.reset_requested.emit()

    def request_scene_clear(self) -> None:
        """Ask the renderer to remove an unavailable or deselected scene."""
        self.scene_cleared.emit()

    def request_output_size(self, width: int, height: int) -> None:
        """Set a bounded fixed renderer framebuffer for native snapshots."""
        if (
            not 1 <= width <= MAX_RENDERER_OUTPUT_SIZE
            or not 1 <= height <= MAX_RENDERER_OUTPUT_SIZE
        ):
            msg = "renderer output dimensions must be between 1 and 8192 pixels"
            raise ValueError(msg)
        self.output_size_requested.emit(width, height)

    def request_snapshot(self) -> None:
        """Capture the framebuffer currently visible in Adjust."""
        self.snapshot_requested.emit()

    def request_renderer_active(self, *, active: bool) -> None:
        """Suspend or resume the interactive renderer's frame loop."""
        self.renderer_active_requested.emit(active)


class CameraDeviceMessage(BaseModel):
    """Validate a browser camera description before it reaches application state."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=4_096)
    description: str = Field(min_length=1, max_length=200)


class PipelineDiagnosticsMessage(BaseModel):
    """Validate bounded local performance counters from the live renderer."""

    model_config = ConfigDict(extra="forbid")

    display_fps: float = Field(ge=0, le=1_000, allow_inf_nan=False)
    mask_fps: float = Field(ge=0, le=1_000, allow_inf_nan=False)
    mask_age_ms: float = Field(ge=0, le=60_000, allow_inf_nan=False)
    dropped_frames: int = Field(ge=0)
    worker_time_ms: float = Field(ge=0, le=60_000, allow_inf_nan=False)
    processing_width: int = Field(gt=0, le=8_192)
    processing_height: int = Field(gt=0, le=8_192)


class LiveRendererBridge(RendererBridge):
    """Add narrow camera and matting messages to the scene-renderer bridge."""

    camera_start_requested = Signal(str, bool)
    camera_stop_requested = Signal()
    mirroring_requested = Signal(bool)
    matting_settings_requested = Signal(str)
    camera_state_changed = Signal(str, str)
    camera_devices_received = Signal(object)
    diagnostics_received = Signal(object)

    @Slot(str, str, result=bool)
    def report_camera_state(self, state: str, message: str) -> bool:
        """Publish one bounded camera lifecycle state."""
        if state not in {"idle", "starting", "live", "error", "lost"}:
            return False
        if not 1 <= len(message) <= MAX_CAMERA_STATUS_LENGTH:
            return False
        self.camera_state_changed.emit(state, message)
        return True

    @Slot(str, result=bool)
    def report_camera_devices(self, payload: str) -> bool:
        """Publish camera identifiers revealed after explicit permission."""
        try:
            devices = TypeAdapter(list[CameraDeviceMessage]).validate_json(payload)
        except ValidationError:
            return False
        self.camera_devices_received.emit(tuple(devices))
        return True

    @Slot(str, result=bool)
    def report_diagnostics(self, payload: str) -> bool:
        """Publish local-only live pipeline diagnostics."""
        try:
            diagnostics = PipelineDiagnosticsMessage.model_validate_json(payload)
        except ValidationError:
            return False
        self.diagnostics_received.emit(diagnostics)
        return True

    def request_camera_start(self, preferred_label: str, *, mirrored: bool) -> None:
        """Start capture after the Python owner records an explicit user action."""
        self.camera_start_requested.emit(preferred_label[:200], mirrored)

    def request_camera_stop(self) -> None:
        """Stop capture and release the browser stream."""
        self.camera_stop_requested.emit()

    def request_mirroring(self, *, mirrored: bool) -> None:
        """Mirror only the webcam foreground, never the spatial room."""
        self.mirroring_requested.emit(mirrored)

    def request_matting_settings(self, payload: str) -> None:
        """Forward Python-validated refinement controls to the worker."""
        self.matting_settings_requested.emit(payload)
