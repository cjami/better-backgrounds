"""Native camera and MatAnyone live-preview session widget."""

from __future__ import annotations

import base64
import binascii
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QStackedLayout, QWidget

from better_backgrounds.desktop.camera.capture import (
    CaptureProfile,
    OutputGeometry,
    QtCameraCapture,
    capture_profile,
)
from better_backgrounds.desktop.live_preview.composition import CompositionCoordinator
from better_backgrounds.desktop.live_preview.seed import SeedCoordinator
from better_backgrounds.desktop.live_preview.surface import (
    NativeCompositeSurface,
    PreparedComposite,
)
from better_backgrounds.harmonization import HarmonizationSettings
from better_backgrounds.matting.contracts import (
    LiveDiagnostics,
    LivePipelineConfig,
    MattingConfig,
    ProcessedFrame,
    SlidingFrameRate,
)
from better_backgrounds.matting.engine import (
    CompletedMatte,
    EngineFailure,
    EngineReady,
    ProcessMattingEngine,
)
from better_backgrounds.matting.runtime import packaged_checkpoint_path
from better_backgrounds.matting.seed import StableFrameSelector

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from better_backgrounds.matting.compositor import LiveComposite
    from better_backgrounds.scene import ManagedSceneResolver, SceneReference, Viewpoint

EngineFactory = Callable[[], ProcessMattingEngine]
BackgroundFactory = Callable[[], QWidget]
ALPHA_MIDPOINT = 128
MINIMUM_MATTE_OCCUPANCY = 0.01
MAXIMUM_MATTE_OCCUPANCY = 0.95
LOST_MATTE_LIMIT = 15
TARGET_MATTE_FPS = 30.0
TARGET_FRAME_BUDGET_MS = 1_000.0 / TARGET_MATTE_FPS
MATTE_INFERENCE_BUDGET_MS = TARGET_FRAME_BUDGET_MS
BACKGROUND_CAPTURE_DELAY_MS = 300
BACKGROUND_CAPTURE_RETRY_MS = 50
BACKGROUND_CAPTURE_RETRY_LIMIT = 20
BACKGROUND_REFRESH_DEBOUNCE_MS = 150
MAX_SNAPSHOT_PAYLOAD_LENGTH = 16 * 1024 * 1024
ENGINE_EVENT_FIELD_COUNT = 2


class NativeLivePreview(QWidget):
    """Own one Qt camera, seed operation, worker, and room snapshot cache."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)
    comparison_frame = Signal(object)
    harmonization_status_changed = Signal(str)
    _engine_event = Signal(object)

    def __init__(
        self,
        resolver: ManagedSceneResolver | None = None,
        parent: QWidget | None = None,
        *,
        background_factory: BackgroundFactory | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """Create retained native resources without opening a camera."""
        super().__init__(parent)
        if background_factory is None:
            from better_backgrounds.desktop.webview import (  # noqa: PLC0415
                create_background_renderer_view,
            )

            def create_background() -> QWidget:
                return create_background_renderer_view(resolver)

            background_factory = create_background
        checkpoint = packaged_checkpoint_path()

        def create_engine() -> ProcessMattingEngine:
            return ProcessMattingEngine(checkpoint)

        self._engine_factory = engine_factory or create_engine
        self._background_renderer = background_factory()
        self._surface = NativeCompositeSurface()
        self._surface.frame_painted.connect(self._frame_painted)
        self._seed_coordinator = SeedCoordinator(self)
        self._seed_coordinator.generated.connect(self._accept_seed)
        self._seed_coordinator.failed.connect(self._reject_seed)
        self._composition = CompositionCoordinator(self._surface, self)
        self._composition.frame_ready.connect(self._present_pending_matte)
        self._composition.failed.connect(self._composition_failed)
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.addWidget(self._background_renderer)
        layout.addWidget(self._surface)
        self._surface.raise_()
        progress = getattr(self._background_renderer, "scene_progressed", None)
        if progress is not None:
            progress.connect(self._scene_progressed)
        snapshot_ready = getattr(self._background_renderer, "snapshot_ready", None)
        self._snapshot_handshake = snapshot_ready is not None
        if snapshot_ready is not None:
            snapshot_ready.connect(self._snapshot_ready)
        self._background_timer = QTimer(self)
        self._background_timer.setSingleShot(True)
        self._background_timer.timeout.connect(self._capture_background)
        self._viewpoint_timer = QTimer(self)
        self._viewpoint_timer.setSingleShot(True)
        self._viewpoint_timer.timeout.connect(self._apply_pending_viewpoint)
        self._background_capture_attempts = 0
        self._poll_timer = self._create_timer(5, self._poll_engine)
        self._engine_event.connect(self._accept_engine_event)
        self._result_stop = threading.Event()
        self._result_thread: threading.Thread | None = None
        self._result_generation = 0
        self._camera_capture = QtCameraCapture(self)
        self._camera_capture.frame_captured.connect(self._camera_frame_captured)
        self._camera_capture.profile_changed.connect(self._capture_profile_changed)
        self._camera_capture.failed.connect(self._camera_failed)
        self._engine: ProcessMattingEngine | None = None
        self._selector = StableFrameSelector()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._seed_frame: NDArray[np.uint8] | None = None
        self._seed_mask: NDArray[np.uint8] | None = None
        self._state = "idle"
        self._mirrored = True
        self._mode = "show"
        self._pipeline_revision = 0
        self._wipe = 52
        self._invalid_matte_count = 0
        self._active_device = "cpu"
        self._internal_size = 360
        self._harmonization_requested = False
        self._harmonization_settings = HarmonizationSettings()
        self._harmonizer_prepare_inflight = False
        self._scene_asset_id = ""
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
        self._pending_harmonization_snapshot: tuple[int, QImage] | None = None
        self._pending_viewpoint: Viewpoint | None = None
        self._background_refresh_started_at: float | None = None
        self._background_refresh_ms = 0.0
        self._resource_active = True
        self._camera_device_id: str | None = None
        self._scene: SceneReference | None = None
        self._scene_viewpoint: Viewpoint | None = None
        self._capture_profile = capture_profile(1280, 720, 30.0, 30.0)
        self._output_geometry = self._capture_profile.output_geometry(16 / 9)
        self._surface.set_output_geometry(self._output_geometry)
        self._set_renderer_output_size(self._output_geometry)
        self._rendered_scene_asset_id = ""
        self._capture_rate = SlidingFrameRate()
        self._display_rate = SlidingFrameRate()
        self._latest_diagnostics: LiveDiagnostics | None = None

    def _create_timer(
        self,
        interval_ms: int,
        callback: Callable[[], None],
        *,
        precise: bool = False,
    ) -> QTimer:
        timer = QTimer(self)
        timer.setInterval(interval_ms)
        if precise:
            timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.timeout.connect(callback)
        return timer

    def start_camera(self, device_id: str, *, mirrored: bool) -> None:
        """Open the selected Qt camera and begin one-shot target selection."""
        self.stop_camera()
        self._mirrored = mirrored
        self._camera_device_id = device_id
        self._surface.set_mirroring(mirrored=mirrored)
        self._surface.reset_camera_harmonization()
        self._selector.reset()
        self._capture_rate.reset()
        self._display_rate.reset()
        if self._resource_active:
            self._poll_timer.start()
        self._set_state("seeding", "Hold still while the person seed is captured…")
        if self._resource_active and not self._camera_capture.start(device_id):
            self._poll_timer.stop()

    def stop_camera(self) -> None:
        """Release camera, model worker, and all queued frame ownership."""
        self._poll_timer.stop()
        self._stop_result_pump()
        self._camera_capture.stop()
        self._camera_device_id = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._selector.reset()
        self._latest_frame = None
        self._seed_frame = None
        self._seed_mask = None
        self._seed_coordinator.reset()
        self._composition.reset()
        self._capture_rate.reset()
        self._display_rate.reset()
        self._surface.reset_camera_harmonization()
        self._surface.clear_matte()
        if self._state != "idle":
            self._set_state("idle", "Camera stopped")

    def confirm_seed(self) -> None:
        """Initialize MatAnyone 2 from the user-confirmed first-frame target."""
        if self._state != "seed-ready" or self._seed_frame is None or self._seed_mask is None:
            return
        if self._engine is not None:
            self._stop_result_pump()
            self._engine.close()
        self._engine = self._engine_factory()
        try:
            self._engine.start(
                self._seed_frame,
                self._seed_mask,
                MattingConfig(
                    internal_size=540,
                    warmup_iterations=10,
                    calibrate=True,
                    latency_budget_ms=MATTE_INFERENCE_BUDGET_MS,
                ),
                self._live_pipeline_config(),
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._engine.close()
            self._engine = None
            self._set_state(
                "seed-ready",
                f"MatAnyone 2 failed to start: {str(error)[:200]}. Retry or choose a new seed.",
            )
            return
        self._set_state("initializing", "Warming up MatAnyone 2…")

        self._sync_engine_pipeline()
        self._engine.configure_harmonization(self._harmonization_settings)
        self._start_result_pump(self._engine)

    def retry_seed(self) -> None:
        """Discard the proposed target and capture a new stable frame."""
        self._begin_reseed("Hold still while a new person seed is captured…")

    def reselect_person(self) -> None:
        """Explicitly clear temporal identity and return to seed capture."""
        self._begin_reseed("Select the person again; hold still for a moment…")

    def set_scene(self, scene: SceneReference, viewpoint: Viewpoint) -> None:
        """Load one room in the hidden canvas-only snapshot renderer."""
        self._set_output_aspect_ratio(viewpoint.aspect_ratio)
        self._scene = scene
        self._scene_viewpoint = viewpoint
        self._viewpoint_timer.stop()
        self._pending_viewpoint = None
        self._scene_asset_id = scene.asset_id
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
        self._invalidate_harmonization_reference()
        setter = getattr(self._background_renderer, "set_scene", None)
        if self._resource_active and callable(setter):
            self._background_refresh_started_at = time.perf_counter()
            setter(scene, viewpoint)
            self._rendered_scene_asset_id = scene.asset_id

    def clear_scene(self) -> None:
        """Clear the room and return the native compositor to black."""
        self._viewpoint_timer.stop()
        self._pending_viewpoint = None
        self._scene_asset_id = ""
        self._scene = None
        self._scene_viewpoint = None
        self._rendered_scene_asset_id = ""
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
        self._pending_harmonization_snapshot = None
        self._background_refresh_started_at = None
        self._background_refresh_ms = 0.0
        clearer = getattr(self._background_renderer, "clear_scene", None)
        if callable(clearer):
            clearer()
        self._surface.clear_background()
        self._sync_engine_pipeline()

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Debounce expensive depth-of-field room snapshots."""
        self._set_output_aspect_ratio(viewpoint.aspect_ratio)
        previous = self._scene_viewpoint
        self._scene_viewpoint = viewpoint
        if previous is None or self._harmonization_viewpoint_changed(previous, viewpoint):
            self._invalidate_harmonization_reference()
        if not self._resource_active:
            return
        self._pending_viewpoint = viewpoint
        self._viewpoint_timer.start(BACKGROUND_REFRESH_DEBOUNCE_MS)

    @Slot()
    def _apply_pending_viewpoint(self) -> None:
        viewpoint = self._pending_viewpoint
        self._pending_viewpoint = None
        setter = getattr(self._background_renderer, "set_viewpoint", None)
        if viewpoint is None or not callable(setter):
            return
        self._background_refresh_started_at = time.perf_counter()
        setter(viewpoint)
        self._schedule_background_capture()

    def set_resource_active(self, active: bool) -> None:  # noqa: FBT001
        """Yield live capture and inference while Adjust owns interaction."""
        if active == self._resource_active:
            return
        self._resource_active = active
        if not active:
            self._poll_timer.stop()
            self._stop_result_pump()
            self._camera_capture.stop()
            self._background_timer.stop()
            self._viewpoint_timer.stop()
            return
        if self._camera_device_id is not None:
            if self._engine is not None:
                self._start_result_pump(self._engine)
            else:
                self._poll_timer.start()
            if not self._camera_capture.start(self._camera_device_id):
                self._poll_timer.stop()
        if self._scene is None or self._scene_viewpoint is None:
            return
        self._background_refresh_started_at = time.perf_counter()
        if self._rendered_scene_asset_id == self._scene.asset_id:
            setter = getattr(self._background_renderer, "set_viewpoint", None)
            if callable(setter):
                setter(self._scene_viewpoint)
                self._schedule_background_capture()
            return
        setter = getattr(self._background_renderer, "set_scene", None)
        if callable(setter):
            setter(self._scene, self._scene_viewpoint)
            self._rendered_scene_asset_id = self._scene.asset_id

    def set_scene_image(self, path: Path | None) -> None:
        """Use a verified preview image until a spatial snapshot is ready."""
        if path is not None:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull() and self._surface.set_background(
                pixmap.toImage(),
                update_harmonization_reference=not self._snapshot_handshake,
            ):
                self._sync_engine_pipeline()

    def set_presentation(self, mode: str, wipe: int = 52) -> None:
        """Reuse this session in Show and Compare without duplicate work."""
        if mode != self._mode:
            self._pipeline_revision += 1
        self._mode = mode
        self._wipe = wipe
        self._surface.set_presentation(mode, wipe)
        if self._engine is not None:
            self._engine.configure_presentation(
                mirrored=self._mirrored,
                retain_standard=mode == "compare",
                revision=self._pipeline_revision,
            )

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Mirror source and alpha together, never the room snapshot."""
        if mirrored != self._mirrored:
            self._pipeline_revision += 1
        self._mirrored = mirrored
        self._surface.set_mirroring(mirrored=mirrored)
        if self._engine is not None:
            self._engine.configure_presentation(
                mirrored=mirrored,
                retain_standard=self._mode == "compare",
                revision=self._pipeline_revision,
            )

    def set_harmonization(self, settings: HarmonizationSettings) -> None:
        """Apply the room-scoped experimental global harmonization switch."""
        self._surface.set_harmonization(settings)
        self._harmonization_settings = settings
        if self._engine is not None:
            self._engine.configure_harmonization(settings)
        self._harmonization_requested = settings.active
        if settings.active:
            engine = self._engine
            live = engine is not None and engine.ready
            if live and not engine.fused:
                self._prepare_harmonizer()
            self.harmonization_status_changed.emit(
                (
                    "Global harmonization enabled; preparing the external checkpoint…"
                    if live
                    else "Global harmonization queued until MatAnyone startup completes."
                ),
            )
        else:
            self.harmonization_status_changed.emit(
                "Global harmonization is off; identical comparison sides are expected.",
            )

    def _prepare_harmonizer(self) -> None:
        if self._harmonizer_prepare_inflight:
            return
        self._harmonizer_prepare_inflight = True

        def prepare() -> None:
            try:
                self._surface.prepare_harmonization()
            finally:
                self._harmonizer_prepare_inflight = False

        threading.Thread(target=prepare, name="harmonizer-prepare", daemon=True).start()

    @Slot(int, int)
    def _scene_progressed(self, loaded: int, total: int) -> None:
        if loaded == total:
            self._schedule_background_capture()

    @Slot(str, int, str, str)
    def _snapshot_ready(
        self,
        asset_id: str,
        revision: int,
        kind: str,
        payload: str,
    ) -> None:
        """Publish validated framebuffer pixels directly to the live compositor."""
        if (
            asset_id != self._scene_asset_id
            or kind not in {"background", "harmonization"}
            or not 1 <= len(payload) <= MAX_SNAPSHOT_PAYLOAD_LENGTH
        ):
            return
        if self._snapshot_revision_is_stale(kind, revision):
            return
        image = self._decode_snapshot(payload)
        if image is None:
            return
        if kind == "harmonization":
            if revision == self._latest_snapshot_revision:
                self._publish_harmonization_snapshot(revision, image)
            else:
                self._pending_harmonization_snapshot = (revision, image.copy())
            return
        self._publish_background_snapshot(revision, image)

    def _snapshot_revision_is_stale(self, kind: str, revision: int) -> bool:
        if kind == "background":
            return revision < self._latest_snapshot_revision
        pending_revision = (
            -1
            if self._pending_harmonization_snapshot is None
            else self._pending_harmonization_snapshot[0]
        )
        return revision <= max(self._latest_harmonization_revision, pending_revision)

    @staticmethod
    def _decode_snapshot(payload: str) -> QImage | None:
        try:
            encoded = payload.encode("ascii")
            pixels = base64.b64decode(encoded, validate=True)
        except UnicodeEncodeError, binascii.Error, ValueError:
            return None
        image = QImage.fromData(pixels)
        return None if image.isNull() else image

    def _publish_background_snapshot(self, revision: int, image: QImage) -> None:
        if self._surface.set_background(image, update_harmonization_reference=False):
            self._sync_engine_pipeline()
            self._background_timer.stop()
            self._latest_snapshot_revision = revision
            pending = self._pending_harmonization_snapshot
            if pending is not None:
                pending_revision, pending_image = pending
                if pending_revision == revision:
                    self._publish_harmonization_snapshot(pending_revision, pending_image)
                elif pending_revision < revision:
                    self._pending_harmonization_snapshot = None
            self._record_background_refresh()

    def _publish_harmonization_snapshot(self, revision: int, image: QImage) -> None:
        if self._surface.set_harmonization_reference(image):
            self._sync_engine_pipeline()
            self._latest_harmonization_revision = revision
            self._pending_harmonization_snapshot = None

    def _invalidate_harmonization_reference(self) -> None:
        self._pending_harmonization_snapshot = None
        self._surface.clear_harmonization_reference()
        self._sync_engine_pipeline()

    @staticmethod
    def _harmonization_viewpoint_changed(previous: Viewpoint, current: Viewpoint) -> bool:
        return previous.model_dump(exclude={"depth_of_field"}) != current.model_dump(
            exclude={"depth_of_field"},
        )

    def _record_background_refresh(self) -> None:
        started_at = self._background_refresh_started_at
        if started_at is not None:
            self._background_refresh_ms = (time.perf_counter() - started_at) * 1_000.0
            self._background_refresh_started_at = None

    def _schedule_background_capture(self) -> None:
        if self._snapshot_handshake:
            return
        self._background_capture_attempts = 0
        self._background_timer.start(BACKGROUND_CAPTURE_DELAY_MS)

    @Slot()
    def _capture_background(self) -> None:
        pixmap = self._background_renderer.grab()
        if not pixmap.isNull() and self._surface.set_background(
            pixmap.toImage(),
            update_harmonization_reference=not self._snapshot_handshake,
        ):
            self._sync_engine_pipeline()
            self._background_capture_attempts = 0
            self._record_background_refresh()
            return
        self._background_capture_attempts += 1
        if self._background_capture_attempts <= BACKGROUND_CAPTURE_RETRY_LIMIT:
            self._background_timer.start(BACKGROUND_CAPTURE_RETRY_MS)

    @Slot(object, float)
    def _camera_frame_captured(self, frame: object, captured_at: float) -> None:
        if not isinstance(frame, np.ndarray):
            return
        source = cast("NDArray[np.uint8]", frame)
        self._capture_rate.record(captured_at)
        if self._latest_frame is not None and source.shape != self._latest_frame.shape:
            self._begin_reseed("Camera format changed; capture a new person seed…")
        self._latest_frame = source
        if self._state not in {"live", "seed-ready"}:
            self._surface.set_raw_frame(source)
        if self._state == "seeding" and not self._seed_coordinator.active:
            candidate = self._selector.offer(source)
            if candidate is not None:
                self._generate_seed(candidate)
        elif self._state == "live" and self._engine is not None:
            self._engine.submit(source, captured_at=captured_at)

    @Slot(object)
    def _capture_profile_changed(self, profile: object) -> None:
        if not isinstance(profile, CaptureProfile):
            return
        self._capture_profile = profile
        aspect_ratio = (
            self._scene_viewpoint.aspect_ratio
            if self._scene_viewpoint is not None
            else self._output_geometry.aspect_ratio
        )
        self._apply_output_geometry(profile.output_geometry(aspect_ratio))

    def _set_output_aspect_ratio(self, aspect_ratio: float) -> None:
        self._apply_output_geometry(self._capture_profile.output_geometry(aspect_ratio))

    def _apply_output_geometry(self, geometry: OutputGeometry) -> None:
        if geometry == self._output_geometry:
            return
        self._output_geometry = geometry
        self._pipeline_revision += 1
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
        self._surface.set_output_geometry(geometry)
        self._set_renderer_output_size(geometry)
        engine = self._engine
        if engine is not None:
            try:
                engine.configure_geometry(self._live_pipeline_config())
            except ValueError:
                self._begin_reseed("Camera format changed; capture a new person seed")
                return
            self._sync_engine_pipeline()

    def _live_pipeline_config(self) -> LivePipelineConfig:
        geometry = self._output_geometry
        return LivePipelineConfig(
            output_width=geometry.width,
            output_height=geometry.height,
            aspect_ratio=geometry.aspect_ratio,
            mirrored=self._mirrored,
            retain_standard=self._mode == "compare",
            revision=self._pipeline_revision,
        )

    def _sync_engine_pipeline(self) -> None:
        engine = self._engine
        if engine is None:
            return
        background, reference, revision = self._surface.background_evidence()
        engine.set_live_background(
            background,
            reference,
            revision=revision,
        )

    def _set_renderer_output_size(self, geometry: OutputGeometry) -> None:
        setter = getattr(self._background_renderer, "set_output_size", None)
        if callable(setter):
            setter(geometry.width, geometry.height)

    def _generate_seed(self, frame: NDArray[np.uint8]) -> None:
        self._set_state("seeding", "Finding the person in the stable frame?")
        self._seed_coordinator.generate(frame)

    @Slot(object, object)
    def _accept_seed(self, frame: object, mask: object) -> None:
        if not isinstance(frame, np.ndarray) or not isinstance(mask, np.ndarray):
            self._reject_seed("Seed provider returned invalid pixels")
            return
        seed_frame = cast("NDArray[np.uint8]", frame)
        seed_mask = cast("NDArray[np.uint8]", mask)
        self._seed_frame = seed_frame
        self._seed_mask = seed_mask
        self._surface.set_seed_preview(seed_frame, seed_mask)
        self._set_state("seed-ready", "Confirm the highlighted person or retry")

    @Slot(str)
    def _reject_seed(self, message: str) -> None:
        self._selector.reset()
        self._set_state("seed-error", f"{message}. Reposition yourself, then retry.")

    @Slot()
    def _poll_engine(self) -> None:
        engine = self._engine
        if engine is None:
            return
        for event in engine.poll():
            if isinstance(event, EngineReady):
                self._seed_frame = None
                self._seed_mask = None
                self._display_rate.reset()
                self._composition.reset()
                self._active_device = event.capabilities.device_type
                self._internal_size = event.selected_internal_size
                self._invalid_matte_count = 0
                if self._harmonization_requested and not event.real_time_supported:
                    self._prepare_harmonizer()
                if event.real_time_error is not None:
                    self.harmonization_status_changed.emit(
                        f"Portable live path active: {event.real_time_error}",
                    )
                device = event.capabilities.device_type.upper()
                self._set_state("live", f"Live · MatAnyone 2 · {device}")
            elif isinstance(event, CompletedMatte):
                self._composition.submit(event)
            elif isinstance(event, ProcessedFrame):
                self._present_processed_frame(event)
            elif isinstance(event, EngineFailure):
                self._pause_tracking(
                    f"Matting stopped: {event.message}. Re-select the person to retry.",
                )

    @Slot(object)
    def _accept_engine_event(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != ENGINE_EVENT_FIELD_COUNT:
            return
        generation, event = payload
        if generation != self._result_generation:
            return
        if isinstance(event, EngineReady):
            self._seed_frame = None
            self._seed_mask = None
            self._display_rate.reset()
            self._composition.reset()
            self._active_device = event.capabilities.device_type
            self._internal_size = event.selected_internal_size
            self._invalid_matte_count = 0
            if self._harmonization_requested and not event.real_time_supported:
                self._prepare_harmonizer()
            if event.real_time_error is not None:
                self.harmonization_status_changed.emit(
                    f"Portable live path active: {event.real_time_error}",
                )
            device = event.capabilities.device_type.upper()
            self._set_state("live", f"Live - MatAnyone 2 - {device}")
        elif isinstance(event, CompletedMatte):
            self._composition.submit(event)
        elif isinstance(event, ProcessedFrame):
            self._present_processed_frame(event)
        elif isinstance(event, EngineFailure):
            self._pause_tracking(
                f"Matting stopped: {event.message}. Re-select the person to retry.",
            )

    def _present_processed_frame(self, processed: ProcessedFrame) -> None:
        if processed.pipeline_revision != self._pipeline_revision:
            return
        if not self._surface.present_processed(processed):
            return
        occupancy = processed.occupancy
        self._invalid_matte_count = (
            self._invalid_matte_count + 1
            if occupancy < MINIMUM_MATTE_OCCUPANCY or occupancy > MAXIMUM_MATTE_OCCUPANCY
            else 0
        )
        if self._invalid_matte_count >= LOST_MATTE_LIMIT:
            self._pause_tracking("Tracking paused. Re-select the person to continue.")
            return
        if processed.harmonized:
            self.harmonization_status_changed.emit(
                f"Global harmonization: {processed.harmonization_ms:.1f} ms/frame via fused CUDA.",
            )
        elif processed.harmonization_degraded:
            self.harmonization_status_changed.emit(
                "Global harmonization fell back to the standard composite: "
                + ", ".join(processed.harmonization_degraded),
            )
        now = time.monotonic()
        engine = self._engine
        timings = processed.timings
        diagnostics = LiveDiagnostics(
            capture_fps=self._capture_rate.rate,
            display_fps=self._display_rate.rate,
            mask_fps=self._display_rate.rate,
            mask_age_ms=max(0.0, now * 1_000.0 - processed.packet.captured_at),
            dropped_frames=0 if engine is None else engine.dropped_frames,
            worker_time_ms=timings.matting_ms,
            capture_width=self._capture_profile.processing_width,
            capture_height=self._capture_profile.processing_height,
            processing_width=round(
                self._capture_profile.processing_width
                * self._internal_size
                / min(
                    self._capture_profile.processing_width,
                    self._capture_profile.processing_height,
                ),
            ),
            processing_height=round(
                self._capture_profile.processing_height
                * self._internal_size
                / min(
                    self._capture_profile.processing_width,
                    self._capture_profile.processing_height,
                ),
            ),
            device_type=self._active_device,
            background_refresh_ms=self._background_refresh_ms,
            normalization_ms=timings.normalization_ms,
            queue_ms=timings.queue_ms,
            matting_ms=timings.matting_ms,
            post_processing_ms=timings.post_processing_ms,
            readback_ms=timings.readback_ms,
            output_width=processed.packet.width,
            output_height=processed.packet.height,
        )
        self._latest_diagnostics = diagnostics
        self._surface.set_diagnostics(diagnostics)
        self.diagnostics_changed.emit(diagnostics)
        if self._mode == "compare":
            comparison = self._surface.comparison_frames
            if comparison is not None:
                self.comparison_frame.emit(comparison)

    def _start_result_pump(self, engine: ProcessMattingEngine) -> None:
        wait_for_events = getattr(engine, "wait", None)
        if not callable(wait_for_events) or not self._resource_active:
            self._poll_timer.start()
            return
        self._stop_result_pump()
        self._poll_timer.stop()
        self._result_stop.clear()
        generation = self._result_generation

        def pump() -> None:
            while not self._result_stop.is_set():
                for event in wait_for_events(0.1):
                    if self._result_stop.is_set():
                        return
                    self._engine_event.emit((generation, event))

        self._result_thread = threading.Thread(
            target=pump,
            name="live-result-pump",
            daemon=True,
        )
        self._result_thread.start()

    def _stop_result_pump(self) -> None:
        self._result_generation += 1
        self._result_stop.set()
        thread = self._result_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        self._result_thread = None

    @Slot()
    def _present_pending_matte(self) -> None:
        prepared = self._composition.take_ready()
        if prepared is None:
            return
        composite = self._surface.present_matte(prepared)
        if composite is not None:
            self._handle_composite(prepared, composite)

    @Slot(str)
    def _composition_failed(self, message: str) -> None:
        self._pause_tracking(
            f"Compositing stopped: {message}. Re-select the person to retry.",
        )

    def _handle_composite(
        self,
        prepared: PreparedComposite,
        composite: LiveComposite,
    ) -> None:
        completed = prepared.completed
        occupancy = (
            float(np.count_nonzero(completed.alpha >= ALPHA_MIDPOINT)) / completed.alpha.size
        )
        self._invalid_matte_count = (
            self._invalid_matte_count + 1
            if occupancy < MINIMUM_MATTE_OCCUPANCY or occupancy > MAXIMUM_MATTE_OCCUPANCY
            else 0
        )
        if self._invalid_matte_count >= LOST_MATTE_LIMIT:
            self._pause_tracking("Tracking paused. Re-select the person to continue.")
            return
        if composite.harmonized:
            self.harmonization_status_changed.emit(
                f"Global harmonization: {composite.harmonization_ms:.1f} ms/frame "
                f"via {self._surface.harmonization_backend} "
                f"(30 FPS budget: {TARGET_FRAME_BUDGET_MS:.1f} ms).",
            )
        elif composite.harmonization_degraded:
            detail = self._surface.harmonization_error
            self.harmonization_status_changed.emit(
                "Global harmonization fell back to the standard composite: "
                + (detail or ", ".join(composite.harmonization_degraded)),
            )
        now = time.monotonic()
        display_rate = self._display_rate.rate
        engine = self._engine
        diagnostics = LiveDiagnostics(
            capture_fps=self._capture_rate.rate,
            display_fps=display_rate,
            mask_fps=display_rate,
            mask_age_ms=max(0.0, now * 1000.0 - completed.packet.captured_at),
            dropped_frames=(0 if engine is None else engine.dropped_frames)
            + self._composition.presentation_drops,
            worker_time_ms=completed.result.inference_ms,
            capture_width=completed.packet.width,
            capture_height=completed.packet.height,
            processing_width=round(
                completed.packet.width
                * self._internal_size
                / min(
                    completed.packet.width,
                    completed.packet.height,
                ),
            ),
            processing_height=round(
                completed.packet.height
                * self._internal_size
                / min(
                    completed.packet.width,
                    completed.packet.height,
                ),
            ),
            device_type=self._active_device,
            background_refresh_ms=self._background_refresh_ms,
            normalization_ms=prepared.normalization_ms,
            queue_ms=prepared.queue_ms,
            matting_ms=completed.result.inference_ms,
            post_processing_ms=prepared.post_processing_ms,
            readback_ms=prepared.readback_ms,
            output_width=prepared.source.shape[1],
            output_height=prepared.source.shape[0],
        )
        self._latest_diagnostics = diagnostics
        self._surface.set_diagnostics(diagnostics)
        self.diagnostics_changed.emit(diagnostics)
        if self._mode == "compare":
            comparison = self._surface.comparison_frames
            if comparison is not None:
                self.comparison_frame.emit(comparison)

    @Slot(float)
    def _frame_painted(self, capture_to_paint_ms: float) -> None:
        """Publish latency and cadence from the paint that actually reached Qt."""
        diagnostics = self._latest_diagnostics
        if diagnostics is None:
            return
        display_rate = self._display_rate.record(time.monotonic() * 1_000.0)
        diagnostics = diagnostics.model_copy(
            update={
                "display_fps": display_rate,
                "mask_fps": display_rate,
                "capture_to_paint_ms": capture_to_paint_ms,
            },
        )
        self._latest_diagnostics = diagnostics
        self._surface.set_diagnostics(diagnostics)
        self.diagnostics_changed.emit(diagnostics)

    def _begin_reseed(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("seeding", message)

    def _pause_tracking(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("lost", message)

    def _reset_tracking(self) -> None:
        self._composition.reset()
        if self._engine is not None:
            self._stop_result_pump()
            self._engine.close()
            self._engine = None
        self._selector.reset()
        self._seed_coordinator.reset()
        self._seed_frame = None
        self._seed_mask = None
        self._invalid_matte_count = 0
        self._surface.clear_matte()

    @Slot(str)
    def _camera_failed(self, message: str) -> None:
        self._set_state("error", message)

    def _set_state(self, state: str, message: str) -> None:
        self._state = state
        self.camera_state_changed.emit(state, message)

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        """Release native resources before the QWidget is destroyed."""
        self.stop_camera()
        self._composition.close()
        super().closeEvent(event)


def create_native_live_view(resolver: ManagedSceneResolver | None = None) -> QWidget:
    """Create the retained native camera, matting, and compositor surface."""
    return NativeLivePreview(resolver)
