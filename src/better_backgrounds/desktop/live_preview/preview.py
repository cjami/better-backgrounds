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

from better_backgrounds.desktop.camera.capture import QtCameraCapture
from better_backgrounds.desktop.live_preview.composition import CompositionCoordinator
from better_backgrounds.desktop.live_preview.seed import SeedCoordinator
from better_backgrounds.desktop.live_preview.surface import NativeCompositeSurface
from better_backgrounds.matting.contracts import LiveDiagnostics, MattingConfig, SlidingFrameRate
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

    from better_backgrounds.harmonization import HarmonizationSettings
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
PRESENTATION_INTERVAL_MS = 33
BACKGROUND_CAPTURE_DELAY_MS = 300
BACKGROUND_CAPTURE_RETRY_MS = 50
BACKGROUND_CAPTURE_RETRY_LIMIT = 20
BACKGROUND_REFRESH_DEBOUNCE_MS = 150
MAX_SNAPSHOT_PAYLOAD_LENGTH = 16 * 1024 * 1024


class NativeLivePreview(QWidget):
    """Own one Qt camera, seed operation, worker, and room snapshot cache."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)
    comparison_frame = Signal(object)
    harmonization_status_changed = Signal(str)

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
        self._seed_coordinator = SeedCoordinator(self)
        self._seed_coordinator.generated.connect(self._accept_seed)
        self._seed_coordinator.failed.connect(self._reject_seed)
        self._composition = CompositionCoordinator(self._surface, self)
        self._composition.buffer_ready.connect(self._start_presentation)
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
        self._presentation_timer = self._create_timer(
            PRESENTATION_INTERVAL_MS,
            self._present_pending_matte,
            precise=True,
        )
        self._camera_capture = QtCameraCapture(self)
        self._camera_capture.frame_captured.connect(self._camera_frame_captured)
        self._camera_capture.failed.connect(self._camera_failed)
        self._engine: ProcessMattingEngine | None = None
        self._selector = StableFrameSelector()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._seed_frame: NDArray[np.uint8] | None = None
        self._seed_mask: NDArray[np.uint8] | None = None
        self._state = "idle"
        self._mirrored = True
        self._mode = "show"
        self._wipe = 52
        self._invalid_matte_count = 0
        self._active_device = "cpu"
        self._internal_size = 360
        self._harmonization_requested = False
        self._harmonizer_prepare_inflight = False
        self._scene_asset_id = ""
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
        self._pending_viewpoint: Viewpoint | None = None
        self._background_refresh_started_at: float | None = None
        self._background_refresh_ms = 0.0
        self._resource_active = True
        self._camera_device_id: str | None = None
        self._scene: SceneReference | None = None
        self._scene_viewpoint: Viewpoint | None = None
        self._rendered_scene_asset_id = ""
        self._capture_rate = SlidingFrameRate()
        self._display_rate = SlidingFrameRate()

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
        self._presentation_timer.stop()
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

    def retry_seed(self) -> None:
        """Discard the proposed target and capture a new stable frame."""
        self._begin_reseed("Hold still while a new person seed is captured…")

    def reselect_person(self) -> None:
        """Explicitly clear temporal identity and return to seed capture."""
        self._begin_reseed("Select the person again; hold still for a moment…")

    def set_scene(self, scene: SceneReference, viewpoint: Viewpoint) -> None:
        """Load one room in the hidden canvas-only snapshot renderer."""
        self._scene = scene
        self._scene_viewpoint = viewpoint
        self._viewpoint_timer.stop()
        self._pending_viewpoint = None
        self._scene_asset_id = scene.asset_id
        self._latest_snapshot_revision = -1
        self._latest_harmonization_revision = -1
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
        self._background_refresh_started_at = None
        self._background_refresh_ms = 0.0
        clearer = getattr(self._background_renderer, "clear_scene", None)
        if callable(clearer):
            clearer()
        self._surface.clear_background()

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Debounce expensive depth-of-field room snapshots."""
        self._scene_viewpoint = viewpoint
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
            self._presentation_timer.stop()
            self._camera_capture.stop()
            self._background_timer.stop()
            self._viewpoint_timer.stop()
            return
        if self._camera_device_id is not None:
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
            if not pixmap.isNull():
                self._surface.set_background(pixmap.toImage())

    def set_presentation(self, mode: str, wipe: int = 52) -> None:
        """Reuse this session in Show and Compare without duplicate work."""
        self._mode = mode
        self._wipe = wipe
        self._surface.set_presentation(mode, wipe)

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Mirror source and alpha together, never the room snapshot."""
        self._mirrored = mirrored
        self._surface.set_mirroring(mirrored=mirrored)

    def set_harmonization(self, settings: HarmonizationSettings) -> None:
        """Apply the room-scoped experimental global harmonization switch."""
        self._surface.set_harmonization(settings)
        self._harmonization_requested = settings.active
        if settings.active:
            engine = self._engine
            live = engine is not None and engine.ready
            if live:
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
        latest_revision = (
            self._latest_snapshot_revision
            if kind == "background"
            else self._latest_harmonization_revision
        )
        if revision < latest_revision:
            return
        try:
            encoded = payload.encode("ascii")
            pixels = base64.b64decode(encoded, validate=True)
        except UnicodeEncodeError, binascii.Error, ValueError:
            return
        image = QImage.fromData(pixels)
        if image.isNull():
            return
        if kind == "harmonization":
            if self._surface.set_harmonization_reference(image):
                self._latest_harmonization_revision = revision
            return
        if self._surface.set_background(image, update_harmonization_reference=False):
            self._background_timer.stop()
            self._latest_snapshot_revision = revision
            self._record_background_refresh()

    def _record_background_refresh(self) -> None:
        started_at = self._background_refresh_started_at
        if started_at is not None:
            self._background_refresh_ms = (time.perf_counter() - started_at) * 1_000.0
            self._background_refresh_started_at = None

    def _schedule_background_capture(self) -> None:
        self._background_capture_attempts = 0
        self._background_timer.start(BACKGROUND_CAPTURE_DELAY_MS)

    @Slot()
    def _capture_background(self) -> None:
        pixmap = self._background_renderer.grab()
        if not pixmap.isNull() and self._surface.set_background(
            pixmap.toImage(),
            update_harmonization_reference=not self._snapshot_handshake,
        ):
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
                if self._harmonization_requested:
                    self._prepare_harmonizer()
                device = event.capabilities.device_type.upper()
                self._set_state("live", f"Live · MatAnyone 2 · {device}")
            elif isinstance(event, CompletedMatte):
                self._composition.submit(event)
            elif isinstance(event, EngineFailure):
                self._pause_tracking(
                    f"Matting stopped: {event.message}. Re-select the person to retry.",
                )

    @Slot()
    def _present_pending_matte(self) -> None:
        prepared = self._composition.take_ready()
        if prepared is None:
            return
        composite = self._surface.present_matte(prepared)
        if composite is not None:
            self._handle_composite(prepared.completed, composite)

    @Slot()
    def _start_presentation(self) -> None:
        if not self._presentation_timer.isActive():
            self._presentation_timer.start()

    @Slot(str)
    def _composition_failed(self, message: str) -> None:
        self._pause_tracking(
            f"Compositing stopped: {message}. Re-select the person to retry.",
        )

    def _handle_composite(
        self,
        completed: CompletedMatte,
        composite: LiveComposite,
    ) -> None:
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
                f"Global Harmonizer: {composite.harmonization_ms:.1f} ms/frame "
                f"on {self._surface.harmonization_backend} "
                f"(30 FPS budget: {TARGET_FRAME_BUDGET_MS:.1f} ms).",
            )
        elif composite.harmonization_degraded:
            detail = self._surface.harmonization_error
            self.harmonization_status_changed.emit(
                "Global harmonization fell back to the standard composite: "
                + (detail or ", ".join(composite.harmonization_degraded)),
            )
        now = time.monotonic()
        display_rate = self._display_rate.record(now * 1_000.0)
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
        )
        self._surface.set_diagnostics(diagnostics)
        self.diagnostics_changed.emit(diagnostics)
        if self._mode == "compare":
            comparison = self._surface.comparison_frames
            if comparison is not None:
                self.comparison_frame.emit(comparison)

    def _begin_reseed(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("seeding", message)

    def _pause_tracking(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("lost", message)

    def _reset_tracking(self) -> None:
        self._presentation_timer.stop()
        self._composition.reset()
        if self._engine is not None:
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
        super().closeEvent(event)


def create_native_live_view(resolver: ManagedSceneResolver | None = None) -> QWidget:
    """Create the retained native camera, matting, and compositor surface."""
    return NativeLivePreview(resolver)
