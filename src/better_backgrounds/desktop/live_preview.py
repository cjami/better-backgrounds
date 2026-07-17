"""Native Qt webcam, MatAnyone 2 session, and exact-frame compositor."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QStackedLayout, QWidget

from better_backgrounds.compositor import (
    LiveComposite,
    background_has_content,
    compose_live_frame,
)
from better_backgrounds.desktop.camera_capture import QtCameraCapture, qimage_to_rgb
from better_backgrounds.harmonizer_runtime import HarmonizerAppearanceHarmonizer
from better_backgrounds.live_matting import LiveDiagnostics, MattingConfig, SlidingFrameRate
from better_backgrounds.matanyone_runtime import packaged_checkpoint_path
from better_backgrounds.matting_engine import (
    CompletedMatte,
    EngineFailure,
    EngineReady,
    ProcessMattingEngine,
)
from better_backgrounds.seed_mask import MediaPipeSeedProvider, StableFrameSelector

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from better_backgrounds.harmonization import HarmonizationSettings
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
PRESENTATION_BUFFER_SIZE = 2
BACKGROUND_CAPTURE_DELAY_MS = 120
BACKGROUND_CAPTURE_RETRY_MS = 50
BACKGROUND_CAPTURE_RETRY_LIMIT = 20


def rgb_to_qimage(pixels: NDArray[np.uint8]) -> QImage:
    """Copy tightly packed RGB pixels into an independently owned Qt image."""
    contiguous = np.ascontiguousarray(pixels)
    height, width = contiguous.shape[:2]
    return QImage(
        contiguous.data,
        width,
        height,
        contiguous.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def gray_to_qimage(pixels: NDArray[np.uint8]) -> QImage:
    """Copy one grayscale plane without expanding it to three channels."""
    contiguous = np.ascontiguousarray(pixels)
    height, width = contiguous.shape
    return QImage(
        contiguous.data,
        width,
        height,
        contiguous.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()


@dataclass(frozen=True, slots=True)
class PreparedComposite:
    """Carry exact-frame pixels from background composition to Qt presentation."""

    completed: CompletedMatte
    source: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    composite: LiveComposite
    compare_mode: bool


class NativeCompositeSurface(QWidget):
    """Paint the current exact-frame composite and comparison wipe."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a GPU-independent presentation surface."""
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAccessibleName("Native MatAnyone 2 webcam composite")
        self._background: NDArray[np.uint8] | None = None
        self._resized_backgrounds: dict[tuple[int, int], NDArray[np.uint8]] = {}
        self._background_revision = 0
        self._raw_source: NDArray[np.uint8] | None = None
        self._packet = None
        self._matte = None
        self._alpha: NDArray[np.uint8] | None = None
        self._source_image = QImage()
        self._composite_image = QImage()
        self._standard_composite_image = QImage()
        self._seed_image = QImage()
        self._mask_image = QImage()
        self._mask_label = ""
        self._mode = "show"
        self._wipe = 52
        self._mirrored = True
        self._diagnostics = "Waiting for camera"
        self._last_composite: LiveComposite | None = None
        self._harmonizer = HarmonizerAppearanceHarmonizer()

    @property
    def last_composite(self) -> LiveComposite | None:
        """Return the most recently painted exact-frame evidence."""
        return self._last_composite

    @property
    def harmonization_backend(self) -> str:
        """Return the active full-resolution appearance backend."""
        return self._harmonizer.backend_name

    @property
    def harmonization_error(self) -> str | None:
        """Return the bounded external-model failure for user-facing diagnostics."""
        return self._harmonizer.error

    def prepare_harmonization(self) -> None:
        """Load the external model outside the live frame callback."""
        self._harmonizer.prepare()

    @property
    def comparison_frames(self) -> tuple[QImage, QImage] | None:
        """Return retained standard and enhanced frames without repainting the widget."""
        if self._standard_composite_image.isNull() or self._composite_image.isNull():
            return None
        return self._standard_composite_image, self._composite_image

    def set_background(self, image: QImage) -> bool:
        """Atomically replace the immutable room snapshot."""
        if image.isNull():
            return False
        background = qimage_to_rgb(image)
        if not background_has_content(background):
            return False
        self._background = background
        self._resized_backgrounds.clear()
        self._background_revision += 1
        self._harmonizer.set_room(background, revision=self._background_revision)
        self._recompose()
        return True

    def clear_background(self) -> None:
        """Discard room-scoped pixels and appearance evidence together."""
        self._background = None
        self._resized_backgrounds.clear()
        self._background_revision += 1
        self._harmonizer.clear_room()
        self._recompose()

    def set_harmonization(self, settings: HarmonizationSettings) -> None:
        """Apply the room-scoped global harmonization switch and refresh the retained frame."""
        self._harmonizer.configure(settings)
        self._recompose()

    def reset_camera_harmonization(self) -> None:
        """Clear live estimates when the camera source changes."""
        self._harmonizer.reset_camera()

    def set_raw_frame(self, source: NDArray[np.uint8]) -> None:
        """Show native camera pixels while preparing or confirming a seed."""
        self._raw_source = source.copy()
        display = np.flip(source, axis=1) if self._mirrored else source
        self._source_image = rgb_to_qimage(display)
        self._seed_image = QImage()
        self.update()

    def set_seed_preview(
        self,
        source: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> None:
        """Tint the proposed target so confirmation is explicit."""
        display_source = np.flip(source, axis=1).copy() if self._mirrored else source.copy()
        display_mask = np.flip(mask, axis=1).copy() if self._mirrored else mask
        tint = np.array([224, 163, 74], dtype=np.float32)
        selected = display_mask >= ALPHA_MIDPOINT
        preview = display_source.copy()
        preview[selected] = np.rint(preview[selected] * 0.65 + tint * 0.35).astype(np.uint8)
        self._raw_source = source.copy()
        self._source_image = rgb_to_qimage(display_source)
        self._seed_image = rgb_to_qimage(preview)
        self._mask_image = gray_to_qimage(display_mask)
        self._mask_label = "PERSON FOUND"
        self.update()

    def apply_matte(self, completed: CompletedMatte) -> LiveComposite:
        """Compose the exact frame/matte pair returned by the scheduler."""
        prepared = self.prepare_matte(completed)
        composite = self.present_matte(prepared)
        if composite is None:
            msg = "room changed before the composite could be presented"
            raise RuntimeError(msg)
        return composite

    def prepare_matte(self, completed: CompletedMatte) -> PreparedComposite:
        """Compose exact-frame pixels without touching QWidget presentation state."""
        source = np.flip(completed.source, axis=1).copy() if self._mirrored else completed.source
        alpha = np.flip(completed.alpha, axis=1).copy() if self._mirrored else completed.alpha
        background = self._background
        revision = self._background_revision
        if background is None:
            background = np.zeros_like(source)
        elif background.shape != source.shape:
            shape = source.shape[:2]
            resized = self._resized_backgrounds.get(shape)
            if resized is None:
                resized = cast(
                    "NDArray[np.uint8]",
                    cv2.resize(
                        background,
                        (source.shape[1], source.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                )
                self._resized_backgrounds[shape] = resized
            background = resized
        compare_mode = self._mode == "compare"
        composite = compose_live_frame(
            completed.packet,
            completed.result,
            source,
            alpha,
            background,
            revision=revision,
            harmonizer=self._harmonizer,
            retain_standard=compare_mode,
        )
        return PreparedComposite(completed, source, alpha, composite, compare_mode)

    def present_matte(self, prepared: PreparedComposite) -> LiveComposite | None:
        """Publish one prepared image set on the Qt thread."""
        composite = prepared.composite
        if composite.background_revision != self._background_revision:
            return None
        self._packet = prepared.completed.packet
        self._matte = prepared.completed.result
        self._raw_source = prepared.completed.source
        self._alpha = prepared.completed.alpha
        self._seed_image = QImage()
        self._mask_label = "LIVE MATTE"
        self._source_image = QImage()
        self._composite_image = rgb_to_qimage(composite.image)
        self._standard_composite_image = (
            rgb_to_qimage(composite.standard_image) if prepared.compare_mode else QImage()
        )
        self._mask_image = gray_to_qimage(prepared.alpha)
        self._last_composite = composite
        self.update()
        return composite

    def set_presentation(self, mode: str, wipe: int) -> None:
        """Switch presentation without duplicating camera or inference work."""
        changed = mode in {"show", "compare"} and mode != self._mode
        if mode in {"show", "compare"}:
            self._mode = mode
        self._wipe = min(100, max(0, wipe))
        if changed and self._packet is not None:
            self._recompose()
        else:
            self.update()

    def set_mirroring(self, *, mirrored: bool) -> None:
        """Mirror source and alpha together while keeping the room unchanged."""
        self._mirrored = mirrored
        if self._packet is not None and self._alpha is not None:
            self._recompose()
        elif self._raw_source is not None:
            self.set_raw_frame(self._raw_source)

    def set_diagnostics(self, diagnostics: LiveDiagnostics) -> None:
        """Paint local performance evidence without adding product controls."""
        self._diagnostics = (
            f"{diagnostics.capture_fps:.0f} camera · {diagnostics.display_fps:.0f} output fps · "
            f"{diagnostics.mask_age_ms:.0f} ms · "
            f"{diagnostics.worker_time_ms:.1f} ms worker · "
            f"{diagnostics.capture_width}x{diagnostics.capture_height} input → "
            f"{diagnostics.processing_width}x{diagnostics.processing_height} matte · "
            f"{diagnostics.device_type.upper()} · {diagnostics.dropped_frames} dropped"
        )
        composite = self._last_composite
        if composite is not None and composite.harmonized:
            self._diagnostics += f" · {composite.harmonization_ms:.1f} ms appearance"
        if composite is not None and composite.harmonization_degraded:
            fallback = ", ".join(composite.harmonization_degraded)
            self._diagnostics += f" · appearance fallback: {fallback}"
        self.update()

    def clear_matte(self) -> None:
        """Discard temporal output before selecting a new target."""
        self._packet = None
        self._matte = None
        self._alpha = None
        self._composite_image = QImage()
        self._standard_composite_image = QImage()
        self._mask_image = QImage()
        self._mask_label = ""
        self._last_composite = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002, N802
        """Draw one image path and an optional comparison clip."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#090a0c"))
        primary = self._seed_image if not self._seed_image.isNull() else self._composite_image
        if primary.isNull():
            primary = self._source_image
        if not primary.isNull():
            self._draw_cover(painter, primary)
        if (
            self._mode == "compare"
            and not self._standard_composite_image.isNull()
            and not self._composite_image.isNull()
        ):
            split = round(self.width() * self._wipe / 100)
            painter.save()
            painter.setClipRect(QRect(0, 0, split, self.height()))
            self._draw_cover(painter, self._standard_composite_image)
            painter.restore()
            painter.setPen(QPen(QColor("#e0a34a"), 2))
            painter.drawLine(split, 0, split, self.height())
        if not self._mask_image.isNull():
            box = QRect(16, self.height() - 106, 144, 81)
            painter.setPen(QColor(235, 236, 239, 190))
            painter.drawText(
                QRectF(box.left(), box.top() - 18, box.width(), 14),
                Qt.AlignmentFlag.AlignLeft,
                self._mask_label,
            )
            painter.drawImage(box, self._mask_image)
            painter.setPen(QPen(QColor("#e0a34a"), 1))
            painter.drawRect(box)
        painter.setPen(QColor(235, 236, 239, 175))
        painter.drawText(
            QRectF(14, self.height() - 24, self.width() - 28, 18),
            Qt.AlignmentFlag.AlignRight,
            self._diagnostics,
        )

    def _recompose(self) -> LiveComposite | None:
        if (
            self._packet is None
            or self._matte is None
            or self._alpha is None
            or self._raw_source is None
        ):
            self.update()
            return None
        completed = CompletedMatte(
            self._packet,
            self._matte,
            self._raw_source,
            self._alpha,
        )
        return self.present_matte(self.prepare_matte(completed))

    def _draw_cover(self, painter: QPainter, image: QImage) -> None:
        scale = max(self.width() / image.width(), self.height() / image.height())
        source_width = self.width() / scale
        source_height = self.height() / scale
        source = QRectF(
            (image.width() - source_width) / 2,
            (image.height() - source_height) / 2,
            source_width,
            source_height,
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(self.rect()), image, source)


class NativeLivePreview(QWidget):
    """Own one Qt camera, seed operation, worker, and room snapshot cache."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)
    comparison_frame = Signal(object)
    harmonization_status_changed = Signal(str)
    _seed_generated = Signal(object, object)
    _seed_failed = Signal(str)
    _composite_prepared = Signal(int, object)
    _composite_failed = Signal(int, str)

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
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.addWidget(self._background_renderer)
        layout.addWidget(self._surface)
        self._surface.raise_()
        progress = getattr(self._background_renderer, "scene_progressed", None)
        if progress is not None:
            progress.connect(self._scene_progressed)
        self._background_timer = QTimer(self)
        self._background_timer.setSingleShot(True)
        self._background_timer.timeout.connect(self._capture_background)
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
        self._seed_inflight = False
        self._state = "idle"
        self._mirrored = True
        self._mode = "show"
        self._wipe = 52
        self._invalid_matte_count = 0
        self._active_device = "cpu"
        self._internal_size = 360
        self._harmonization_requested = False
        self._initialize_composition_state()
        self._capture_rate = SlidingFrameRate()
        self._display_rate = SlidingFrameRate()
        self._connect_internal_signals()

    def _initialize_composition_state(self) -> None:
        self._pending_matte: CompletedMatte | None = None
        self._prepared_composites: deque[PreparedComposite] = deque(
            maxlen=PRESENTATION_BUFFER_SIZE,
        )
        self._composition_inflight = False
        self._session_revision = 0
        self._presentation_drops = 0
        self._harmonizer_prepare_inflight = False

    def _connect_internal_signals(self) -> None:
        self._seed_generated.connect(self._accept_seed)
        self._seed_failed.connect(self._reject_seed)
        self._composite_prepared.connect(self._accept_prepared_composite)
        self._composite_failed.connect(self._reject_prepared_composite)

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
        self._surface.set_mirroring(mirrored=mirrored)
        self._surface.reset_camera_harmonization()
        self._selector.reset()
        self._capture_rate.reset()
        self._display_rate.reset()
        self._poll_timer.start()
        self._set_state("seeding", "Hold still while the person seed is captured…")
        if not self._camera_capture.start(device_id):
            self._poll_timer.stop()

    def stop_camera(self) -> None:
        """Release camera, model worker, and all queued frame ownership."""
        self._poll_timer.stop()
        self._presentation_timer.stop()
        self._camera_capture.stop()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._selector.reset()
        self._latest_frame = None
        self._seed_frame = None
        self._seed_mask = None
        self._seed_inflight = False
        self._pending_matte = None
        self._prepared_composites.clear()
        self._composition_inflight = False
        self._session_revision += 1
        self._presentation_drops = 0
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
        setter = getattr(self._background_renderer, "set_scene", None)
        if callable(setter):
            setter(scene, viewpoint)

    def clear_scene(self) -> None:
        """Clear the room and return the native compositor to black."""
        clearer = getattr(self._background_renderer, "clear_scene", None)
        if callable(clearer):
            clearer()
        self._surface.clear_background()

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Refresh the immutable room snapshot after viewpoint changes."""
        setter = getattr(self._background_renderer, "set_viewpoint", None)
        if callable(setter):
            setter(viewpoint)
            self._schedule_background_capture()

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

    def _schedule_background_capture(self) -> None:
        self._background_capture_attempts = 0
        self._background_timer.start(BACKGROUND_CAPTURE_DELAY_MS)

    @Slot()
    def _capture_background(self) -> None:
        pixmap = self._background_renderer.grab()
        if not pixmap.isNull() and self._surface.set_background(pixmap.toImage()):
            self._background_capture_attempts = 0
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
        if self._state == "seeding" and not self._seed_inflight:
            candidate = self._selector.offer(source)
            if candidate is not None:
                self._generate_seed(candidate)
        elif self._state == "live" and self._engine is not None:
            self._engine.submit(source, captured_at=captured_at)

    def _generate_seed(self, frame: NDArray[np.uint8]) -> None:
        self._seed_inflight = True
        self._set_state("seeding", "Finding the person in the stable frame…")

        def generate() -> None:
            provider = None
            try:
                provider = MediaPipeSeedProvider()
                mask = provider.generate(frame)
            except (OSError, RuntimeError, ValueError) as error:
                self._seed_failed.emit(str(error)[:240])
            else:
                self._seed_generated.emit(frame, mask)
            finally:
                if provider is not None:
                    provider.close()

        threading.Thread(target=generate, name="mediapipe-seed", daemon=True).start()

    @Slot(object, object)
    def _accept_seed(self, frame: object, mask: object) -> None:
        self._seed_inflight = False
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
        self._seed_inflight = False
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
                self._pending_matte = None
                self._prepared_composites.clear()
                self._composition_inflight = False
                self._presentation_drops = 0
                self._active_device = event.capabilities.device_type
                self._internal_size = event.selected_internal_size
                self._invalid_matte_count = 0
                if self._harmonization_requested:
                    self._prepare_harmonizer()
                device = event.capabilities.device_type.upper()
                self._set_state("live", f"Live · MatAnyone 2 · {device}")
            elif isinstance(event, CompletedMatte):
                self._queue_composition(event)
            elif isinstance(event, EngineFailure):
                self._pause_tracking(
                    f"Matting stopped: {event.message}. Re-select the person to retry.",
                )

    @Slot()
    def _present_pending_matte(self) -> None:
        if not self._prepared_composites:
            return
        prepared = self._prepared_composites.popleft()
        composite = self._surface.present_matte(prepared)
        if composite is not None:
            self._handle_composite(prepared.completed, composite)

    def _queue_composition(self, completed: CompletedMatte) -> None:
        if self._composition_inflight:
            if self._pending_matte is not None:
                self._presentation_drops += 1
            self._pending_matte = completed
            return
        self._start_composition(completed)

    def _start_composition(self, completed: CompletedMatte) -> None:
        self._composition_inflight = True
        revision = self._session_revision

        def compose() -> None:
            try:
                prepared = self._surface.prepare_matte(completed)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._composite_failed.emit(revision, str(error)[:240])
            else:
                self._composite_prepared.emit(revision, prepared)

        threading.Thread(target=compose, name="live-compositor", daemon=True).start()

    @Slot(int, object)
    def _accept_prepared_composite(self, revision: int, prepared: object) -> None:
        if revision != self._session_revision or not isinstance(prepared, PreparedComposite):
            return
        self._composition_inflight = False
        if len(self._prepared_composites) == PRESENTATION_BUFFER_SIZE:
            self._presentation_drops += 1
        self._prepared_composites.append(prepared)
        pending = self._pending_matte
        self._pending_matte = None
        if pending is not None:
            self._start_composition(pending)
        if (
            not self._presentation_timer.isActive()
            and len(self._prepared_composites) == PRESENTATION_BUFFER_SIZE
        ):
            self._presentation_timer.start()

    @Slot(int, str)
    def _reject_prepared_composite(self, revision: int, message: str) -> None:
        if revision != self._session_revision:
            return
        self._composition_inflight = False
        self._pause_tracking(f"Compositing stopped: {message}. Re-select the person to retry.")

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
            + self._presentation_drops,
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
        self._pending_matte = None
        self._prepared_composites.clear()
        self._composition_inflight = False
        self._session_revision += 1
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._selector.reset()
        self._seed_frame = None
        self._seed_mask = None
        self._seed_inflight = False
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
