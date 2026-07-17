"""Native Qt webcam, MatAnyone 2 session, and exact-frame compositor."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QStackedLayout, QWidget

from better_backgrounds.compositor import LiveComposite, compose_live_frame
from better_backgrounds.desktop.camera_capture import QtCameraCapture, qimage_to_rgb
from better_backgrounds.live_matting import LiveDiagnostics, MattingConfig
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

    from better_backgrounds.scene import ManagedSceneResolver, SceneReference, Viewpoint

EngineFactory = Callable[[], ProcessMattingEngine]
BackgroundFactory = Callable[[], QWidget]
ALPHA_MIDPOINT = 128
MINIMUM_MATTE_OCCUPANCY = 0.01
MAXIMUM_MATTE_OCCUPANCY = 0.95
LOST_MATTE_LIMIT = 15


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


class NativeCompositeSurface(QWidget):
    """Paint the current exact-frame composite and comparison wipe."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a GPU-independent presentation surface."""
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAccessibleName("Native MatAnyone 2 webcam composite")
        self._background: NDArray[np.uint8] | None = None
        self._background_revision = 0
        self._raw_source: NDArray[np.uint8] | None = None
        self._packet = None
        self._matte = None
        self._alpha: NDArray[np.uint8] | None = None
        self._source_image = QImage()
        self._composite_image = QImage()
        self._seed_image = QImage()
        self._mask_image = QImage()
        self._mask_label = ""
        self._mode = "show"
        self._wipe = 52
        self._mirrored = True
        self._diagnostics = "Waiting for camera"
        self._last_composite: LiveComposite | None = None

    @property
    def last_composite(self) -> LiveComposite | None:
        """Return the most recently painted exact-frame evidence."""
        return self._last_composite

    def set_background(self, image: QImage) -> None:
        """Atomically replace the immutable room snapshot."""
        if image.isNull():
            return
        self._background = qimage_to_rgb(image)
        self._background_revision += 1
        self._recompose()

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
        self._mask_image = rgb_to_qimage(np.repeat(display_mask[..., None], 3, axis=2))
        self._mask_label = "PERSON FOUND"
        self.update()

    def apply_matte(self, completed: CompletedMatte) -> LiveComposite:
        """Compose the exact frame/matte pair returned by the scheduler."""
        self._packet = completed.packet
        self._matte = completed.result
        self._raw_source = completed.source
        self._alpha = completed.alpha
        self._seed_image = QImage()
        self._mask_label = "LIVE MATTE"
        composite = self._recompose()
        if composite is None:
            msg = "compositor did not produce an image"
            raise RuntimeError(msg)
        return composite

    def set_presentation(self, mode: str, wipe: int) -> None:
        """Switch presentation without duplicating camera or inference work."""
        if mode in {"show", "compare"}:
            self._mode = mode
        self._wipe = min(100, max(0, wipe))
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
            f"{diagnostics.mask_fps:.0f} mattes/s · {diagnostics.mask_age_ms:.0f} ms · "
            f"{diagnostics.worker_time_ms:.1f} ms worker · "
            f"{diagnostics.processing_width}x{diagnostics.processing_height} · "
            f"{diagnostics.device_type.upper()} · {diagnostics.dropped_frames} dropped"
        )
        self.update()

    def clear_matte(self) -> None:
        """Discard temporal output before selecting a new target."""
        self._packet = None
        self._matte = None
        self._alpha = None
        self._composite_image = QImage()
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
            and not self._source_image.isNull()
            and not self._composite_image.isNull()
        ):
            split = round(self.width() * self._wipe / 100)
            painter.save()
            painter.setClipRect(QRect(0, 0, split, self.height()))
            self._draw_cover(painter, self._source_image)
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
        source = np.flip(self._raw_source, axis=1).copy() if self._mirrored else self._raw_source
        alpha = np.flip(self._alpha, axis=1).copy() if self._mirrored else self._alpha
        background = self._background
        if background is None:
            background = np.zeros_like(source)
        composite = compose_live_frame(
            self._packet,
            self._matte,
            source,
            alpha,
            background,
            revision=self._background_revision,
        )
        self._source_image = rgb_to_qimage(source)
        self._composite_image = rgb_to_qimage(composite.image)
        self._mask_image = rgb_to_qimage(np.repeat(alpha[..., None], 3, axis=2))
        self._last_composite = composite
        self.update()
        return composite

    def _draw_cover(self, painter: QPainter, image: QImage) -> None:
        scaled = image.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        source = QRect(
            max(0, (scaled.width() - self.width()) // 2),
            max(0, (scaled.height() - self.height()) // 2),
            self.width(),
            self.height(),
        )
        painter.drawImage(self.rect(), scaled, source)


class NativeLivePreview(QWidget):
    """Own one Qt camera, seed operation, worker, and room snapshot cache."""

    camera_state_changed = Signal(str, str)
    diagnostics_changed = Signal(object)
    comparison_frame = Signal(object)
    _seed_generated = Signal(object, object)
    _seed_failed = Signal(str)

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
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(5)
        self._poll_timer.timeout.connect(self._poll_engine)
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
        self._mask_count = 0
        self._rate_started = time.monotonic()
        self._seed_generated.connect(self._accept_seed)
        self._seed_failed.connect(self._reject_seed)

    def start_camera(self, device_id: str, *, mirrored: bool) -> None:
        """Open the selected Qt camera and begin one-shot target selection."""
        self.stop_camera()
        self._mirrored = mirrored
        self._surface.set_mirroring(mirrored=mirrored)
        self._selector.reset()
        self._poll_timer.start()
        self._set_state("seeding", "Hold still while the person seed is captured…")
        if not self._camera_capture.start(device_id):
            self._poll_timer.stop()

    def stop_camera(self) -> None:
        """Release camera, model worker, and all queued frame ownership."""
        self._poll_timer.stop()
        self._camera_capture.stop()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._selector.reset()
        self._latest_frame = None
        self._seed_frame = None
        self._seed_mask = None
        self._seed_inflight = False
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
                MattingConfig(internal_size=540, warmup_iterations=10, calibrate=True),
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

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Refresh the immutable room snapshot after viewpoint changes."""
        setter = getattr(self._background_renderer, "set_viewpoint", None)
        if callable(setter):
            setter(viewpoint)
            self._background_timer.start(120)

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

    @Slot(int, int)
    def _scene_progressed(self, loaded: int, total: int) -> None:
        if loaded == total:
            self._background_timer.start(120)

    @Slot()
    def _capture_background(self) -> None:
        pixmap = self._background_renderer.grab()
        if not pixmap.isNull():
            self._surface.set_background(pixmap.toImage())

    @Slot(object, float)
    def _camera_frame_captured(self, frame: object, captured_at: float) -> None:
        if not isinstance(frame, np.ndarray):
            return
        source = cast("NDArray[np.uint8]", frame)
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
                self._mask_count = 0
                self._rate_started = time.monotonic()
                self._active_device = event.capabilities.device_type
                self._internal_size = event.selected_internal_size
                self._invalid_matte_count = 0
                device = event.capabilities.device_type.upper()
                self._set_state("live", f"Live · MatAnyone 2 · {device}")
            elif isinstance(event, CompletedMatte):
                self._handle_matte(event)
            elif isinstance(event, EngineFailure):
                self._pause_tracking(
                    f"Matting stopped: {event.message}. Re-select the person to retry.",
                )

    def _handle_matte(self, completed: CompletedMatte) -> None:
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
        self._surface.apply_matte(completed)
        self._mask_count += 1
        now = time.monotonic()
        elapsed = max(0.001, now - self._rate_started)
        rate = self._mask_count / elapsed
        engine = self._engine
        diagnostics = LiveDiagnostics(
            display_fps=rate,
            mask_fps=rate,
            mask_age_ms=max(0.0, now * 1000.0 - completed.packet.captured_at),
            dropped_frames=0 if engine is None else engine.dropped_frames,
            worker_time_ms=completed.result.inference_ms,
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
            self.comparison_frame.emit(self._surface.grab())

    def _begin_reseed(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("seeding", message)

    def _pause_tracking(self, message: str) -> None:
        self._reset_tracking()
        self._set_state("lost", message)

    def _reset_tracking(self) -> None:
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
