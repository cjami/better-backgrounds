"""Native Qt surface for exact-frame webcam composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from better_backgrounds.desktop.camera.capture import qimage_to_rgb
from better_backgrounds.harmonization.runtime import HarmonizerAppearanceHarmonizer
from better_backgrounds.matting.compositor import (
    LiveComposite,
    background_has_content,
    compose_live_frame,
)
from better_backgrounds.matting.engine import CompletedMatte
from better_backgrounds.matting.refinement import decontaminate_foreground

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from better_backgrounds.harmonization import HarmonizationSettings
    from better_backgrounds.matting.contracts import LiveDiagnostics

ALPHA_MIDPOINT = 128


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
        self._harmonization_background: NDArray[np.uint8] | None = None
        self._resized_harmonization_backgrounds: dict[tuple[int, int], NDArray[np.uint8]] = {}
        self._harmonization_revision = 0
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

    def set_background(
        self,
        image: QImage,
        *,
        update_harmonization_reference: bool = True,
    ) -> bool:
        """Atomically replace the immutable room snapshot."""
        if image.isNull():
            return False
        background = qimage_to_rgb(image)
        if not background_has_content(background):
            return False
        self._background = background
        self._resized_backgrounds.clear()
        self._background_revision += 1
        if update_harmonization_reference:
            self._set_harmonization_background(background)
        self._recompose()
        return True

    def set_harmonization_reference(self, image: QImage) -> bool:
        """Replace the sharp room evidence used only for subject appearance."""
        if image.isNull():
            return False
        background = qimage_to_rgb(image)
        if not background_has_content(background):
            return False
        self._set_harmonization_background(background)
        self._recompose()
        return True

    def clear_background(self) -> None:
        """Discard room-scoped pixels and appearance evidence together."""
        self._background = None
        self._resized_backgrounds.clear()
        self._background_revision += 1
        self._harmonization_background = None
        self._resized_harmonization_backgrounds.clear()
        self._harmonization_revision += 1
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
        harmonization_background = self._harmonization_background
        revision = self._background_revision
        if background is None:
            background = np.zeros_like(source)
        else:
            background = self._resize_background(
                background,
                source,
                self._resized_backgrounds,
            )
        if harmonization_background is None:
            harmonization_background = background
        else:
            harmonization_background = self._resize_background(
                harmonization_background,
                source,
                self._resized_harmonization_backgrounds,
            )
        compare_mode = self._mode == "compare"
        foreground = decontaminate_foreground(source, alpha)
        composite = compose_live_frame(
            completed.packet,
            completed.result,
            source,
            alpha,
            background,
            revision=revision,
            foreground=foreground,
            harmonizer=self._harmonizer,
            harmonization_background=harmonization_background,
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
        if diagnostics.background_refresh_ms > 0:
            self._diagnostics += f" · {diagnostics.background_refresh_ms:.0f} ms background refresh"
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

    def _set_harmonization_background(self, background: NDArray[np.uint8]) -> None:
        self._harmonization_background = background
        self._resized_harmonization_backgrounds.clear()
        self._harmonization_revision += 1
        self._harmonizer.set_room(background, revision=self._harmonization_revision)

    @staticmethod
    def _resize_background(
        background: NDArray[np.uint8],
        source: NDArray[np.uint8],
        cache: dict[tuple[int, int], NDArray[np.uint8]],
    ) -> NDArray[np.uint8]:
        if background.shape == source.shape:
            return background
        shape = source.shape[:2]
        resized = cache.get(shape)
        if resized is None:
            resized = cast(
                "NDArray[np.uint8]",
                cv2.resize(
                    background,
                    (source.shape[1], source.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                ),
            )
            cache[shape] = resized
        return resized

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
