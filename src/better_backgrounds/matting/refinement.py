"""Low-cost temporal and colour refinement for live alpha boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TARGET_FRAME_INTERVAL_MS = 1_000.0 / 30.0
MAXIMUM_STABILIZATION_GAP_MS = 100.0
BOUNDARY_ALPHA_MINIMUM = 8
BOUNDARY_ALPHA_MAXIMUM = 247
MAXIMUM_STABILIZED_DELTA = 64
TEMPORAL_RETENTION = 0.35
BACKGROUND_ESTIMATE_SCALE = 0.25
BACKGROUND_ESTIMATE_KERNEL = (7, 7)
DECONTAMINATION_STRENGTH = 0.85
MINIMUM_RECOVERY_ALPHA = 0.12
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
ALPHA_DIMENSIONS = 2
SRGB_DECODE_THRESHOLD = 0.04045
SRGB_ENCODE_THRESHOLD = 0.0031308
SRGB_DECODE_DIVISOR = 12.92
SRGB_ENCODE_MULTIPLIER = 12.92
SRGB_OFFSET = 0.055
SRGB_SCALE = 1.055
SRGB_GAMMA = 2.4
LINEAR_LOOKUP_MAXIMUM = 4_096
LIGHT_WRAP_STRENGTH = 0.06
LIGHT_WRAP_RADIUS_DIVISOR = 360
MAXIMUM_LIGHT_WRAP_RADIUS = 4

_SRGB_VALUES = np.linspace(0.0, 1.0, 256, dtype=np.float32)
SRGB_TO_LINEAR_LOOKUP = np.where(
    _SRGB_VALUES <= SRGB_DECODE_THRESHOLD,
    _SRGB_VALUES / SRGB_DECODE_DIVISOR,
    ((_SRGB_VALUES + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA,
).astype(np.float32)
_LINEAR_VALUES = np.linspace(0.0, 1.0, LINEAR_LOOKUP_MAXIMUM + 1, dtype=np.float32)
LINEAR_TO_SRGB_LOOKUP = np.rint(
    np.where(
        _LINEAR_VALUES <= SRGB_ENCODE_THRESHOLD,
        _LINEAR_VALUES * SRGB_ENCODE_MULTIPLIER,
        SRGB_SCALE * np.power(_LINEAR_VALUES, 1.0 / SRGB_GAMMA) - SRGB_OFFSET,
    )
    * 255.0,
).astype(np.uint8)


@dataclass(slots=True)
class TemporalAlphaStabilizer:
    """Damp small uncertain-edge changes while releasing real movement."""

    _previous: NDArray[np.uint8] | None = field(default=None, init=False, repr=False)
    _captured_at: float | None = field(default=None, init=False, repr=False)

    def apply(
        self,
        alpha: NDArray[np.uint8],
        *,
        captured_at: float,
    ) -> NDArray[np.uint8]:
        """Return a stabilized copy associated with the current source frame."""
        if alpha.dtype != np.uint8 or alpha.ndim != ALPHA_DIMENSIONS:
            msg = "alpha must be a two-dimensional uint8 array"
            raise ValueError(msg)
        if not math.isfinite(captured_at) or captured_at < 0:
            msg = "capture timestamp must be finite and non-negative"
            raise ValueError(msg)
        previous = self._previous
        previous_at = self._captured_at
        elapsed = None if previous_at is None else captured_at - previous_at
        if (
            previous is None
            or previous.shape != alpha.shape
            or elapsed is None
            or elapsed <= 0
            or elapsed > MAXIMUM_STABILIZATION_GAP_MS
        ):
            return self._remember(alpha.copy(), captured_at=captured_at)

        difference = cv2.absdiff(alpha, previous)
        boundary = cv2.bitwise_or(
            cv2.inRange(alpha, BOUNDARY_ALPHA_MINIMUM, BOUNDARY_ALPHA_MAXIMUM),
            cv2.inRange(previous, BOUNDARY_ALPHA_MINIMUM, BOUNDARY_ALPHA_MAXIMUM),
        )
        _threshold, stable_delta = cv2.threshold(
            difference,
            MAXIMUM_STABILIZED_DELTA,
            255,
            cv2.THRESH_BINARY_INV,
        )
        stable_boundary = cv2.bitwise_and(
            boundary,
            stable_delta,
        )
        elapsed_frames = max(elapsed / TARGET_FRAME_INTERVAL_MS, 1.0)
        retention = TEMPORAL_RETENTION**elapsed_frames
        blended = cv2.addWeighted(alpha, 1.0 - retention, previous, retention, 0.0)
        stabilized = alpha.copy()
        cv2.copyTo(blended, stable_boundary, stabilized)
        return self._remember(stabilized, captured_at=captured_at)

    def reset(self) -> None:
        """Discard matte history at a camera or tracking boundary."""
        self._previous = None
        self._captured_at = None

    def _remember(
        self,
        alpha: NDArray[np.uint8],
        *,
        captured_at: float,
    ) -> NDArray[np.uint8]:
        """Retain an owned matte as history; callers pass a private array."""
        self._previous = alpha
        self._captured_at = captured_at
        return alpha


def decontaminate_foreground(
    source: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Remove estimated original-background colour from soft foreground edges."""
    if source.dtype != np.uint8 or source.ndim != RGB_DIMENSIONS or source.shape[2] != RGB_CHANNELS:
        msg = "source must be uint8 RGB"
        raise ValueError(msg)
    if alpha.dtype != np.uint8 or alpha.shape != source.shape[:2]:
        msg = "alpha must be uint8 and match source"
        raise ValueError(msg)
    uncertain = (alpha >= BOUNDARY_ALPHA_MINIMUM) & (alpha <= BOUNDARY_ALPHA_MAXIMUM)
    if not np.any(uncertain):
        return source

    height, width = alpha.shape
    estimate_size = (
        max(1, round(width * BACKGROUND_ESTIMATE_SCALE)),
        max(1, round(height * BACKGROUND_ESTIMATE_SCALE)),
    )
    linear_source = decode_srgb(source)
    small_source = cv2.resize(
        linear_source,
        estimate_size,
        interpolation=cv2.INTER_AREA,
    )
    small_alpha = (
        cv2.resize(alpha, estimate_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    )
    background_weight = np.power(1.0 - small_alpha, 4.0)
    weighted_source = small_source * background_weight[..., None]
    numerator = cv2.boxFilter(
        weighted_source,
        cv2.CV_32F,
        BACKGROUND_ESTIMATE_KERNEL,
        normalize=False,
    )
    denominator = cv2.boxFilter(
        background_weight,
        cv2.CV_32F,
        BACKGROUND_ESTIMATE_KERNEL,
        normalize=False,
    )
    estimate = numerator / np.maximum(denominator[..., None], 1e-4)
    estimate = cv2.resize(
        np.clip(estimate, 0.0, 1.0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    observed = linear_source[uncertain]
    edge_alpha = alpha[uncertain].astype(np.float32)[:, None] / 255.0
    recovered = (observed - (1.0 - edge_alpha) * estimate[uncertain]) / np.maximum(
        edge_alpha, MINIMUM_RECOVERY_ALPHA
    )
    cleaned = linear_source.copy()
    cleaned[uncertain] = np.clip(
        observed + (recovered - observed) * DECONTAMINATION_STRENGTH,
        0.0,
        1.0,
    )
    return encode_srgb(cleaned)


def compose_linear_light(
    foreground: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Composite and lightly wrap one foreground in linear light."""
    _validate_rgb(foreground, name="foreground")
    _validate_rgb(background, name="background")
    if foreground.shape != background.shape:
        msg = "foreground and background must match"
        raise ValueError(msg)
    _validate_alpha(alpha, foreground)
    foreground_linear = decode_srgb(foreground)
    background_linear = decode_srgb(background)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    composite = foreground_linear * weight + background_linear * (1.0 - weight)
    return encode_srgb(_apply_light_wrap(composite, weight, background_linear))


def add_light_wrap(
    composite: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Add restrained destination light within the soft matte boundary."""
    _validate_rgb(composite, name="composite")
    _validate_rgb(background, name="background")
    if composite.shape != background.shape:
        msg = "composite and background must match"
        raise ValueError(msg)
    _validate_alpha(alpha, composite)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    result = _apply_light_wrap(
        decode_srgb(composite),
        weight,
        decode_srgb(background),
    )
    return encode_srgb(result)


def decode_srgb(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Decode uint8 sRGB pixels into normalized linear light."""
    return SRGB_TO_LINEAR_LOOKUP[image]


def encode_srgb(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Encode normalized linear-light pixels through a bounded lookup."""
    indices = np.rint(
        np.clip(image, 0.0, 1.0) * LINEAR_LOOKUP_MAXIMUM,
    ).astype(np.uint16)
    return LINEAR_TO_SRGB_LOOKUP[indices]


def light_wrap_radius(height: int, width: int) -> int:
    """Scale the destination blur conservatively with output resolution."""
    return min(
        MAXIMUM_LIGHT_WRAP_RADIUS,
        max(1, round(min(height, width) / LIGHT_WRAP_RADIUS_DIVISOR)),
    )


def _apply_light_wrap(
    composite: NDArray[np.float32],
    alpha: NDArray[np.float32],
    background: NDArray[np.float32],
) -> NDArray[np.float32]:
    radius = light_wrap_radius(*alpha.shape[:2])
    kernel_size = radius * 2 + 1
    wrapped_light = cv2.boxFilter(
        background,
        cv2.CV_32F,
        (kernel_size, kernel_size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    boundary = 4.0 * alpha * (1.0 - alpha) * LIGHT_WRAP_STRENGTH
    screened = composite + (1.0 - composite) * wrapped_light
    return np.clip(composite + (screened - composite) * boundary, 0.0, 1.0)


def _validate_rgb(image: NDArray[np.uint8], *, name: str) -> None:
    if image.dtype != np.uint8 or image.ndim != RGB_DIMENSIONS or image.shape[2] != RGB_CHANNELS:
        msg = f"{name} must be uint8 RGB"
        raise ValueError(msg)


def _validate_alpha(alpha: NDArray[np.uint8], image: NDArray[np.uint8]) -> None:
    if alpha.dtype != np.uint8 or alpha.shape != image.shape[:2]:
        msg = "alpha must be uint8 and match the RGB image"
        raise ValueError(msg)
