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
            return self._remember(alpha, captured_at=captured_at)

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
        remembered = alpha.copy()
        self._previous = remembered
        self._captured_at = captured_at
        return remembered


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
    small_source = cv2.resize(source, estimate_size, interpolation=cv2.INTER_AREA).astype(
        np.float32,
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
    estimate = np.rint(np.clip(estimate, 0.0, 255.0)).astype(np.uint8)
    estimate = cv2.resize(estimate, (width, height), interpolation=cv2.INTER_LINEAR)

    observed = source[uncertain].astype(np.float32)
    edge_alpha = alpha[uncertain].astype(np.float32)[:, None] / 255.0
    recovered = (observed - (1.0 - edge_alpha) * estimate[uncertain]) / np.maximum(
        edge_alpha, MINIMUM_RECOVERY_ALPHA
    )
    recovered = np.clip(recovered, 0.0, 255.0)
    cleaned = source.copy()
    cleaned[uncertain] = np.rint(
        observed + (recovered - observed) * DECONTAMINATION_STRENGTH,
    ).astype(np.uint8)
    return cleaned
