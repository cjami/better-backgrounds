"""Numeric quality measurements and release gate for video alpha mattes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray

MATTE_SEQUENCE_DIMENSIONS = 3
MINIMUM_QUALITY_FRAMES = 2


class MattingQuality(BaseModel):
    """Record foreground, boundary, and temporal errors against alpha truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mad: float = Field(ge=0, allow_inf_nan=False)
    gradient_error: float = Field(ge=0, allow_inf_nan=False)
    temporal_error: float = Field(ge=0, allow_inf_nan=False)


class MattingQualityGate(BaseModel):
    """Explain every condition in the MatAnyone 2 quality gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    boundary_improvement: float = Field(allow_inf_nan=False)
    mad_regression: float = Field(allow_inf_nan=False)
    temporal_regression: float = Field(allow_inf_nan=False)
    passed: bool


def measure_quality(
    predicted: NDArray[np.uint8],
    truth: NDArray[np.uint8],
) -> MattingQuality:
    """Measure MAD, Sobel gradient error, and temporal-derivative SSD."""
    if predicted.dtype != np.uint8 or truth.dtype != np.uint8 or predicted.shape != truth.shape:
        msg = "predicted and truth mattes must be matching uint8 TxHxW arrays"
        raise ValueError(msg)
    if predicted.ndim != MATTE_SEQUENCE_DIMENSIONS or predicted.shape[0] < MINIMUM_QUALITY_FRAMES:
        msg = "quality measurement requires at least two matte frames"
        raise ValueError(msg)
    predicted_float = predicted.astype(np.float32) / 255.0
    truth_float = truth.astype(np.float32) / 255.0
    mad = float(np.mean(np.abs(predicted_float - truth_float)) * 1000.0)
    predicted_gradient = _gradient_sequence(predicted_float)
    truth_gradient = _gradient_sequence(truth_float)
    gradient_error = float(np.mean(np.abs(predicted_gradient - truth_gradient)) * 1000.0)
    predicted_delta = np.diff(predicted_float, axis=0)
    truth_delta = np.diff(truth_float, axis=0)
    temporal_error = float(np.sqrt(np.mean(np.square(predicted_delta - truth_delta))) * 1000.0)
    return MattingQuality(
        mad=round(mad, 6),
        gradient_error=round(gradient_error, 6),
        temporal_error=round(temporal_error, 6),
    )


def quality_gate(
    candidate: MattingQuality,
    baseline: MattingQuality,
    *,
    minimum_boundary_improvement: float = 0.20,
    maximum_regression: float = 0.05,
) -> MattingQualityGate:
    """Require materially better edges without foreground or temporal regressions."""
    boundary = _relative_improvement(candidate.gradient_error, baseline.gradient_error)
    mad_regression = _relative_regression(candidate.mad, baseline.mad)
    temporal_regression = _relative_regression(
        candidate.temporal_error,
        baseline.temporal_error,
    )
    return MattingQualityGate(
        boundary_improvement=round(boundary, 6),
        mad_regression=round(mad_regression, 6),
        temporal_regression=round(temporal_regression, 6),
        passed=(
            boundary >= minimum_boundary_improvement
            and mad_regression <= maximum_regression
            and temporal_regression <= maximum_regression
        ),
    )


def _gradient_sequence(mattes: NDArray[np.float32]) -> NDArray[np.float32]:
    gradients = []
    for matte in mattes:
        horizontal = cv2.Sobel(matte, cv2.CV_32F, 1, 0, ksize=3)
        vertical = cv2.Sobel(matte, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(horizontal, vertical))
    return np.stack(gradients)


def _relative_improvement(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else -candidate
    return (baseline - candidate) / baseline


def _relative_regression(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else candidate
    return (candidate - baseline) / baseline
