"""Tests for MatAnyone 2 quality release gates."""

from __future__ import annotations

import numpy as np

from better_backgrounds.matting_quality import MattingQuality, measure_quality, quality_gate


def test_quality_gate_requires_boundary_improvement_without_regressions() -> None:
    """Pass only when detail improves and MAD/temporal quality remain bounded."""
    baseline = MattingQuality(mad=10.0, gradient_error=20.0, temporal_error=5.0)
    candidate = MattingQuality(mad=10.4, gradient_error=15.5, temporal_error=5.1)

    gate = quality_gate(candidate, baseline)

    assert gate.boundary_improvement == 0.225
    assert gate.passed


def test_quality_gate_rejects_good_edges_with_unstable_time_series() -> None:
    """Prevent a sharper but visibly flickering model from replacing the baseline."""
    baseline = MattingQuality(mad=10.0, gradient_error=20.0, temporal_error=5.0)
    candidate = MattingQuality(mad=10.0, gradient_error=15.0, temporal_error=6.0)

    assert not quality_gate(candidate, baseline).passed


def test_identical_matte_sequence_has_zero_error() -> None:
    """Keep metric implementations numerically grounded."""
    mattes = np.zeros((3, 8, 8), dtype=np.uint8)
    mattes[:, 2:6, 2:6] = 255

    quality = measure_quality(mattes, mattes)

    assert quality == MattingQuality(mad=0.0, gradient_error=0.0, temporal_error=0.0)
