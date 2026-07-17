"""Tests for reproducible MatAnyone 2 performance reports."""

from __future__ import annotations

from better_backgrounds.matting_benchmark import benchmark_measurement


def test_measurement_reports_p95_throughput_and_gate() -> None:
    """Record the release metrics from synchronized per-frame timings."""
    measurement = benchmark_measurement(432, [20.0, 30.0, 40.0, 50.0], budget_ms=66.0)

    assert measurement.internal_size == 432
    assert measurement.frames == 4
    assert measurement.p50_ms == 35.0
    assert measurement.p95_ms == 48.5
    assert measurement.mattes_per_second == 28.57
    assert measurement.passed


def test_measurement_fails_both_latency_and_minimum_rate_gate() -> None:
    """Do not describe a slow functional CPU result as real time."""
    measurement = benchmark_measurement(360, [100.0, 100.0], budget_ms=66.0)

    assert not measurement.passed
