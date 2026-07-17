"""Video probing and deterministic capture-quality tests."""

from pathlib import Path

import numpy as np
import pytest

from better_backgrounds.video_analysis import (
    CaptureConstraints,
    FrameMetrics,
    VideoProbe,
    analyse_metrics,
    parse_ffprobe,
    select_frames,
    validate_probe,
)

ROTATED_RIGHT_ANGLE = 270
MAXIMUM_SELECTED = 100
EARLY_TIMESTAMP = 2
LATE_TIMESTAMP = 117
CANDIDATE_COUNT = 90
SELECTED_COUNT = 80


def probe(**changes: object) -> VideoProbe:
    """Build one otherwise supported video description."""
    values: dict[str, object] = {
        "path": Path("room video.mp4"),
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
        "duration_seconds": 22.0,
        "width": 1920,
        "height": 1080,
        "frame_rate": 30.0,
        "rotation_degrees": 0,
    }
    values.update(changes)
    return VideoProbe(**values)


def test_ffprobe_parsing_normalizes_rotation_and_fractional_rate() -> None:
    """Keep metadata normalization independent from the ffprobe executable."""
    result = parse_ffprobe(
        Path("部屋 video.mov"),
        {
            "format": {"format_name": "mov,mp4", "duration": "20.5"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "30000/1001",
                    "side_data_list": [{"rotation": -90}],
                },
            ],
        },
    )

    assert result.path.name == "部屋 video.mov"
    assert result.rotation_degrees == ROTATED_RIGHT_ANGLE
    assert (result.display_width, result.display_height) == (1920, 1080)
    assert result.frame_rate == pytest.approx(29.97, rel=1e-3)


def test_ffprobe_rejects_inconsistent_video_orientations() -> None:
    """Do not mix camera streams whose normalized dimensions would change."""
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30/1",
    }
    payload = {
        "format": {"format_name": "mp4", "duration": "20"},
        "streams": [stream, {**stream, "side_data_list": [{"rotation": 90}]}],
    }

    with pytest.raises(ValueError, match="inconsistent orientations"):
        parse_ffprobe(Path("room.mp4"), payload)


@pytest.mark.parametrize(
    ("changed", "code"),
    [
        ({"duration_seconds": 4.0}, "duration_too_short"),
        ({"duration_seconds": 61.0}, "duration_too_long"),
        ({"codec": "vp9"}, "unsupported_codec"),
        ({"width": 640, "height": 360}, "resolution_too_low"),
    ],
)
def test_unsupported_capture_has_precise_recovery(changed: dict[str, object], code: str) -> None:
    """Reject unsuitable media before frame extraction."""
    issues = validate_probe(probe(**changed), CaptureConstraints())

    assert issues[0].code == code
    assert issues[0].recovery_action


@pytest.mark.parametrize(
    ("width", "height", "rotation_degrees"),
    [
        (1920, 1080, 0),
        (1080, 1920, 0),
        (1920, 1080, 90),
    ],
)
def test_supported_resolution_is_independent_of_orientation(
    width: int,
    height: int,
    rotation_degrees: int,
) -> None:
    """Accept 1080p captures in landscape, portrait, or metadata rotation."""
    issues = validate_probe(
        probe(width=width, height=height, rotation_degrees=rotation_degrees),
        CaptureConstraints(),
    )

    assert not [issue for issue in issues if issue.code.startswith("resolution_")]


def test_frame_selection_is_bounded_and_temporally_distributed() -> None:
    """Prefer useful frames without clustering the selection in one moment."""
    metrics = [
        FrameMetrics(
            index=index,
            timestamp_seconds=float(index),
            sharpness=80.0 + index,
            clipped_fraction=0.01,
            motion=0.2,
            similarity=0.7,
        )
        for index in range(120)
    ]

    selected = select_frames(metrics, minimum=60, maximum=100)

    assert len(selected) == MAXIMUM_SELECTED
    assert selected[0].timestamp_seconds < EARLY_TIMESTAMP
    assert selected[-1].timestamp_seconds > LATE_TIMESTAMP
    assert [item.timestamp_seconds for item in selected] == sorted(
        item.timestamp_seconds for item in selected
    )


def test_capture_analysis_reports_explainable_signals() -> None:
    """Summarize measured inputs instead of claiming learned success."""
    metrics = [
        FrameMetrics(
            index=index,
            timestamp_seconds=index / 3,
            sharpness=100.0,
            clipped_fraction=0.02,
            motion=0.25,
            similarity=0.75,
        )
        for index in range(90)
    ]
    diagnostics = analyse_metrics(probe(), metrics, metrics[:80])

    assert diagnostics.suitable
    assert diagnostics.candidate_frames == CANDIDATE_COUNT
    assert diagnostics.selected_frames == SELECTED_COUNT
    assert diagnostics.overlap_proxy == pytest.approx(0.75)
    assert np.isfinite(diagnostics.movement)
