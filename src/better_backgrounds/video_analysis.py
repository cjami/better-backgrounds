"""Deterministic video probing and explainable capture diagnostics."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence  # noqa: TC003
from itertools import pairwise
from pathlib import Path  # noqa: TC003
from typing import Any, Literal, Self, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MINIMUM_SELECTED_FRAMES = 60
DARK_CLIP_LEVEL = 5
BRIGHT_CLIP_LEVEL = 250
LOW_MOVEMENT_WARNING = 0.03
HIGH_MOVEMENT_WARNING = 0.65
LOW_OVERLAP_WARNING = 0.45
HIGH_CLIPPING_WARNING = 0.2
MOVING_OBJECT_WARNING = 0.15


class AnalysisModel(BaseModel):
    """Reject unknown data at media and persisted-analysis boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VideoProbe(AnalysisModel):
    """Normalized metadata for the primary video stream."""

    path: Path
    container: str = Field(min_length=1)
    codec: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0, allow_inf_nan=False)
    rotation_degrees: Literal[0, 90, 180, 270] = 0

    @property
    def display_width(self) -> int:
        """Return the width after rotation metadata is normalized."""
        return self.height if self.rotation_degrees in {90, 270} else self.width

    @property
    def display_height(self) -> int:
        """Return the height after rotation metadata is normalized."""
        return self.width if self.rotation_degrees in {90, 270} else self.height

    @property
    def display_long_edge(self) -> int:
        """Return the longer displayed edge regardless of orientation."""
        return max(self.display_width, self.display_height)

    @property
    def display_short_edge(self) -> int:
        """Return the shorter displayed edge regardless of orientation."""
        return min(self.display_width, self.display_height)


class CaptureConstraints(AnalysisModel):
    """Bound the intentionally narrow Phase 4 input envelope."""

    minimum_duration: float = 10.0
    maximum_duration: float = 45.0
    minimum_long_edge: int = 1280
    minimum_short_edge: int = 720
    maximum_long_edge: int = 3840
    maximum_short_edge: int = 2160
    codecs: frozenset[str] = frozenset({"h264", "hevc"})
    containers: frozenset[str] = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})


class CaptureIssue(AnalysisModel):
    """Explain one hard validation failure and its recovery."""

    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)
    recovery_action: str = Field(min_length=1, max_length=500)


class FrameMetrics(AnalysisModel):
    """Store normalized quality evidence for one timestamped candidate."""

    index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0, allow_inf_nan=False)
    sharpness: float = Field(ge=0, allow_inf_nan=False)
    clipped_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    motion: float = Field(ge=0, le=1, allow_inf_nan=False)
    similarity: float = Field(ge=0, le=1, allow_inf_nan=False)
    moving_fraction: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)

    @property
    def quality_score(self) -> float:
        """Combine transparent heuristics for selection inside a time bucket."""
        sharpness = min(self.sharpness / 150.0, 1.0)
        exposure = 1.0 - min(self.clipped_fraction * 4.0, 1.0)
        useful_motion = 1.0 - min(abs(self.motion - 0.25) / 0.75, 1.0)
        useful_overlap = 1.0 - min(abs(self.similarity - 0.72) / 0.72, 1.0)
        return 0.35 * sharpness + 0.25 * exposure + 0.2 * useful_motion + 0.2 * useful_overlap


class CaptureDiagnostics(AnalysisModel):
    """Present measured suitability evidence without promising reconstruction."""

    probe: VideoProbe
    suitable: bool
    candidate_frames: int = Field(ge=0)
    selected_frames: int = Field(ge=0)
    movement: float = Field(ge=0, le=1, allow_inf_nan=False)
    overlap_proxy: float = Field(ge=0, le=1, allow_inf_nan=False)
    exposure_stability: float = Field(ge=0, le=1, allow_inf_nan=False)
    moving_object_proxy: float = Field(ge=0, le=1, allow_inf_nan=False)
    median_sharpness: float = Field(ge=0, allow_inf_nan=False)
    warnings: tuple[str, ...] = ()
    issues: tuple[CaptureIssue, ...] = ()

    @model_validator(mode="after")
    def suitability_matches_issues(self) -> Self:
        """Keep the top-level decision consistent with its evidence."""
        if self.suitable == bool(self.issues):
            msg = "suitable must be true exactly when no blocking issues exist"
            raise ValueError(msg)
        return self


def _fraction(value: object) -> float:
    text = str(value)
    numerator, separator, denominator = text.partition("/")
    if not separator:
        return float(numerator)
    divisor = float(denominator)
    return float(numerator) / divisor if divisor else 0.0


def parse_ffprobe(path: Path, payload: dict[str, Any]) -> VideoProbe:
    """Normalize a machine-readable ffprobe response."""
    streams = payload.get("streams")
    if not isinstance(streams, list):
        msg = "ffprobe did not return a video stream"
        raise TypeError(msg)
    video_streams = [
        item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"
    ]
    stream = video_streams[0] if video_streams else None
    media_format = payload.get("format")
    if stream is None or not isinstance(media_format, dict):
        msg = "ffprobe did not return readable video metadata"
        raise ValueError(msg)
    side_data = stream.get("side_data_list", [])
    rotation = 0
    if isinstance(side_data, list):
        rotation = next(
            (
                int(item["rotation"])
                for item in side_data
                if isinstance(item, dict) and "rotation" in item
            ),
            0,
        )
    rotation %= 360
    if rotation not in {0, 90, 180, 270}:
        msg = "video orientation changes or uses an unsupported rotation"
        raise ValueError(msg)
    stream_rotations = {_stream_rotation(item) for item in video_streams}
    if len(stream_rotations) > 1:
        msg = "video streams use inconsistent orientations"
        raise ValueError(msg)
    return VideoProbe(
        path=path.resolve(),
        container=str(media_format.get("format_name", "")),
        codec=str(stream.get("codec_name", "")),
        duration_seconds=float(media_format.get("duration", 0)),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        frame_rate=_fraction(stream.get("avg_frame_rate", "0/1")),
        rotation_degrees=cast("Literal[0, 90, 180, 270]", rotation),
    )


def _stream_rotation(stream: dict[str, Any]) -> int:
    side_data = stream.get("side_data_list", [])
    if not isinstance(side_data, list):
        return 0
    return next(
        (
            int(item["rotation"]) % 360
            for item in side_data
            if isinstance(item, dict) and "rotation" in item
        ),
        0,
    )


def probe_video(
    path: Path,
    ffprobe: Path | str = "ffprobe",
    *,
    timeout: float = 15.0,
) -> VideoProbe:
    """Probe one local video through bounded, non-shell ffprobe execution."""
    if not path.is_file():
        msg = f"video does not exist: {path}"
        raise ValueError(msg)
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration:stream=codec_type,codec_name,width,height,avg_frame_rate:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        msg = "The video could not be read. Export an MP4 or MOV using H.264 or HEVC."
        raise ValueError(msg) from error
    if not isinstance(payload, dict):
        msg = "ffprobe returned an invalid response"
        raise TypeError(msg)
    return parse_ffprobe(path, payload)


def validate_probe(
    probe: VideoProbe,
    constraints: CaptureConstraints | None = None,
) -> tuple[CaptureIssue, ...]:
    """Return every hard input-envelope failure in a stable order."""
    rules = constraints or CaptureConstraints()
    issues: list[CaptureIssue] = []
    if probe.duration_seconds < rules.minimum_duration:
        issues.append(
            CaptureIssue(
                code="duration_too_short",
                message=f"The capture is only {probe.duration_seconds:.1f} seconds long.",
                recovery_action=(
                    f"Record at least {rules.minimum_duration:.0f} seconds with slow movement."
                ),
            )
        )
    if probe.duration_seconds > rules.maximum_duration:
        issues.append(
            CaptureIssue(
                code="duration_too_long",
                message=f"The capture is {probe.duration_seconds:.1f} seconds long.",
                recovery_action=f"Trim it to {rules.maximum_duration:.0f} seconds or less.",
            )
        )
    if probe.codec not in rules.codecs:
        issues.append(
            CaptureIssue(
                code="unsupported_codec",
                message=f"The {probe.codec or 'unknown'} video codec is not supported.",
                recovery_action="Export an MP4 or MOV using H.264 or HEVC.",
            )
        )
    container_names = set(probe.container.split(","))
    if not container_names.intersection(rules.containers):
        issues.append(
            CaptureIssue(
                code="unsupported_container",
                message=f"The {probe.container or 'unknown'} container is not supported.",
                recovery_action="Export the capture as an MP4 or MOV file.",
            )
        )
    if (
        probe.display_long_edge < rules.minimum_long_edge
        or probe.display_short_edge < rules.minimum_short_edge
    ):
        issues.append(
            CaptureIssue(
                code="resolution_too_low",
                message=f"The capture is only {probe.display_width}x{probe.display_height}.",
                recovery_action="Record at 720p or 1080p with a fixed lens and orientation.",
            )
        )
    if (
        probe.display_long_edge > rules.maximum_long_edge
        or probe.display_short_edge > rules.maximum_short_edge
    ):
        issues.append(
            CaptureIssue(
                code="resolution_too_high",
                message=f"The capture is {probe.display_width}x{probe.display_height}.",
                recovery_action=(
                    "Export at 1080p to keep reconstruction within its resource budget."
                ),
            )
        )
    return tuple(issues)


def score_frame(
    image: np.ndarray[Any, np.dtype[Any]],
    previous: np.ndarray[Any, np.dtype[Any]] | None,
    *,
    index: int,
    timestamp_seconds: float,
) -> FrameMetrics:
    """Measure blur, clipping, motion, and similarity for one decoded frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= DARK_CLIP_LEVEL) | (gray >= BRIGHT_CLIP_LEVEL)))
    if previous is None:
        motion = 0.0
        similarity = 1.0
        moving_fraction = 0.0
    else:
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(gray, previous_gray)
        mean_difference = float(np.mean(difference)) / 255.0
        motion = min(mean_difference * 2.0, 1.0)
        similarity = max(0.0, 1.0 - mean_difference)
        median_difference = float(np.median(difference))
        median_deviation = float(np.median(np.abs(difference - median_difference)))
        residual_threshold = max(25.0, median_difference + 3.0 * median_deviation)
        moving_fraction = float(np.mean(difference > residual_threshold))
    return FrameMetrics(
        index=index,
        timestamp_seconds=timestamp_seconds,
        sharpness=sharpness,
        clipped_fraction=clipped,
        motion=motion,
        similarity=similarity,
        moving_fraction=moving_fraction,
    )


def select_frames(
    candidates: Sequence[FrameMetrics],
    *,
    minimum: int = 60,
    maximum: int = 100,
) -> tuple[FrameMetrics, ...]:
    """Select the best candidate in evenly distributed temporal buckets."""
    if minimum <= 0 or maximum < minimum:
        msg = "frame selection bounds are invalid"
        raise ValueError(msg)
    ordered = sorted(candidates, key=lambda item: (item.timestamp_seconds, item.index))
    if len(ordered) <= maximum:
        return tuple(ordered)
    edges = np.linspace(0, len(ordered), maximum + 1, dtype=int)
    selected = [
        max(ordered[start:end], key=lambda item: (item.quality_score, -item.index))
        for start, end in pairwise(edges)
        if start < end
    ]
    return tuple(sorted(selected, key=lambda item: (item.timestamp_seconds, item.index)))


def analyse_metrics(
    probe: VideoProbe,
    candidates: Sequence[FrameMetrics],
    selected: Sequence[FrameMetrics],
) -> CaptureDiagnostics:
    """Produce stable capture diagnostics from explicit heuristics."""
    hard_issues = list(validate_probe(probe))
    if len(selected) < MINIMUM_SELECTED_FRAMES:
        hard_issues.append(
            CaptureIssue(
                code="insufficient_useful_frames",
                message=f"Only {len(selected)} useful frames were found.",
                recovery_action="Record more slowly with steady translation and overlapping views.",
            )
        )
    movement = float(np.median([item.motion for item in candidates])) if candidates else 0.0
    overlap = float(np.median([item.similarity for item in candidates])) if candidates else 0.0
    clipped = [item.clipped_fraction for item in candidates]
    exposure_stability = max(0.0, 1.0 - float(np.std(clipped)) * 4.0) if clipped else 0.0
    sharpness = float(np.median([item.sharpness for item in candidates])) if candidates else 0.0
    moving_objects = (
        float(np.median([item.moving_fraction for item in candidates])) if candidates else 0.0
    )
    warnings: list[str] = []
    if movement < LOW_MOVEMENT_WARNING:
        warnings.append("Camera movement is limited; translate around the room instead of panning.")
    elif movement > HIGH_MOVEMENT_WARNING:
        warnings.append("Camera movement may be too fast for reliable overlap.")
    if overlap < LOW_OVERLAP_WARNING:
        warnings.append("Neighbouring frames have weak visual overlap.")
    if clipped and float(np.mean(clipped)) > HIGH_CLIPPING_WARNING:
        warnings.append("Large areas are clipped dark or bright.")
    if moving_objects > MOVING_OBJECT_WARNING:
        warnings.append("Independent movement may reduce reconstruction stability.")
    return CaptureDiagnostics(
        probe=probe,
        suitable=not hard_issues,
        candidate_frames=len(candidates),
        selected_frames=len(selected),
        movement=movement,
        overlap_proxy=overlap,
        exposure_stability=exposure_stability,
        moving_object_proxy=moving_objects,
        median_sharpness=sharpness,
        warnings=tuple(warnings),
        issues=tuple(hard_issues),
    )


def analyse_video_file(
    path: Path,
    ffprobe: Path | str = "ffprobe",
) -> CaptureDiagnostics:
    """Run a bounded read-only analysis for the CLI and review surface."""
    probe = probe_video(path, ffprobe)
    if validate_probe(probe):
        return analyse_metrics(probe, (), ())
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = "The video frames could not be decoded. Export a new H.264 or HEVC file."
        raise ValueError(msg)
    sample_count = min(180, max(60, math.ceil(probe.duration_seconds * 6)))
    timestamps = np.linspace(0.0, max(probe.duration_seconds - 0.05, 0.0), sample_count)
    metrics: list[FrameMetrics] = []
    previous = None
    try:
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            metrics.append(
                score_frame(
                    frame,
                    previous,
                    index=index,
                    timestamp_seconds=float(timestamp),
                )
            )
            previous = frame
    finally:
        capture.release()
    selected = select_frames(metrics)
    return analyse_metrics(probe, metrics, selected)
