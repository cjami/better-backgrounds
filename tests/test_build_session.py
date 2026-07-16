"""Behavior tests for a build session independent of app navigation."""

from pathlib import Path

import pytest

from better_backgrounds.build_session import (
    BuildSession,
    CompletedBuild,
    FailedBuild,
    InvalidBuildStateError,
    ReviewBuild,
    RunningBuild,
    VideoSelection,
)
from better_backgrounds.protocol import ErrorEvent, ProgressEvent, ResultEvent


def selection() -> VideoSelection:
    """Create a stable video selection."""
    return VideoSelection("room.mp4", Path("room.mp4"))


def test_successful_build_completes_without_controlling_navigation() -> None:
    """Track review, progress, and completion for one job."""
    session = BuildSession()
    assert isinstance(session.select_video(selection()), ReviewBuild)
    assert isinstance(session.start("job-1"), RunningBuild)

    assert session.apply(
        ProgressEvent(
            job_id="job-1",
            stage="camera_estimation",
            progress=0.5,
            message="Estimating poses",
        ),
    )
    assert isinstance(session.state, RunningBuild)
    assert session.apply(ResultEvent(job_id="job-1", scene_id="scene-1", message="Ready"))
    assert isinstance(session.state, CompletedBuild)


def test_stale_job_events_are_ignored() -> None:
    """Prevent an old process from changing the active build."""
    session = BuildSession()
    session.select_video(selection())
    session.start("current")

    accepted = session.apply(
        ResultEvent(job_id="stale", scene_id="wrong", message="Stale result"),
    )

    assert not accepted
    assert isinstance(session.state, RunningBuild)


def test_failure_preserves_selection_for_retry() -> None:
    """Retain user context after a failed build."""
    session = BuildSession()
    session.select_video(selection())
    session.start("job-1")

    session.apply(
        ErrorEvent(
            job_id="job-1",
            code="failed",
            message="Could not estimate cameras",
            recovery_action="Capture more overlap",
        ),
    )

    assert isinstance(session.state, FailedBuild)
    assert isinstance(session.retry(), ReviewBuild)


def test_build_requires_a_reviewed_video() -> None:
    """Reject a build that has no selected video."""
    with pytest.raises(InvalidBuildStateError):
        BuildSession().start("job-1")
