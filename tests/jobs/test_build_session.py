"""Feature-first: Behavior tests for a build session independent of app navigation."""

from pathlib import Path

import pytest

from better_backgrounds.jobs.build_session import (
    BuildSession,
    CompletedBuild,
    FailedBuild,
    InvalidBuildStateError,
    ReviewBuild,
    RunningBuild,
)
from better_backgrounds.jobs.events import ErrorEvent, ProgressEvent, ResultEvent
from better_backgrounds.reconstruction.sharp import SceneImageSelection


def selection() -> SceneImageSelection:
    """Create a stable room-image selection."""
    return SceneImageSelection("room.jpg", Path("room.jpg"))


def test_successful_build_completes_without_controlling_navigation() -> None:
    """Track review, progress, and completion for one job."""
    session = BuildSession()
    assert isinstance(session.select_image(selection()), ReviewBuild)
    assert isinstance(session.start("job-1"), RunningBuild)

    assert session.apply(
        ProgressEvent(
            job_id="job-1",
            stage="inference",
            progress=0.5,
            message="Predicting Gaussians",
        ),
    )
    assert isinstance(session.state, RunningBuild)
    assert session.apply(ResultEvent(job_id="job-1", scene_id="scene-1", message="Ready"))
    assert isinstance(session.state, CompletedBuild)


def test_stale_job_events_are_ignored() -> None:
    """Prevent an old process from changing the active build."""
    session = BuildSession()
    session.select_image(selection())
    session.start("current")

    accepted = session.apply(
        ResultEvent(job_id="stale", scene_id="wrong", message="Stale result"),
    )

    assert not accepted
    assert isinstance(session.state, RunningBuild)


def test_failure_preserves_selection_for_retry() -> None:
    """Retain user context after a failed build."""
    session = BuildSession()
    session.select_image(selection())
    session.start("job-1")

    session.apply(
        ErrorEvent(
            job_id="job-1",
            code="failed",
            message="Could not load the SHARP model",
            recovery_action="Prepare the checkpoint",
        ),
    )

    assert isinstance(session.state, FailedBuild)
    assert isinstance(session.retry(), ReviewBuild)


def test_build_requires_a_reviewed_image() -> None:
    """Reject a build that has no selected image."""
    with pytest.raises(InvalidBuildStateError):
        BuildSession().start("job-1")
