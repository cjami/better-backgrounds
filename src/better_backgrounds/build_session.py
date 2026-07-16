"""State for one room-building job, independent of desktop navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from better_backgrounds.protocol import (
    CancelledEvent,
    ErrorEvent,
    JobEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)

if TYPE_CHECKING:
    from pathlib import Path


class InvalidBuildStateError(RuntimeError):
    """Raised when a build action is not valid for the current job state."""


@dataclass(frozen=True, slots=True)
class VideoSelection:
    """A room video selected entirely within the trusted Python process."""

    display_name: str
    source_path: Path | None
    sample: bool = False


@dataclass(frozen=True, slots=True)
class IdleBuild:
    """Wait for a room video."""


@dataclass(frozen=True, slots=True)
class ReviewBuild:
    """Show deterministic placeholder capture diagnostics."""

    selection: VideoSelection


@dataclass(frozen=True, slots=True)
class RunningBuild:
    """Track one active reconstruction job."""

    selection: VideoSelection
    job_id: str
    stage: str = "validation"
    progress: float | None = 0.0
    message: str = "Preparing the build"


@dataclass(frozen=True, slots=True)
class FailedBuild:
    """Preserve enough context for an explicit retry."""

    selection: VideoSelection
    message: str
    recovery_action: str | None


@dataclass(frozen=True, slots=True)
class CompletedBuild:
    """Record the room created by a successful job."""

    selection: VideoSelection
    scene_id: str


BuildState = IdleBuild | ReviewBuild | RunningBuild | FailedBuild | CompletedBuild


class BuildSession:
    """Own the lifecycle of a build without controlling the app's tabs."""

    def __init__(self) -> None:
        """Start without a selected video."""
        self._state: BuildState = IdleBuild()

    @property
    def state(self) -> BuildState:
        """Return the current immutable build state."""
        return self._state

    def select_video(self, selection: VideoSelection) -> ReviewBuild:
        """Select a video unless a build is currently running."""
        if isinstance(self._state, RunningBuild):
            msg = "A video cannot be replaced while its build is running."
            raise InvalidBuildStateError(msg)
        next_state = ReviewBuild(selection)
        self._state = next_state
        return next_state

    def start(self, job_id: str) -> RunningBuild:
        """Start a job after reviewing a selection."""
        if not isinstance(self._state, ReviewBuild):
            msg = "A build requires a reviewed video selection."
            raise InvalidBuildStateError(msg)
        next_state = RunningBuild(selection=self._state.selection, job_id=job_id)
        self._state = next_state
        return next_state

    def apply(self, event: JobEvent) -> bool:
        """Apply an event and ignore events from earlier jobs."""
        state = self._state
        if not isinstance(state, RunningBuild) or event.job_id != state.job_id:
            return False
        if isinstance(event, ProgressEvent):
            self._state = RunningBuild(
                selection=state.selection,
                job_id=state.job_id,
                stage=event.stage,
                progress=event.progress,
                message=event.message,
            )
        elif isinstance(event, WarningEvent):
            self._state = RunningBuild(
                selection=state.selection,
                job_id=state.job_id,
                stage=state.stage,
                progress=state.progress,
                message=event.message,
            )
        elif isinstance(event, ResultEvent):
            self._state = CompletedBuild(state.selection, event.scene_id)
        elif isinstance(event, ErrorEvent):
            self._state = FailedBuild(
                selection=state.selection,
                message=event.message,
                recovery_action=event.recovery_action,
            )
        elif isinstance(event, CancelledEvent):
            self._state = ReviewBuild(state.selection)
        return True

    def retry(self) -> ReviewBuild:
        """Return a failed build to its reviewed selection."""
        if not isinstance(self._state, FailedBuild):
            msg = "Only a failed build can be retried."
            raise InvalidBuildStateError(msg)
        next_state = ReviewBuild(self._state.selection)
        self._state = next_state
        return next_state
