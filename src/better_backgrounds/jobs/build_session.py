"""State for one room-building job, independent of desktop navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from better_backgrounds.jobs.events import (
    CancelledEvent,
    ErrorEvent,
    JobEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)

if TYPE_CHECKING:
    from better_backgrounds.reconstruction import SplatSelection
    from better_backgrounds.reconstruction.sharp import SceneImageSelection

    RoomSourceSelection = SceneImageSelection | SplatSelection


class InvalidBuildStateError(RuntimeError):
    """Raised when a build action is not valid for the current job state."""


@dataclass(frozen=True, slots=True)
class IdleBuild:
    """Wait for a room source."""


@dataclass(frozen=True, slots=True)
class ReviewBuild:
    """Show the selected source and its diagnostics."""

    selection: RoomSourceSelection


@dataclass(frozen=True, slots=True)
class RunningBuild:
    """Track one active room publication job."""

    selection: RoomSourceSelection
    job_id: str
    stage: str = "validation"
    progress: float | None = 0.0
    message: str = "Preparing the build"


@dataclass(frozen=True, slots=True)
class FailedBuild:
    """Preserve enough context for an explicit retry."""

    selection: RoomSourceSelection
    message: str
    recovery_action: str | None


@dataclass(frozen=True, slots=True)
class CompletedBuild:
    """Record the room created by a successful job."""

    selection: RoomSourceSelection
    scene_id: str


BuildState = IdleBuild | ReviewBuild | RunningBuild | FailedBuild | CompletedBuild


class BuildSession:
    """Own the lifecycle of a room build or import without controlling tabs."""

    def __init__(self) -> None:
        """Start without a selected image."""
        self._state: BuildState = IdleBuild()

    @property
    def state(self) -> BuildState:
        """Return the current immutable build state."""
        return self._state

    def select_source(self, selection: RoomSourceSelection) -> ReviewBuild:
        """Select a room source unless a build is currently running."""
        if isinstance(self._state, RunningBuild):
            msg = "A room source cannot be replaced while its build is running."
            raise InvalidBuildStateError(msg)
        next_state = ReviewBuild(selection)
        self._state = next_state
        return next_state

    def select_image(self, selection: RoomSourceSelection) -> ReviewBuild:
        """Retain the original selection API for existing callers."""
        return self.select_source(selection)

    def start(self, job_id: str) -> RunningBuild:
        """Start a job after reviewing a selection."""
        if not isinstance(self._state, ReviewBuild):
            msg = "A build requires a reviewed room source."
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
