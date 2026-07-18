"""Focused tests for reconstruction controller transitions."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.main_window.build_controller import BuildController
from better_backgrounds.jobs.build_session import ReviewBuild
from better_backgrounds.jobs.events import CancelledEvent, ErrorEvent, ResultEvent
from better_backgrounds.reconstruction.sharp import SceneImageSelection

if TYPE_CHECKING:
    import pytest

    from better_backgrounds.desktop.pages import BuildPage
    from better_backgrounds.jobs.runner import JobRunner
    from better_backgrounds.reconstruction.sharp import SharpCheckpointInstaller
    from better_backgrounds.scene import SceneLibrary


class FakeBuildPage(QObject):
    """Provide the BuildPage signal and update boundary without widgets."""

    image_requested = Signal()
    build_requested = Signal(str)
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self) -> None:
        """Create empty page evidence."""
        super().__init__()
        self.completed: str | None = None
        self.reviewed: SceneImageSelection | None = None

    def set_model_ready(self, *, ready: bool) -> None:
        """Accept checkpoint status updates."""

    def show_upload(self) -> None:
        """Accept navigation to upload state."""

    def show_review(self, selection: SceneImageSelection, _diagnostics: object = None) -> None:
        """Record restored retry context."""
        self.reviewed = selection

    def set_completed(self, room_name: str) -> None:
        """Record successful publication."""
        self.completed = room_name

    def set_failed(self, _message: str, _recovery: str | None) -> None:
        """Accept a stable failure update."""

    def set_progress(self, _stage: str, _progress: float, _message: str) -> None:
        """Accept progress updates."""

    def reset_progress(self) -> None:
        """Accept a new build attempt."""

    def set_image_error(self, _message: str) -> None:
        """Accept image validation failures."""


class FakeCheckpoint:
    """Report a prepared checkpoint."""

    @staticmethod
    def is_ready() -> bool:
        """Avoid checkpoint preparation in controller tests."""
        return True


class FakeLibrary:
    """Record completed scene registration."""

    def register(self, scene_id: str, fallback_name: str) -> tuple[str, str]:
        """Return deterministic room evidence."""
        return fallback_name, scene_id


class FakeRunner:
    """Record cooperative cancellation."""

    def __init__(self) -> None:
        """Create an empty cancellation log."""
        self.cancelled: list[str] = []

    def cancel(self, job_id: str) -> None:
        """Record the running job identifier."""
        self.cancelled.append(job_id)


def create_controller() -> tuple[BuildController, FakeBuildPage]:
    """Create a controller with no external worker or model dependencies."""
    QApplication.instance() or QApplication([])
    page = FakeBuildPage()
    controller = BuildController(
        QWidget(),
        cast("BuildPage", page),
        cast("SceneLibrary", FakeLibrary()),
        lambda _job_id, _outcome: [],
        None,
        None,
        cast("SharpCheckpointInstaller", FakeCheckpoint()),
        lambda: None,
    )
    return controller, page


def test_build_controller_completion_retry_cancellation_and_checkpoint_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep terminal and continuation transitions owned by the controller."""
    controller, page = create_controller()
    selection = SceneImageSelection("room.jpg", None)
    completed: list[tuple[str, str]] = []
    controller.scene_completed.connect(lambda scene_id, name: completed.append((scene_id, name)))
    controller.session.select_image(selection)
    controller.session.start("job-complete")
    controller._handle_job_event(  # noqa: SLF001
        ResultEvent(job_id="job-complete", scene_id="scene-1", message="done"),
    )
    assert completed == [("scene-1", "Room")]
    assert page.completed == "Room"

    controller.session.select_image(selection)
    controller.session.start("job-retry")
    controller._handle_job_event(  # noqa: SLF001
        ErrorEvent(job_id="job-retry", code="failed", message="failed"),
    )
    controller._retry_build()  # noqa: SLF001
    assert isinstance(controller.session.state, ReviewBuild)
    assert page.reviewed == selection

    controller.session.start("job-cancel")
    runner = FakeRunner()
    controller._runner = cast("JobRunner", runner)  # noqa: SLF001
    controller._cancel_build()  # noqa: SLF001
    assert runner.cancelled == ["job-cancel"]
    controller._handle_job_event(  # noqa: SLF001
        CancelledEvent(job_id="job-cancel", message="cancelled"),
    )

    source = tmp_source()
    controller.session.select_image(SceneImageSelection("source.jpg", source))
    controller.session.start("job-prepare")
    launched: list[tuple[str, str]] = []
    controller._preparing_checkpoint = True  # noqa: SLF001
    monkeypatch.setattr(
        controller,
        "_launch_sharp",
        lambda job_id, _selection, device: launched.append((job_id, device)),
    )
    controller._handle_job_event(  # noqa: SLF001
        ResultEvent(job_id="job-prepare", scene_id="checkpoint", message="ready"),
    )
    assert launched == [("job-prepare", "auto")]


def tmp_source() -> Path:
    """Return a source path; checkpoint continuation does not read it."""
    return Path("source.jpg")
