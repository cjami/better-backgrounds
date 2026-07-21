"""Focused tests for reconstruction controller transitions."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.camera import InputCameraSource
from better_backgrounds.desktop.main_window.build_controller import BuildController
from better_backgrounds.jobs.build_session import ReviewBuild
from better_backgrounds.jobs.events import CancelledEvent, ErrorEvent, ResultEvent
from better_backgrounds.reconstruction import SplatDiagnostics, SplatSelection
from better_backgrounds.reconstruction.sharp import SceneImageSelection

if TYPE_CHECKING:
    import pytest

    from better_backgrounds.desktop.pages import BuildPage
    from better_backgrounds.jobs.runner import JobRunner
    from better_backgrounds.reconstruction.sharp import SharpCheckpointInstaller
    from better_backgrounds.scene import SceneLibrary


class FakeBuildPage(QObject):
    """Provide the BuildPage signal and update boundary without widgets."""

    file_requested = Signal()
    file_dropped = Signal(str)
    capture_requested = Signal()
    capture_now_requested = Signal()
    capture_cancelled = Signal()
    build_requested = Signal(str)
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self) -> None:
        """Create empty page evidence."""
        super().__init__()
        self.completed: str | None = None
        self.reviewed: SceneImageSelection | None = None
        self.splat_reviewed: SplatSelection | None = None
        self.importing = False

    def set_model_ready(self, *, ready: bool) -> None:
        """Accept checkpoint status updates."""

    def show_upload(self) -> None:
        """Accept navigation to upload state."""

    def show_capture(self) -> None:
        """Accept navigation to the camera-capture state."""

    def set_capture_frame(self, _image: object) -> None:
        """Accept live capture-preview frames."""

    def set_countdown(self, _seconds: int) -> None:
        """Accept countdown updates."""

    def set_capture_error(self, _message: str) -> None:
        """Accept capture failures."""

    def show_review(self, selection: SceneImageSelection, _diagnostics: object = None) -> None:
        """Record restored retry context."""
        self.reviewed = selection

    def show_splat_review(
        self,
        selection: SplatSelection,
        _diagnostics: object = None,
    ) -> None:
        """Record direct-import review updates."""
        self.splat_reviewed = selection

    def set_completed(self, room_name: str) -> None:
        """Record successful publication."""
        self.completed = room_name

    def set_failed(self, _message: str, _recovery: str | None) -> None:
        """Accept a stable failure update."""

    def set_progress(self, _stage: str, _progress: float, _message: str) -> None:
        """Accept progress updates."""

    def reset_progress(self, *, importing: bool = False) -> None:
        """Accept a new build attempt."""
        self.importing = importing

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
        InputCameraSource(provider=tuple),
        lambda: None,
        Path("captures"),
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


def test_dispatch_path_routes_by_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route photos to SHARP image selection and splats to managed import."""
    controller, _page = create_controller()
    images: list[SceneImageSelection] = []
    splats: list[SplatSelection] = []
    monkeypatch.setattr(controller, "_select_image", images.append)
    monkeypatch.setattr(controller, "_select_splat", splats.append)

    controller._dispatch_path(Path("lounge.JPG"))  # noqa: SLF001
    controller._dispatch_path(Path("scene.ply"))  # noqa: SLF001
    controller._dispatch_path(Path("museum.ssog"))  # noqa: SLF001

    assert [selection.source_kind for selection in images] == ["upload"]
    assert images[0].source_path == Path("lounge.JPG")
    assert [selection.display_name for selection in splats] == ["scene.ply", "museum.ssog"]


def test_captured_room_selects_camera_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed a captured empty-room frame through image selection as a camera source."""
    controller, _page = create_controller()
    selections: list[SceneImageSelection] = []
    monkeypatch.setattr(controller, "_select_image", selections.append)
    monkeypatch.setattr(controller, "_open_build_tab", lambda: None)

    controller._accept_capture(Path("captures/room.png"))  # noqa: SLF001

    assert len(selections) == 1
    assert selections[0].source_kind == "camera"
    assert selections[0].source_path == Path("captures/room.png")


def test_splat_build_bypasses_checkpoint_and_uses_import_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch direct imports without entering SHARP preparation."""
    controller, page = create_controller()
    selection = SplatSelection("room.ply", Path("room.ply"))
    controller._splat_factory = lambda job_id, source: [job_id, str(source)]  # noqa: SLF001
    controller._splat_diagnostics = SplatDiagnostics(  # noqa: SLF001
        gaussian_count=4,
        file_size=128,
        layout="standard",
        framing="Automatic COLMAP framing",
        bounds_minimum=(-1.0, -1.0, 2.0),
        bounds_maximum=(1.0, 1.0, 3.0),
        center_of_mass=(0.0, 0.0, 2.5),
    )
    commands: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        controller,
        "_start_runner",
        lambda command, job_id: commands.append((list(command), job_id)),
    )
    controller.session.select_source(selection)

    controller._start_build("cuda")  # noqa: SLF001

    assert page.importing
    assert len(commands) == 1
    assert commands[0][0][1] == "room.ply"
