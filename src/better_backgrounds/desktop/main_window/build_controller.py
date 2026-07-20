"""Room reconstruction build controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, SignalInstance, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from better_backgrounds.desktop.camera import RoomCaptureController
from better_backgrounds.desktop.pages.build import SPLAT_EXTENSIONS
from better_backgrounds.jobs.build_session import (
    BuildSession,
    CompletedBuild,
    FailedBuild,
    ReviewBuild,
    RunningBuild,
)
from better_backgrounds.jobs.events import (
    CancelledEvent,
    ErrorEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)
from better_backgrounds.jobs.runner import JobRunner
from better_backgrounds.reconstruction import (
    SplatDiagnostics,
    SplatSelection,
    inspect_gaussian_scene,
)
from better_backgrounds.reconstruction.sharp import (
    SceneImageDiagnostics,
    SceneImageSelection,
    SharpCheckpointInstaller,
    inspect_scene_image,
)

if TYPE_CHECKING:
    from better_backgrounds.desktop.camera import InputCameraSource
    from better_backgrounds.desktop.pages import BuildPage
    from better_backgrounds.scene import SceneLibrary

CommandFactory = Callable[[str, str], Sequence[str]]
SharpCommandFactory = Callable[[str, Path, str, str], Sequence[str]]
SharpPrepareCommandFactory = Callable[[str], Sequence[str]]
SplatCommandFactory = Callable[[str, Path], Sequence[str]]


class RunnerSignals(QObject):
    """Marshal worker-thread callbacks onto the Qt main thread."""

    event_received = Signal(object)


class BuildController(QObject):
    """Own reconstruction state, checkpoint continuation, and worker lifetime."""

    scene_completed = Signal(str, str)

    def __init__(
        self,
        parent: QWidget,
        page: BuildPage,
        library: SceneLibrary,
        command_factory: CommandFactory,
        sharp_factory: SharpCommandFactory | None,
        prepare_factory: SharpPrepareCommandFactory | None,
        checkpoint: SharpCheckpointInstaller,
        open_build_tab: Callable[[], None],
        camera_source: InputCameraSource,
        selected_camera_id: Callable[[], str | None],
        capture_root: Path,
        splat_factory: SplatCommandFactory | None = None,
    ) -> None:
        """Connect a build page to its reconstruction and camera-capture services."""
        super().__init__(parent)
        self._parent = parent
        self._page = page
        self._library = library
        self._command_factory = command_factory
        self._sharp_factory = sharp_factory
        self._prepare_factory = prepare_factory
        self._checkpoint = checkpoint
        self._splat_factory = splat_factory
        self._open_build_tab = open_build_tab
        self._session = BuildSession()
        self._runner: JobRunner | None = None
        self._signals = RunnerSignals(self)
        self._signals.event_received.connect(self._handle_job_event)
        self._preparing_checkpoint = False
        self._pending_device = "auto"
        self._image_diagnostics: SceneImageDiagnostics | None = None
        self._splat_diagnostics: SplatDiagnostics | None = None
        self._capture = RoomCaptureController(
            page,
            camera_source,
            selected_camera_id,
            capture_root,
            parent=self,
        )
        self._capture.captured.connect(self._accept_capture)
        page.file_requested.connect(self._choose_file)
        page.file_dropped.connect(self._accept_dropped_path)
        page.build_requested.connect(self._start_build)
        page.cancel_requested.connect(self._cancel_build)
        page.retry_requested.connect(self._retry_build)
        page.set_model_ready(ready=checkpoint.is_ready())

    @property
    def capture_active(self) -> SignalInstance:
        """Expose the camera-capture active signal for live-preview coordination."""
        return self._capture.active

    @property
    def session(self) -> BuildSession:
        """Return the current build state machine."""
        return self._session

    def open(self) -> None:
        """Open the build tab without interrupting a running job."""
        self._open_build()

    def start_smoke(self) -> None:
        """Run the prepared sample through the successful fake worker."""
        self._open_build_tab()
        self._select_image(SceneImageSelection("Prepared smoke room", None))
        self._start_build("auto")

    def shutdown(self) -> None:
        """Close the active worker process tree and release the capture camera."""
        self._capture.shutdown()
        if self._runner is not None:
            self._runner.close()

    @Slot()
    def _open_build(self) -> None:
        if not isinstance(self._session.state, RunningBuild):
            self._page.show_upload()
        self._open_build_tab()

    @Slot()
    def _choose_file(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self._parent,
            "Choose a room photo or Gaussian splat",
            "",
            "Room sources (*.jpg *.jpeg *.png *.webp *.ply *.ssog *.zip);;All files (*)",
        )
        if path:
            self._dispatch_path(Path(path))

    @Slot(str)
    def _accept_dropped_path(self, path: str) -> None:
        self._open_build_tab()
        self._dispatch_path(Path(path))

    def _dispatch_path(self, path: Path) -> None:
        """Route one selected file to image build or splat import by extension."""
        suffix = path.suffix.lower()
        if suffix in SPLAT_EXTENSIONS:
            self._select_splat(SplatSelection(path.name, path))
            return
        # Images and unknown types fall through to image validation, which
        # produces the friendly "Choose a JPEG, PNG, or WebP…" error for
        # anything that is not a supported room photo.
        self._select_image(
            SceneImageSelection(display_name=path.name, source_path=path, source_kind="upload"),
        )

    @Slot(object)
    def _accept_capture(self, path: object) -> None:
        if not isinstance(path, Path):
            return
        self._open_build_tab()
        self._select_image(
            SceneImageSelection(display_name=path.name, source_path=path, source_kind="camera"),
        )

    def _select_image(self, selection: SceneImageSelection) -> None:
        self._session.select_source(selection)
        self._splat_diagnostics = None
        if selection.source_path is None:
            self._image_diagnostics = None
            self._page.show_review(selection)
            return
        try:
            diagnostics = inspect_scene_image(selection.source_path)
        except ValueError as error:
            self._image_diagnostics = None
            self._page.show_review(selection)
            self._page.set_image_error(str(error))
            return
        self._image_diagnostics = diagnostics
        self._page.show_review(selection, diagnostics)

    def _select_splat(self, selection: SplatSelection) -> None:
        self._session.select_source(selection)
        self._image_diagnostics = None
        try:
            diagnostics = inspect_gaussian_scene(selection.source_path, validate_values=False)
        except ValueError as error:
            self._splat_diagnostics = None
            self._page.show_splat_review(selection)
            self._page.set_image_error(str(error))
            return
        self._splat_diagnostics = diagnostics
        self._page.show_splat_review(selection, diagnostics)

    @Slot(str)
    def _start_build(self, device_name: str) -> None:
        state = self._session.state
        if not isinstance(state, ReviewBuild):
            return
        selection = state.selection
        if isinstance(selection, SplatSelection):
            job_id = uuid4().hex
            self._session.start(job_id)
            self._page.reset_progress(importing=True)
            factory = self._splat_factory
            if factory is None:
                self._handle_job_event(
                    ErrorEvent(
                        job_id=job_id,
                        code="splat_worker_unavailable",
                        message="The Gaussian splat import worker is unavailable.",
                        recovery_action="Reinstall the complete application runtime.",
                    ),
                )
                return
            self._start_runner(factory(job_id, selection.source_path), job_id)
            return
        if selection.source_path is not None and not self._checkpoint.is_ready():
            if self._prepare_factory is None:
                self._page.set_image_error(
                    "The SHARP checkpoint is not prepared. Run prepare-sharp after reviewing "
                    "Apple's research-only model license."
                )
                return
            answer = QMessageBox.question(
                self._parent,
                "Prepare Apple SHARP model",
                "SHARP's 2.8 GB checkpoint is licensed only for non-commercial scientific "
                "research and excludes product development. Accept that model license and "
                "download the pinned checkpoint now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        job_id = uuid4().hex
        self._session.start(job_id)
        self._page.reset_progress(importing=False)
        self._pending_device = device_name
        if selection.source_path is None:
            self._start_runner(self._command_factory(job_id, "success"), job_id)
        elif not self._checkpoint.is_ready():
            self._preparing_checkpoint = True
            factory = self._prepare_factory
            if factory is None:
                return
            self._start_runner(factory(job_id), job_id)
        else:
            self._launch_sharp(job_id, selection, device_name)

    def _launch_sharp(
        self,
        job_id: str,
        selection: SceneImageSelection,
        device_name: str,
    ) -> None:
        source = selection.source_path
        factory = self._sharp_factory
        if source is None or factory is None:
            self._handle_job_event(
                ErrorEvent(
                    job_id=job_id,
                    code="sharp_worker_unavailable",
                    message="The SHARP build worker is unavailable.",
                    recovery_action="Reinstall the complete application runtime.",
                )
            )
            return
        self._start_runner(
            factory(job_id, source, device_name, selection.source_kind),
            job_id,
        )

    def _start_runner(self, command: Sequence[str], job_id: str) -> None:
        runner = JobRunner(self._signals.event_received.emit)
        self._runner = runner
        runner.start(command, job_id=job_id)

    @Slot()
    def _cancel_build(self) -> None:
        state = self._session.state
        if isinstance(state, RunningBuild) and self._runner is not None:
            self._runner.cancel(state.job_id)

    @Slot()
    def _retry_build(self) -> None:
        state = self._session.retry()
        selection = state.selection
        if isinstance(selection, SplatSelection):
            try:
                self._splat_diagnostics = inspect_gaussian_scene(
                    selection.source_path,
                    validate_values=False,
                )
            except ValueError as error:
                self._page.show_splat_review(selection)
                self._page.set_image_error(str(error))
            else:
                self._page.show_splat_review(selection, self._splat_diagnostics)
            return
        if selection.source_path is None:
            self._page.show_review(selection)
            return
        try:
            self._image_diagnostics = inspect_scene_image(selection.source_path)
        except ValueError as error:
            self._page.show_review(selection)
            self._page.set_image_error(str(error))
        else:
            self._page.show_review(selection, self._image_diagnostics)

    @Slot(object)
    def _handle_job_event(self, event: object) -> None:
        if not isinstance(
            event,
            ProgressEvent | WarningEvent | ResultEvent | ErrorEvent | CancelledEvent,
        ):
            return
        if self._finish_checkpoint_preparation(event):
            return
        if not self._session.apply(event):
            return
        state = self._session.state
        if isinstance(event, ProgressEvent | WarningEvent) and isinstance(state, RunningBuild):
            self._page.set_progress(state.stage, state.progress, state.message)
        elif isinstance(state, FailedBuild):
            self._page.set_failed(state.message, state.recovery_action)
        elif isinstance(state, ReviewBuild):
            self._preparing_checkpoint = False
            if isinstance(state.selection, SplatSelection):
                self._page.show_splat_review(state.selection, self._splat_diagnostics)
            else:
                diagnostics = (
                    self._image_diagnostics if state.selection.source_path is not None else None
                )
                self._page.show_review(state.selection, diagnostics)
        elif isinstance(state, CompletedBuild):
            room_name, room_id = self._library.register(
                state.scene_id,
                self._room_name_for(state.selection),
            )
            self._page.set_completed(room_name)
            self.scene_completed.emit(room_id, room_name)

    def _finish_checkpoint_preparation(self, event: object) -> bool:
        if not self._preparing_checkpoint or not isinstance(event, ResultEvent):
            return False
        state = self._session.state
        if (
            not isinstance(state, RunningBuild)
            or event.job_id != state.job_id
            or not isinstance(state.selection, SceneImageSelection)
        ):
            return False
        self._preparing_checkpoint = False
        self._page.set_model_ready(ready=True)
        self._launch_sharp(state.job_id, state.selection, self._pending_device)
        return True

    @staticmethod
    def _room_name_for(selection: SceneImageSelection | SplatSelection) -> str:
        return Path(selection.display_name).stem.replace("_", " ").strip().title()
