"""Main window coordinating independent product tabs and room-building jobs."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from platformdirs import user_cache_path, user_data_path
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.build_session import (
    BuildSession,
    CompletedBuild,
    FailedBuild,
    ReviewBuild,
    RunningBuild,
    VideoSelection,
)
from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.pages import AdjustPage, BuildPage, ComparePage, ShowPage
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.input_camera import (
    InputCamera,
    InputCameraSelectionStore,
    InputCameraSource,
)
from better_backgrounds.job_runner import JobRunner
from better_backgrounds.managed_tools import resolved_executable_paths
from better_backgrounds.matting import LivePreferencesStore, MattingSettings
from better_backgrounds.protocol import (
    CancelledEvent,
    ErrorEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)
from better_backgrounds.reconstruction import ReconstructionQuality
from better_backgrounds.scene import (
    AssetInstaller,
    ManagedSceneResolver,
    SceneCatalogue,
    SceneReference,
    Viewpoint,
    ViewpointStore,
    load_sample_manifest,
)
from better_backgrounds.video_analysis import CaptureDiagnostics, analyse_video_file

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent

CommandFactory = Callable[[str, str], Sequence[str]]
ReconstructionCommandFactory = Callable[
    [str, Path, bool, ReconstructionQuality],
    Sequence[str],
]
RendererFactory = Callable[[], QWidget]

TAB_NAMES = ("Show", "Build", "Adjust", "Compare")
COMPARE_TAB = 3
DEFAULT_WIPE = 52
DEFAULT_ROOMS = (
    "Table Tennis Room — Sample",
    "Loft — North Window",
    "Studio — West Wall",
    "Living room",
    "Bookshelf corner",
)


class RunnerSignals(QObject):
    """Marshal worker-thread callbacks onto the Qt main thread."""

    event_received = Signal(object)


class AssetSignals(QObject):
    """Marshal sample install callbacks onto the Qt main thread."""

    progressed = Signal(int, int)
    completed = Signal()
    failed = Signal(str)


class AnalysisSignals(QObject):
    """Marshal video-analysis results onto the Qt main thread."""

    completed = Signal(object)
    failed = Signal(str, str)


class TabHeader(QFrame):
    """Provide direct navigation between the four product areas."""

    tab_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the brand, tabs, and local-status indicator."""
        super().__init__(parent)
        self.setObjectName("header")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        logo = QLabel()
        logo.setFixedSize(22, 22)
        logo.setPixmap(application_icon().pixmap(22, 22))
        layout.addWidget(logo)
        brand = QLabel("Better Backgrounds")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        divider = QFrame()
        divider.setObjectName("headerDivider")
        divider.setFixedSize(1, 26)
        layout.addSpacing(14)
        layout.addWidget(divider)
        layout.addSpacing(8)
        self._tabs: list[QPushButton] = []
        for index, title in enumerate(TAB_NAMES):
            tab = QPushButton(title)
            tab.setObjectName("tab")
            tab.setAccessibleName(f"Open {title} tab")
            tab.clicked.connect(
                lambda _checked=False, tab_index=index: self.tab_selected.emit(tab_index),
            )
            self._tabs.append(tab)
            layout.addWidget(tab)
        layout.addStretch()
        room_pill = QFrame()
        room_pill.setObjectName("roomPill")
        room_layout = QHBoxLayout(room_pill)
        room_layout.setContentsMargins(12, 0, 12, 0)
        room_layout.setSpacing(7)
        dot = QLabel("●")
        dot.setObjectName("success")
        room_layout.addWidget(dot)
        self._room = QLabel("No room selected")
        room_layout.addWidget(self._room)
        layout.addWidget(room_pill)
        for label, accessible_name in (("?", "Help"), ("⚙︎", "Settings")):
            action = QPushButton(label)
            action.setObjectName("headerIcon")
            action.setAccessibleName(accessible_name)
            layout.addWidget(action)

    def set_active_tab(self, index: int) -> None:
        """Highlight the selected product tab."""
        for tab_index, tab in enumerate(self._tabs):
            tab.setProperty("active", tab_index == index)
            tab.style().unpolish(tab)
            tab.style().polish(tab)

    def set_room(self, room: str) -> None:
        """Show the room shared by the room-dependent tabs."""
        self._room.setText(room)


class MainWindow(QMainWindow):
    """Coordinate four independent tabs around one selected room."""

    room_ready = Signal()
    input_camera_changed = Signal(str)
    virtual_camera_changed = Signal(bool)

    def __init__(
        self,
        *,
        command_factory: CommandFactory,
        reconstruction_command_factory: ReconstructionCommandFactory | None = None,
        renderer_factory: RendererFactory | None = None,
        live_renderer_factory: RendererFactory | None = None,
        camera_source: InputCameraSource | None = None,
        scene_cache_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        """Create tabs and connect their task-specific signals."""
        super().__init__()
        self._command_factory = command_factory
        self._reconstruction_command_factory = reconstruction_command_factory
        self._resume_job_id: str | None = None
        self._build_session = BuildSession()
        self._runner: JobRunner | None = None
        self._signals = RunnerSignals(self)
        self._signals.event_received.connect(self._handle_job_event)
        self._setup_scene_services(scene_cache_root, data_root)
        self._camera_source = camera_source or InputCameraSource(parent=self)
        self._input_cameras: tuple[InputCamera, ...] = ()
        self._selected_input_camera_id: str | None = None
        self._preview_started = False
        generated_rooms = [scene.display_name for scene in self._generated_scenes.values()]
        self._rooms = [
            *generated_rooms,
            *[room for room in DEFAULT_ROOMS if room not in generated_rooms],
        ]
        self._room_ids = {
            room: (
                self._sample_scene.asset_id
                if room == self._sample_scene.display_name
                else next(
                    (
                        scene.asset_id
                        for scene in self._generated_scenes.values()
                        if scene.display_name == room
                    ),
                    self._room_id(room),
                )
            )
            for room in self._rooms
        }
        self._selected_room = self._rooms[0]

        self.setWindowTitle("Better Backgrounds")
        self.resize(1180, 760)
        self.setMinimumSize(920, 640)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._header = TabHeader()
        root.addWidget(self._header)
        self._tabs = QStackedWidget()
        self._tabs.setObjectName("tabPages")
        tab_layout = self._tabs.layout()
        if not isinstance(tab_layout, QStackedLayout):
            msg = "QStackedWidget did not provide its required stacked layout"
            raise TypeError(msg)
        tab_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        root.addWidget(self._tabs, 1)
        self.setCentralWidget(container)

        actual_renderer_factory = renderer_factory or self._default_renderer_factory
        actual_live_factory = live_renderer_factory
        if actual_live_factory is None:
            actual_live_factory = (
                ScenePreview if renderer_factory is not None else self._default_live_factory
            )
        self._live_preview = actual_live_factory()
        self._show_page = ShowPage(self._rooms, lambda: self._live_preview)
        self._build_page = BuildPage()
        self._adjust_page = AdjustPage(actual_renderer_factory)
        self._compare_page = ComparePage()
        for page in (
            self._show_page,
            self._build_page,
            self._adjust_page,
            self._compare_page,
        ):
            self._tabs.addWidget(page)

        self._header.tab_selected.connect(self.select_tab)
        self._show_page.room_selected.connect(self.select_room)
        self._show_page.input_camera_selected.connect(self.select_input_camera)
        self._show_page.preview_restart_requested.connect(self._restart_preview)
        self._show_page.camera_changed.connect(self.virtual_camera_changed)
        self._camera_source.cameras_changed.connect(self._refresh_input_cameras)
        self._show_page.sample_install_requested.connect(self._install_sample)
        self._show_page.build_requested.connect(self._open_build)
        self._build_page.video_requested.connect(self._choose_video)
        self._build_page.sample_requested.connect(self._use_sample)
        self._build_page.build_requested.connect(self._start_build)
        self._build_page.cancel_requested.connect(self._cancel_build)
        self._build_page.retry_requested.connect(self._retry_build)
        self._adjust_page.viewpoint_saved.connect(self._save_viewpoint)
        self._adjust_page.viewpoint_previewed.connect(self._preview_viewpoint)
        self._adjust_page.matting_changed.connect(self._change_matting)
        self._adjust_page.mirroring_changed.connect(self._change_mirroring)
        self._compare_page.wipe_changed.connect(self._set_compare_wipe)
        camera_state = getattr(self._live_preview, "camera_state_changed", None)
        if camera_state is not None:
            camera_state.connect(self._show_page.set_camera_state)
        diagnostics = getattr(self._live_preview, "diagnostics_changed", None)
        if diagnostics is not None:
            diagnostics.connect(self._record_diagnostics)
        comparison_frame = getattr(self._live_preview, "comparison_frame", None)
        if comparison_frame is not None:
            comparison_frame.connect(self._compare_page.set_live_frame)
        self._adjust_page.set_live_preferences(
            self._live_preferences.matting,
            mirrored=self._live_preferences.mirrored,
        )
        self._show_page.configure_sample(
            self._sample_scene.display_name,
            size=self._sample_scene.expected_size,
            attribution=self._sample_scene.attribution,
            installed=self._asset_installer.is_ready(self._sample_scene),
        )
        self._refresh_input_cameras()
        self.select_room(self._selected_room)
        self.select_tab(0)
        self._start_preview()

    def _setup_scene_services(
        self,
        scene_cache_root: Path | None,
        data_root: Path | None,
    ) -> None:
        """Create application-owned sample cache and viewpoint persistence."""
        self._sample_scene = load_sample_manifest().scenes[0]
        cache_root = scene_cache_root or (
            Path(user_cache_path("Better Backgrounds", "Better Backgrounds")) / "scenes-v1"
        )
        actual_data_root = data_root or Path(
            user_data_path("Better Backgrounds", "Better Backgrounds"),
        )
        self._scene_cache_root = cache_root
        self._data_root = actual_data_root
        self._catalogue = SceneCatalogue(actual_data_root / "scene-catalogue-v1.json")
        self._generated_scenes = {scene.asset_id: scene for scene in self._catalogue.scenes()}
        self._asset_installer = AssetInstaller(cache_root)
        self._asset_resolver = ManagedSceneResolver(
            self._asset_installer,
            [self._sample_scene, *self._generated_scenes.values()],
        )
        self._viewpoints = ViewpointStore(actual_data_root / "viewpoints-v1.json")
        self._input_camera_selection = InputCameraSelectionStore(
            actual_data_root / "input-camera-v1.json",
        )
        self._preferred_input_camera_id = self._input_camera_selection.load()
        self._live_preferences_store = LivePreferencesStore(
            actual_data_root / "live-preferences-v1.json",
        )
        self._live_preferences = self._live_preferences_store.load()
        self._latest_diagnostics: object | None = None
        self._ffprobe = resolved_executable_paths(cache_root.parent / "tools-v1").get("ffprobe")
        self._analysis_signals = AnalysisSignals(self)
        self._analysis_signals.completed.connect(self._analysis_completed)
        self._analysis_signals.failed.connect(self._analysis_failed)
        self._asset_signals = AssetSignals(self)
        self._asset_signals.progressed.connect(self._show_sample_progress)
        self._asset_signals.completed.connect(self._sample_installed)
        self._asset_signals.failed.connect(self._sample_failed)
        self._sample_installing = False

    @property
    def build_session(self) -> BuildSession:
        """Expose the build state for smoke tests and diagnostics."""
        return self._build_session

    @property
    def selected_room(self) -> str:
        """Return the room shared by Show, Adjust, and Compare."""
        return self._selected_room

    @property
    def selected_room_id(self) -> str:
        """Return the stable identifier shared by room-dependent tabs."""
        return self._room_ids[self._selected_room]

    @property
    def selected_input_camera_id(self) -> str | None:
        """Return the effective foreground video-input identifier."""
        return self._selected_input_camera_id

    @property
    def active_tab(self) -> int:
        """Return the visible product-tab index."""
        return self._tabs.currentIndex()

    def start_smoke_build(self) -> None:
        """Run the prepared sample through the successful fake worker."""
        self.select_tab(1)
        self._use_sample()
        self._start_build("success", ReconstructionQuality.BALANCED.value)

    def _default_renderer_factory(self) -> QWidget:
        try:
            from better_backgrounds.desktop.webview import create_renderer_view  # noqa: PLC0415

            return create_renderer_view(self._asset_resolver)
        except ImportError:
            return ScenePreview()

    def _default_live_factory(self) -> QWidget:
        try:
            from better_backgrounds.desktop.webview import (  # noqa: PLC0415
                create_live_renderer_view,
            )

            return create_live_renderer_view(self._asset_resolver)
        except ImportError:
            return ScenePreview()

    @Slot()
    def _refresh_input_cameras(self) -> None:
        """Reconcile hot-plug changes without discarding the user's preference."""
        cameras = self._camera_source.cameras()
        available_ids = {camera.device_id for camera in cameras}
        if self._preferred_input_camera_id in available_ids:
            selected = self._preferred_input_camera_id
        elif self._selected_input_camera_id in available_ids:
            selected = self._selected_input_camera_id
        else:
            default = next((camera for camera in cameras if camera.is_default), None)
            selected = default.device_id if default is not None else None
            if selected is None and cameras:
                selected = cameras[0].device_id
        changed = selected != self._selected_input_camera_id
        self._input_cameras = cameras
        self._selected_input_camera_id = selected
        self._show_page.set_input_cameras(cameras, selected)
        if changed:
            self.input_camera_changed.emit(selected or "")
            if self._preview_started:
                if selected is None:
                    stopper = getattr(self._live_preview, "stop_camera", None)
                    if callable(stopper):
                        stopper()
                    self._show_page.set_camera_state(
                        "lost",
                        "Camera disconnected — reconnect it and restart",
                    )
                else:
                    self._restart_preview()

    @Slot(str)
    def select_input_camera(self, device_id: str) -> None:
        """Persist one explicit input-camera selection and publish it to consumers."""
        if device_id not in {camera.device_id for camera in self._input_cameras}:
            return
        changed = device_id != self._selected_input_camera_id
        self._preferred_input_camera_id = device_id
        self._selected_input_camera_id = device_id
        self._input_camera_selection.save(device_id)
        self._show_page.set_input_cameras(self._input_cameras, device_id)
        if changed:
            self.input_camera_changed.emit(device_id)
            if self._preview_started:
                self._restart_preview()

    def _start_preview(self) -> None:
        """Start the local preview independently of virtual-camera output."""
        if self._selected_input_camera_id is None:
            self._show_page.set_camera_state("error", "No camera is available")
            return
        self._preview_started = True
        self._show_page.set_camera_state("starting", "Requesting camera permission…")
        starter = getattr(self._live_preview, "start_camera", None)
        if callable(starter):
            starter(self._selected_camera_label(), mirrored=self._live_preferences.mirrored)

    def _restart_preview(self) -> None:
        """Apply a device change without creating a second preview stream."""
        stopper = getattr(self._live_preview, "stop_camera", None)
        starter = getattr(self._live_preview, "start_camera", None)
        if callable(stopper):
            stopper()
        if self._selected_input_camera_id is None:
            self._show_page.set_camera_state("error", "No camera is available")
            return
        self._preview_started = True
        if callable(starter):
            self._show_page.set_camera_state("starting", "Restarting selected camera…")
            starter(self._selected_camera_label(), mirrored=self._live_preferences.mirrored)

    def _selected_camera_label(self) -> str:
        selected = next(
            (
                camera
                for camera in self._input_cameras
                if camera.device_id == self._selected_input_camera_id
            ),
            None,
        )
        return selected.description if selected is not None else ""

    @Slot(int)
    def select_tab(self, index: int) -> None:
        """Open any product tab without workflow gating."""
        if 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)
            self._header.set_active_tab(index)
            presentation = getattr(self._live_preview, "set_presentation", None)
            if index == COMPARE_TAB:
                if callable(presentation):
                    presentation("compare", DEFAULT_WIPE)
            elif callable(presentation):
                presentation("show", DEFAULT_WIPE)

    @Slot(str)
    def select_room(self, room: str) -> None:
        """Share the selected room across room-dependent tabs."""
        if room not in self._rooms:
            return
        self._selected_room = room
        room_id = self._room_ids[room]
        self._header.set_room(room)
        self._show_page.set_room(room)
        scene = self._scene_for_room(room)
        installed = scene is not None and self._asset_installer.is_ready(scene)
        self._adjust_page.set_room(
            room_id,
            scene,
            installed=installed,
            viewpoint=self._viewpoints.load(room_id),
        )
        if installed and scene is not None:
            live_viewpoint = self._viewpoints.load(room_id) or scene.default_viewpoint
            live_viewpoint = live_viewpoint.model_copy(
                update={"scene_transform": scene.default_viewpoint.scene_transform},
            )
            scene_setter = getattr(self._live_preview, "set_scene", None)
            if callable(scene_setter):
                scene_setter(scene, live_viewpoint)
        else:
            scene_clearer = getattr(self._live_preview, "clear_scene", None)
            if callable(scene_clearer):
                scene_clearer()
        preview = None
        if installed and scene is not None and scene.preview is not None:
            preview = self._asset_installer.resource_path(scene, scene.preview)
        self._show_page.set_preview_image(preview)
        self._compare_page.set_room(room)

    @Slot()
    def _install_sample(self) -> None:
        if self._sample_installing or self._asset_installer.is_ready(self._sample_scene):
            return
        self._sample_installing = True
        self._show_page.set_sample_downloading(0, self._sample_scene.expected_size)

        def install() -> None:
            try:
                self._asset_installer.install(
                    self._sample_scene,
                    self._asset_signals.progressed.emit,
                )
            except (OSError, ValueError) as error:
                self._asset_signals.failed.emit(str(error)[:300])
            else:
                self._asset_signals.completed.emit()

        threading.Thread(target=install, name="sample-scene-install", daemon=True).start()

    @Slot(int, int)
    def _show_sample_progress(self, completed: int, total: int) -> None:
        self._show_page.set_sample_downloading(completed, total)

    @Slot()
    def _sample_installed(self) -> None:
        self._sample_installing = False
        self._show_page.set_sample_ready(ready=True)
        if self._selected_room == self._sample_scene.display_name:
            self.select_room(self._selected_room)

    @Slot(str)
    def _sample_failed(self, message: str) -> None:
        self._sample_installing = False
        self._show_page.set_sample_error(f"Sample download failed: {message}")

    @Slot(str, object)
    def _save_viewpoint(self, room_id: str, viewpoint: object) -> None:
        if isinstance(viewpoint, Viewpoint):
            self._viewpoints.save(room_id, viewpoint)

    @Slot(object)
    def _preview_viewpoint(self, viewpoint: object) -> None:
        """Keep Show and Compare on Adjust's current room camera."""
        if isinstance(viewpoint, Viewpoint):
            setter = getattr(self._live_preview, "set_viewpoint", None)
            if callable(setter):
                setter(viewpoint)

    @Slot(object)
    def _change_matting(self, settings: object) -> None:
        """Persist and apply bounded mask refinement without restarting capture."""
        if not isinstance(settings, MattingSettings):
            return
        self._live_preferences = self._live_preferences.model_copy(update={"matting": settings})
        self._live_preferences_store.save(self._live_preferences)
        setter = getattr(self._live_preview, "set_matting_settings", None)
        if callable(setter):
            setter(settings.worker_payload())

    @Slot(bool)
    def _change_mirroring(self, mirrored: bool) -> None:  # noqa: FBT001
        """Persist and apply foreground-only mirroring."""
        self._live_preferences = self._live_preferences.model_copy(update={"mirrored": mirrored})
        self._live_preferences_store.save(self._live_preferences)
        setter = getattr(self._live_preview, "set_mirroring", None)
        if callable(setter):
            setter(mirrored=mirrored)

    @Slot(int)
    def _set_compare_wipe(self, value: int) -> None:
        setter = getattr(self._live_preview, "set_presentation", None)
        if callable(setter):
            setter("compare", value)

    @Slot(object)
    def _record_diagnostics(self, diagnostics: object) -> None:
        self._latest_diagnostics = diagnostics

    @Slot()
    def _open_build(self) -> None:
        if not isinstance(self._build_session.state, RunningBuild):
            self._build_page.show_upload()
        self.select_tab(1)

    @Slot()
    def _choose_video(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose a room video",
            "",
            "Room videos (*.mp4 *.mov);;All files (*)",
        )
        if path:
            self._select_video(
                VideoSelection(display_name=Path(path).name, source_path=Path(path)),
            )

    @Slot()
    def _use_sample(self) -> None:
        self._select_video(VideoSelection("Prepared loft sample", None, sample=True))

    def _select_video(self, selection: VideoSelection) -> None:
        self._build_session.select_video(selection)
        self._build_page.show_review(selection)
        if selection.sample:
            self._build_page.set_prepared_sample_ready()
            return
        self._build_page.set_analysis_pending()
        ffprobe = self._ffprobe
        if selection.source_path is None or ffprobe is None:
            self._build_page.set_analysis_error(
                "Video analysis requires the pinned FFmpeg tools. Run setup --tools and doctor."
            )
            return
        source = selection.source_path.resolve()

        def analyse() -> None:
            try:
                diagnostics = analyse_video_file(source, ffprobe)
            except (OSError, TypeError, ValueError) as error:
                self._analysis_signals.failed.emit(str(source), str(error)[:500])
            else:
                self._analysis_signals.completed.emit(diagnostics)

        threading.Thread(target=analyse, name="capture-analysis", daemon=True).start()

    @Slot(object)
    def _analysis_completed(self, result: object) -> None:
        if not isinstance(result, CaptureDiagnostics) or not self._review_matches(
            result.probe.path
        ):
            return
        self._build_page.set_capture_diagnostics(result)

    @Slot(str, str)
    def _analysis_failed(self, source: str, message: str) -> None:
        if self._review_matches(Path(source)):
            self._build_page.set_analysis_error(message)

    def _review_matches(self, source: Path) -> bool:
        state = self._build_session.state
        return bool(
            isinstance(state, ReviewBuild)
            and state.selection.source_path is not None
            and state.selection.source_path.resolve() == source.resolve()
        )

    @Slot(str, str)
    def _start_build(self, outcome: str, quality_name: str) -> None:
        state = self._build_session.state
        if not isinstance(state, ReviewBuild):
            return
        job_id = self._resume_job_id or uuid4().hex
        resume = self._resume_job_id is not None
        self._resume_job_id = None
        self._build_session.start(job_id)
        self._build_page.reset_progress()
        runner = JobRunner(self._signals.event_received.emit)
        self._runner = runner
        quality = ReconstructionQuality(quality_name)
        if (
            state.selection.source_path is not None
            and self._reconstruction_command_factory is not None
        ):
            command = self._reconstruction_command_factory(
                job_id,
                state.selection.source_path,
                resume,
                quality,
            )
        else:
            command = self._command_factory(job_id, outcome)
        runner.start(command, job_id=job_id)

    @Slot()
    def _cancel_build(self) -> None:
        state = self._build_session.state
        if isinstance(state, RunningBuild) and self._runner is not None:
            self._runner.cancel(state.job_id)

    @Slot()
    def _retry_build(self) -> None:
        failed_state = self._build_session.state
        if isinstance(failed_state, FailedBuild) and self._runner is not None:
            self._resume_job_id = self._runner.job_id
        state = self._build_session.retry()
        self._build_page.show_review(state.selection)

    @Slot(object)
    def _handle_job_event(self, event: object) -> None:
        if not isinstance(
            event,
            ProgressEvent | WarningEvent | ResultEvent | ErrorEvent | CancelledEvent,
        ):
            return
        previous_state = self._build_session.state
        if not self._build_session.apply(event):
            return
        state = self._build_session.state
        if isinstance(event, ProgressEvent | WarningEvent) and isinstance(state, RunningBuild):
            self._build_page.set_progress(state.stage, state.progress, state.message)
        elif isinstance(state, FailedBuild):
            self._build_page.set_failed(state.message, state.recovery_action)
        elif isinstance(state, ReviewBuild):
            if isinstance(previous_state, RunningBuild) and isinstance(event, CancelledEvent):
                self._resume_job_id = previous_state.job_id
            self._build_page.show_review(state.selection)
        elif isinstance(state, CompletedBuild):
            scene = self._catalogue.find(state.scene_id)
            if scene is not None and self._asset_installer.is_ready(scene):
                self._generated_scenes[scene.asset_id] = scene
                self._asset_resolver.register(scene)
            room_name = (
                scene.display_name if scene is not None else self._room_name_for(state.selection)
            )
            if room_name not in self._rooms:
                self._rooms.insert(0, room_name)
            self._room_ids[room_name] = (
                scene.asset_id if scene is not None else self._room_id(room_name)
            )
            self._show_page.set_rooms(self._rooms, room_name)
            self.select_room(room_name)
            self._build_page.set_completed(room_name)
            self.room_ready.emit()

    def _scene_for_room(self, room: str) -> SceneReference | None:
        if room == self._sample_scene.display_name:
            return self._sample_scene
        room_id = self._room_ids.get(room)
        return self._generated_scenes.get(room_id) if room_id is not None else None

    @staticmethod
    def _room_name_for(selection: VideoSelection) -> str:
        if selection.sample:
            return "Loft — North Window"
        return Path(selection.display_name).stem.replace("_", " ").strip().title()

    @staticmethod
    def _room_id(room_name: str) -> str:
        value = "".join(
            character.lower() if character.isalnum() else "-" for character in room_name
        )
        return "-".join(part for part in value.split("-") if part)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Clean up the complete active process tree before closing."""
        stopper = getattr(self._live_preview, "stop_camera", None)
        if callable(stopper):
            stopper()
        if self._runner is not None:
            self._runner.close()
        event.accept()


def development_worker_command(job_id: str, outcome: str) -> list[str]:
    """Run the worker through the active development interpreter."""
    return [
        sys.executable,
        "-m",
        "better_backgrounds.cli",
        "fake-job",
        "--job-id",
        job_id,
        "--outcome",
        outcome,
    ]
