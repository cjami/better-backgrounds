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
from better_backgrounds.job_runner import JobRunner
from better_backgrounds.protocol import (
    CancelledEvent,
    ErrorEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)
from better_backgrounds.scene import (
    AssetInstaller,
    ManagedSceneResolver,
    Viewpoint,
    ViewpointStore,
    load_sample_manifest,
)

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent

CommandFactory = Callable[[str, str], Sequence[str]]
RendererFactory = Callable[[], QWidget]

TAB_NAMES = ("Show", "Build", "Adjust", "Compare")
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

    def __init__(
        self,
        *,
        command_factory: CommandFactory,
        renderer_factory: RendererFactory | None = None,
        scene_cache_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        """Create tabs and connect their task-specific signals."""
        super().__init__()
        self._command_factory = command_factory
        self._build_session = BuildSession()
        self._runner: JobRunner | None = None
        self._signals = RunnerSignals(self)
        self._signals.event_received.connect(self._handle_job_event)
        self._setup_scene_services(scene_cache_root, data_root)
        self._rooms = list(DEFAULT_ROOMS)
        self._room_ids = {
            room: (
                self._sample_scene.asset_id
                if room == self._sample_scene.display_name
                else self._room_id(room)
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
        root.addWidget(self._tabs, 1)
        self.setCentralWidget(container)

        actual_renderer_factory = renderer_factory or self._default_renderer_factory
        self._show_page = ShowPage(self._rooms, ScenePreview)
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
        self._show_page.sample_install_requested.connect(self._install_sample)
        self._show_page.build_requested.connect(self._open_build)
        self._build_page.video_requested.connect(self._choose_video)
        self._build_page.sample_requested.connect(self._use_sample)
        self._build_page.build_requested.connect(self._start_build)
        self._build_page.cancel_requested.connect(self._cancel_build)
        self._build_page.retry_requested.connect(self._retry_build)
        self._adjust_page.viewpoint_saved.connect(self._save_viewpoint)
        self._show_page.configure_sample(
            self._sample_scene.display_name,
            size=self._sample_scene.expected_size,
            attribution=self._sample_scene.attribution,
            installed=self._asset_installer.is_ready(self._sample_scene),
        )
        self.select_room(self._selected_room)
        self.select_tab(0)

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
        self._asset_installer = AssetInstaller(cache_root)
        self._asset_resolver = ManagedSceneResolver(self._asset_installer, [self._sample_scene])
        self._viewpoints = ViewpointStore(actual_data_root / "viewpoints-v1.json")
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
    def active_tab(self) -> int:
        """Return the visible product-tab index."""
        return self._tabs.currentIndex()

    def start_smoke_build(self) -> None:
        """Run the prepared sample through the successful fake worker."""
        self.select_tab(1)
        self._use_sample()
        self._start_build("success")

    def _default_renderer_factory(self) -> QWidget:
        try:
            from better_backgrounds.desktop.webview import create_renderer_view  # noqa: PLC0415

            return create_renderer_view(self._asset_resolver)
        except ImportError:
            return ScenePreview()

    @Slot(int)
    def select_tab(self, index: int) -> None:
        """Open any product tab without workflow gating."""
        if 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)
            self._header.set_active_tab(index)

    @Slot(str)
    def select_room(self, room: str) -> None:
        """Share the selected room across room-dependent tabs."""
        if room not in self._rooms:
            return
        self._selected_room = room
        room_id = self._room_ids[room]
        self._header.set_room(room)
        self._show_page.set_room(room)
        scene = self._sample_scene if room == self._sample_scene.display_name else None
        installed = scene is not None and self._asset_installer.is_ready(scene)
        self._adjust_page.set_room(
            room_id,
            scene,
            installed=installed,
            viewpoint=self._viewpoints.load(room_id),
        )
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

    @Slot(str)
    def _start_build(self, outcome: str) -> None:
        job_id = uuid4().hex
        self._build_session.start(job_id)
        self._build_page.reset_progress()
        runner = JobRunner(self._signals.event_received.emit)
        self._runner = runner
        runner.start(self._command_factory(job_id, outcome), job_id=job_id)

    @Slot()
    def _cancel_build(self) -> None:
        state = self._build_session.state
        if isinstance(state, RunningBuild) and self._runner is not None:
            self._runner.cancel(state.job_id)

    @Slot()
    def _retry_build(self) -> None:
        state = self._build_session.retry()
        self._build_page.show_review(state.selection)

    @Slot(object)
    def _handle_job_event(self, event: object) -> None:
        if not isinstance(
            event,
            ProgressEvent | WarningEvent | ResultEvent | ErrorEvent | CancelledEvent,
        ):
            return
        if not self._build_session.apply(event):
            return
        state = self._build_session.state
        if isinstance(event, ProgressEvent | WarningEvent) and isinstance(state, RunningBuild):
            self._build_page.set_progress(state.stage, state.progress, state.message)
        elif isinstance(state, FailedBuild):
            self._build_page.set_failed(state.message, state.recovery_action)
        elif isinstance(state, ReviewBuild):
            self._build_page.show_review(state.selection)
        elif isinstance(state, CompletedBuild):
            room_name = self._room_name_for(state.selection)
            if room_name not in self._rooms:
                self._rooms.insert(0, room_name)
                self._room_ids[room_name] = self._room_id(room_name)
            self._show_page.set_rooms(self._rooms, room_name)
            self.select_room(room_name)
            self._build_page.set_completed(room_name)
            self.room_ready.emit()

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
