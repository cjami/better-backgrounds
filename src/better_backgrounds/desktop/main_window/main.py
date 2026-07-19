"""Desktop window assembly and room/tab coordination."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_cache_path, user_data_path
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.camera import InputCameraSource
from better_backgrounds.desktop.main_window.build_controller import BuildController
from better_backgrounds.desktop.main_window.header import TabHeader
from better_backgrounds.desktop.main_window.live_preview_controller import LivePreviewController
from better_backgrounds.desktop.pages import AdjustPage, BuildPage, ComparePage, ShowPage
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.reconstruction.sharp import SharpCheckpointInstaller
from better_backgrounds.scene import SceneLibrary, Viewpoint

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent

CommandFactory = Callable[[str, str], Sequence[str]]
SharpCommandFactory = Callable[[str, Path, str, str], Sequence[str]]
SharpPrepareCommandFactory = Callable[[str], Sequence[str]]
SplatCommandFactory = Callable[[str, Path], Sequence[str]]
RendererFactory = Callable[[], QWidget]

COMPARE_TAB = 3
ADJUST_TAB = 2
DEFAULT_WIPE = 52


class AssetSignals(QObject):
    """Marshal sample-install callbacks onto the Qt main thread."""

    progressed = Signal(int, int)
    completed = Signal()
    failed = Signal(str)


class MainWindow(QMainWindow):
    """Assemble feature views and coordinate shared room and tab selection."""

    room_ready = Signal()
    input_camera_changed = Signal(str)
    virtual_camera_changed = Signal(bool)

    def __init__(
        self,
        *,
        command_factory: CommandFactory,
        sharp_command_factory: SharpCommandFactory | None = None,
        sharp_prepare_command_factory: SharpPrepareCommandFactory | None = None,
        splat_command_factory: SplatCommandFactory | None = None,
        renderer_factory: RendererFactory | None = None,
        live_renderer_factory: RendererFactory | None = None,
        camera_source: InputCameraSource | None = None,
        scene_cache_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        """Create feature services, views, and their narrow controllers."""
        super().__init__()
        cache_root = scene_cache_root or (
            Path(user_cache_path("Better Backgrounds", "Better Backgrounds")) / "scenes-v1"
        )
        actual_data_root = data_root or Path(
            user_data_path("Better Backgrounds", "Better Backgrounds"),
        )
        self._library = SceneLibrary(cache_root, actual_data_root)
        self._rooms = self._library.rooms
        self._room_ids = self._library.room_ids
        selected_room_id = self._library.selection.load()
        self._selected_room = next(
            (room for room in self._rooms if self._room_ids[room] == selected_room_id),
            self._rooms[0],
        )
        self._sample_installing = False
        self._asset_signals = AssetSignals(self)
        self._asset_signals.progressed.connect(self._show_sample_progress)
        self._asset_signals.completed.connect(self._sample_installed)
        self._asset_signals.failed.connect(self._sample_failed)
        self._create_shell()

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
        self._adjust_page.set_resource_active(False)
        self._compare_page = ComparePage()
        for page in (
            self._show_page,
            self._build_page,
            self._adjust_page,
            self._compare_page,
        ):
            self._tabs.addWidget(page)

        source = camera_source or InputCameraSource(parent=self)
        self._live_controller = LivePreviewController(
            self,
            self._show_page,
            self._compare_page,
            self._live_preview,
            source,
            actual_data_root,
        )
        checkpoint = SharpCheckpointInstaller(cache_root.parent / "models-v1" / "sharp")
        self._build_controller = BuildController(
            self,
            self._build_page,
            self._library,
            command_factory,
            sharp_command_factory,
            sharp_prepare_command_factory,
            checkpoint,
            lambda: self.select_tab(1),
            splat_factory=splat_command_factory,
        )
        self._connect_views()
        self._show_page.configure_sample(
            self._library.sample_scene.display_name,
            size=self._library.sample_scene.expected_size,
            attribution=self._library.sample_scene.attribution,
            installed=self._library.assets.is_ready(self._library.sample_scene),
        )
        self._adjust_page.set_live_preferences(mirrored=self._live_controller.mirrored)
        self.select_room(self._selected_room)
        self.select_tab(0)
        self._live_controller.start()

    def _create_shell(self) -> None:
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

    def _connect_views(self) -> None:
        self._header.tab_selected.connect(self.select_tab)
        self._show_page.room_selected.connect(self.select_room)
        self._show_page.camera_changed.connect(self.virtual_camera_changed)
        self._show_page.sample_install_requested.connect(self._install_sample)
        self._show_page.build_requested.connect(self._build_controller.open)
        self._adjust_page.viewpoint_saved.connect(self._save_viewpoint)
        self._adjust_page.viewpoint_previewed.connect(self._preview_viewpoint)
        self._adjust_page.snapshot_generated.connect(self._save_snapshot)
        self._adjust_page.mirroring_changed.connect(self._live_controller.change_mirroring)
        self._adjust_page.harmonization_changed.connect(
            self._live_controller.change_harmonization,
        )
        self._compare_page.wipe_changed.connect(self._live_controller.set_compare_wipe)
        self._live_controller.input_camera_changed.connect(self.input_camera_changed)
        self._build_controller.scene_completed.connect(self._scene_completed)

    @property
    def build_session(self):  # noqa: ANN201
        """Expose the build state for smoke tests and diagnostics."""
        return self._build_controller.session

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
        return self._live_controller.selected_camera_id

    @property
    def active_tab(self) -> int:
        """Return the visible product-tab index."""
        return self._tabs.currentIndex()

    def start_smoke_build(self) -> None:
        """Start the deterministic packaged-worker smoke build."""
        self._build_controller.start_smoke()

    def _default_renderer_factory(self) -> QWidget:
        try:
            from better_backgrounds.desktop.webview import create_renderer_view  # noqa: PLC0415

            return create_renderer_view(self._library.resolver)
        except ImportError:
            return ScenePreview()

    def _default_live_factory(self) -> QWidget:
        try:
            from better_backgrounds.desktop.live_preview import (  # noqa: PLC0415
                create_native_live_view,
            )

            return create_native_live_view(self._library.resolver)
        except ImportError:
            return ScenePreview()

    @Slot(int)
    def select_tab(self, index: int) -> None:
        """Open any product tab without workflow gating."""
        if 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)
            self._header.set_active_tab(index)
            self._adjust_page.set_resource_active(index == ADJUST_TAB)
            self._live_controller.set_resource_active(index != ADJUST_TAB)
            mode = "compare" if index == COMPARE_TAB else "show"
            self._live_controller.set_presentation(mode, DEFAULT_WIPE)

    @Slot(str)
    def select_room(self, room: str) -> None:
        """Share the selected room across room-dependent tabs."""
        if room not in self._rooms:
            return
        self._selected_room = room
        room_id = self._room_ids[room]
        self._library.selection.save(room_id)
        self._header.set_room(room)
        self._show_page.set_room(room)
        scene = self._library.scene_for_room(room)
        installed = scene is not None and self._library.assets.is_ready(scene)
        viewpoint = self._library.viewpoints.load(room_id)
        self._adjust_page.set_room(room_id, scene, installed=installed, viewpoint=viewpoint)
        output_viewpoint = viewpoint or (
            scene.default_viewpoint if scene is not None else Viewpoint()
        )
        self._show_page.set_output_aspect_ratio(output_viewpoint.aspect_ratio)
        aspect_setter = getattr(self._live_preview, "set_output_aspect_ratio", None)
        if callable(aspect_setter):
            aspect_setter(output_viewpoint.aspect_ratio)
        cached = None
        if installed and scene is not None:
            live_viewpoint = viewpoint or scene.default_viewpoint
            live_viewpoint = live_viewpoint.model_copy(
                update={"scene_transform": scene.default_viewpoint.scene_transform},
            )
            cached = self._library.snapshots.load(scene, live_viewpoint)
            if cached is not None:
                setter = getattr(self._live_preview, "set_scene_snapshot", None)
                if not callable(setter) or not setter(cached.background):
                    cached = None
        else:
            clearer = getattr(self._live_preview, "clear_scene", None)
            if callable(clearer):
                clearer()
        preview = None
        if installed and scene is not None and scene.preview is not None:
            preview = self._library.assets.resource_path(scene, scene.preview)
        self._show_page.set_preview_image(preview if cached is None else None)
        self._compare_page.set_room(room)

    @Slot()
    def _install_sample(self) -> None:
        scene = self._library.sample_scene
        if self._sample_installing or self._library.assets.is_ready(scene):
            return
        self._sample_installing = True
        self._show_page.set_sample_downloading(0, scene.expected_size)

        def install() -> None:
            try:
                self._library.assets.install(scene, self._asset_signals.progressed.emit)
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
        if self._selected_room == self._library.sample_scene.display_name:
            self.select_room(self._selected_room)

    @Slot(str)
    def _sample_failed(self, message: str) -> None:
        self._sample_installing = False
        self._show_page.set_sample_error(f"Sample download failed: {message}")

    @Slot(str, object)
    def _save_viewpoint(self, room_id: str, viewpoint: object) -> None:
        if isinstance(viewpoint, Viewpoint):
            self._library.viewpoints.save(room_id, viewpoint)

    @Slot(object)
    def _preview_viewpoint(self, viewpoint: object) -> None:
        if isinstance(viewpoint, Viewpoint):
            self._show_page.set_output_aspect_ratio(viewpoint.aspect_ratio)
            setter = getattr(self._live_preview, "set_output_aspect_ratio", None)
            if callable(setter):
                setter(viewpoint.aspect_ratio)

    @Slot(str, str, object, object)
    def _save_snapshot(
        self,
        room_id: str,
        asset_id: str,
        viewpoint: object,
        background: object,
    ) -> None:
        if not isinstance(viewpoint, Viewpoint) or not isinstance(background, bytes):
            return
        scene = self._library.scene_for_id(asset_id)
        if scene is None or room_id != scene.asset_id:
            return
        try:
            snapshot = self._library.snapshots.save(
                scene,
                viewpoint,
                background,
            )
        except OSError, ValueError:
            return
        if self.selected_room_id != room_id:
            return
        setter = getattr(self._live_preview, "set_scene_snapshot", None)
        if callable(setter):
            setter(snapshot.background)

    @Slot(str, str)
    def _scene_completed(self, scene_id: str, room_name: str) -> None:
        self._library.viewpoints.delete(scene_id)
        self._adjust_page.discard_viewpoint(scene_id)
        self._show_page.set_rooms(self._rooms, room_name)
        self.select_room(room_name)
        self.room_ready.emit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Release camera and worker resources before closing."""
        self._live_controller.shutdown()
        self._build_controller.shutdown()
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
