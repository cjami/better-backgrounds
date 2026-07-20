"""Foreground camera and retained live-preview controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from better_backgrounds.desktop.camera import (
    InputCamera,
    InputCameraSelectionStore,
    InputCameraSource,
)
from better_backgrounds.harmonization import HarmonizationSettings
from better_backgrounds.matting import LivePreferencesStore

if TYPE_CHECKING:
    from pathlib import Path

    from PySide6.QtWidgets import QWidget

    from better_backgrounds.desktop.pages import ShowPage


class LivePreviewController(QObject):
    """Own camera selection, preview lifecycle, preferences, and diagnostics."""

    input_camera_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        show_page: ShowPage,
        live_preview: QWidget,
        camera_source: InputCameraSource,
        data_root: Path,
    ) -> None:
        """Connect retained preview services to their desktop controls."""
        super().__init__(parent)
        self._show_page = show_page
        self._live_preview = live_preview
        self._camera_source = camera_source
        self._input_camera_selection = InputCameraSelectionStore(
            data_root / "input-camera-v1.json",
        )
        self._preferred_input_camera_id = self._input_camera_selection.load()
        self._live_preferences_store = LivePreferencesStore(
            data_root / "live-preferences-v1.json",
        )
        self._live_preferences = self._live_preferences_store.load()
        self._input_cameras: tuple[InputCamera, ...] = ()
        self._selected_input_camera_id: str | None = None
        self._preview_started = False
        self._latest_diagnostics: object | None = None
        show_page.input_camera_selected.connect(self.select_input_camera)
        show_page.preview_restart_requested.connect(self._restart_preview)
        show_page.seed_confirmed.connect(self._confirm_person_seed)
        show_page.seed_retry_requested.connect(self._retry_person_seed)
        show_page.reseed_requested.connect(self._reselect_person)
        show_page.person_candidate_selected.connect(self._select_person_candidate)
        camera_source.cameras_changed.connect(self._refresh_input_cameras)
        camera_state = getattr(live_preview, "camera_state_changed", None)
        if camera_state is not None:
            camera_state.connect(show_page.set_camera_state)
        diagnostics = getattr(live_preview, "diagnostics_changed", None)
        if diagnostics is not None:
            diagnostics.connect(self._record_diagnostics)
        candidates = getattr(live_preview, "person_candidates_changed", None)
        if candidates is not None:
            candidates.connect(show_page.set_person_candidates)
        configure_data = getattr(live_preview, "configure_data_root", None)
        if callable(configure_data):
            configure_data(data_root)

    @property
    def selected_camera_id(self) -> str | None:
        """Return the effective foreground camera identifier."""
        return self._selected_input_camera_id

    @property
    def mirrored(self) -> bool:
        """Return the persisted foreground mirroring preference."""
        return self._live_preferences.mirrored

    def start(self) -> None:
        """Discover cameras and start the retained local preview."""
        preparer = getattr(self._live_preview, "prepare_matting", None)
        if callable(preparer):
            preparer()
        self._refresh_input_cameras()
        self._start_preview()

    def set_resource_active(self, active: bool) -> None:  # noqa: FBT001
        """Suspend hidden live work while Adjust owns the graphics device."""
        setter = getattr(self._live_preview, "set_resource_active", None)
        if callable(setter):
            setter(active)

    def shutdown(self) -> None:
        """Release the retained camera preview."""
        stopper = getattr(self._live_preview, "stop_camera", None)
        if callable(stopper):
            stopper()

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
                    stopper = getattr(self._live_preview, "stop_capture", None)
                    if not callable(stopper):
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
            starter(self._selected_input_camera_id, mirrored=self._live_preferences.mirrored)

    def _restart_preview(self) -> None:
        """Apply a device change without creating a second preview stream."""
        stopper = getattr(self._live_preview, "stop_capture", None)
        if not callable(stopper):
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
            starter(self._selected_input_camera_id, mirrored=self._live_preferences.mirrored)

    @Slot()
    def _confirm_person_seed(self) -> None:
        confirmer = getattr(self._live_preview, "confirm_seed", None)
        if callable(confirmer):
            confirmer()

    @Slot()
    def _retry_person_seed(self) -> None:
        retry = getattr(self._live_preview, "retry_seed", None)
        if callable(retry):
            retry()

    @Slot()
    def _reselect_person(self) -> None:
        reseed = getattr(self._live_preview, "reselect_person", None)
        if callable(reseed):
            reseed()

    @Slot(int)
    def _select_person_candidate(self, candidate_id: int) -> None:
        selector = getattr(self._live_preview, "select_person_candidate", None)
        if callable(selector):
            selector(candidate_id)

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

    @Slot(bool)
    def change_mirroring(self, mirrored: bool) -> None:  # noqa: FBT001
        """Persist and apply foreground-only mirroring."""
        self._live_preferences = self._live_preferences.model_copy(update={"mirrored": mirrored})
        self._live_preferences_store.save(self._live_preferences)
        setter = getattr(self._live_preview, "set_mirroring", None)
        if callable(setter):
            setter(mirrored=mirrored)

    @Slot(object)
    def change_harmonization(self, settings: object) -> None:
        """Apply one room-scoped experimental appearance configuration."""
        if not isinstance(settings, HarmonizationSettings):
            return
        setter = getattr(self._live_preview, "set_harmonization", None)
        if callable(setter):
            setter(settings)

    @Slot(object)
    def _record_diagnostics(self, diagnostics: object) -> None:
        self._latest_diagnostics = diagnostics
