"""Headless tests for the Python-owned desktop boundary."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget

from better_backgrounds.build_session import IdleBuild
from better_backgrounds.desktop.app import packaged_worker_command
from better_backgrounds.desktop.bridge import RendererBridge
from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.main_window import MainWindow
from better_backgrounds.desktop.preview import ScenePreview
from better_backgrounds.desktop.webview import navigation_is_allowed

if TYPE_CHECKING:
    import pytest

TAB_COUNT = 4
BUILD_TAB = 1
COMPARE_TAB = 3


def application() -> QApplication:
    """Return the one application allowed by Qt per process."""
    existing = cast("QApplication | None", QApplication.instance())
    return existing or QApplication([])


def test_main_window_contains_four_independent_product_tabs() -> None:
    """Construct the complete tabbed shell without native media behavior."""
    app = application()
    window = MainWindow(
        command_factory=lambda _job_id, _outcome: [],
        renderer_factory=ScenePreview,
    )

    stack = window.findChild(QStackedWidget, "tabPages")
    tabs = window.findChildren(QPushButton, "tab")

    assert app.applicationName() is not None
    assert stack is not None
    assert stack.count() == TAB_COUNT
    assert [tab.text() for tab in tabs] == ["Show", "Build", "Adjust", "Compare"]
    assert all(tab.isEnabled() for tab in tabs)
    assert window.active_tab == 0
    assert isinstance(window.build_session.state, IdleBuild)
    window.close()


def test_application_icon_loads_from_package_data() -> None:
    """Keep the shared vector mark available to source and packaged builds."""
    application()

    assert not application_icon().isNull()


def test_tabs_can_be_opened_in_any_order() -> None:
    """Keep navigation separate from build-session state."""
    window = MainWindow(command_factory=lambda _job_id, _outcome: [], renderer_factory=ScenePreview)

    window.select_tab(COMPARE_TAB)
    assert window.active_tab == COMPARE_TAB
    window.select_tab(BUILD_TAB)
    assert window.active_tab == BUILD_TAB
    assert isinstance(window.build_session.state, IdleBuild)
    window.close()


def test_show_tab_has_a_clear_camera_toggle() -> None:
    """Expose explicit start and stop actions for the virtual camera."""
    window = MainWindow(command_factory=lambda _job_id, _outcome: [], renderer_factory=ScenePreview)
    camera = window.findChild(QPushButton, "cameraToggle")

    assert camera is not None
    assert camera.text() == "●  Start virtual camera"
    camera.click()
    assert camera.text() == "■  Stop virtual camera"
    camera.click()
    assert camera.text() == "●  Start virtual camera"
    window.close()


def test_renderer_bridge_rejects_invalid_viewpoint() -> None:
    """Expose a task-specific validated method instead of general Python access."""
    bridge = RendererBridge()

    assert not bridge.submit_viewpoint('{"field_of_view":200}')
    assert bridge.submit_viewpoint(
        '{"field_of_view":42,"horizon":-1.5,"subject_depth":2.4,"focus_depth":2.6}',
    )


def test_navigation_is_restricted_to_synthetic_origin() -> None:
    """Block filesystem and network navigation from the embedded renderer."""
    assert navigation_is_allowed(QUrl("https://app.better-backgrounds.invalid/viewer.html"))
    assert not navigation_is_allowed(QUrl("file:///private/room.sog"))
    assert not navigation_is_allowed(QUrl("https://example.com/"))


def test_frozen_application_reuses_its_executable_for_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the fake process contract operational in a standalone package."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = packaged_worker_command("job-1", "success")

    assert command[:2] == [str(Path(sys.argv[0]).resolve()), "--fake-worker"]
