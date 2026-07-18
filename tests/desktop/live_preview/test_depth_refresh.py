"""Focused tests for debounced and revision-safe scene snapshots."""

from typing import cast

from PySide6.QtCore import QBuffer, QIODevice, QRect, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.live_preview.preview import (
    BACKGROUND_REFRESH_DEBOUNCE_MS,
    NativeLivePreview,
)
from better_backgrounds.scene import Viewpoint


class SnapshotRenderer(QWidget):
    """Record hidden-renderer requests and expose deterministic capture pixels."""

    scene_progressed = Signal(int, int)
    snapshot_ready = Signal(str, int, str, str)

    def __init__(self) -> None:
        """Create empty request logs and a non-uniform capture image."""
        super().__init__()
        self.viewpoints: list[Viewpoint] = []
        self._image = QImage(8, 8, QImage.Format.Format_RGB888)
        self._image.fill(QColor("#202020"))
        self._image.setPixelColor(0, 0, QColor("#f0f0f0"))
        buffer = QBuffer()
        assert buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        assert self._image.save(buffer, "PNG")  # ty: ignore[no-matching-overload]
        encoded = cast("bytes", buffer.data().toBase64().data())
        self.payload = encoded.decode("ascii")

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Record a debounced viewpoint request."""
        self.viewpoints.append(viewpoint)

    def grab(self, /, rectangle: QRect = QRect()) -> QPixmap:  # noqa: ARG002, B008
        """Return deterministic fallback pixels without a WebEngine surface."""
        return QPixmap.fromImage(self._image)


def application() -> QApplication:
    """Return the process-wide Qt application."""
    existing = QApplication.instance()
    return cast("QApplication", existing) if existing is not None else QApplication([])


def test_live_viewpoint_refresh_keeps_only_the_latest_debounced_request() -> None:
    """Avoid running splat depth and DOF passes for every slider event."""
    application()
    renderer = SnapshotRenderer()
    preview = NativeLivePreview(background_factory=lambda: renderer)
    first = Viewpoint(field_of_view=35)
    latest = Viewpoint(field_of_view=55)

    preview.set_viewpoint(first)
    preview.set_viewpoint(latest)

    assert BACKGROUND_REFRESH_DEBOUNCE_MS == 150
    assert preview._viewpoint_timer.isActive()  # noqa: SLF001
    assert renderer.viewpoints == []
    preview._apply_pending_viewpoint()  # noqa: SLF001
    assert renderer.viewpoints == [latest]
    assert preview._background_timer.isActive()  # noqa: SLF001
    preview._background_timer.stop()  # noqa: SLF001
    preview._capture_background()  # noqa: SLF001
    assert preview._surface._background_revision == 1  # noqa: SLF001
    assert preview._surface._harmonization_revision == 0  # noqa: SLF001
    preview.close()


def test_live_snapshot_rejects_a_stale_revision() -> None:
    """Keep the previous complete layer pair when an older capture arrives."""
    application()
    renderer = SnapshotRenderer()
    preview = NativeLivePreview(background_factory=lambda: renderer)
    preview._scene_asset_id = "room-v1"  # noqa: SLF001

    renderer.snapshot_ready.emit("room-v1", 2, "background", renderer.payload)
    renderer.snapshot_ready.emit("room-v1", 1, "background", renderer.payload)

    assert preview._latest_snapshot_revision == 2  # noqa: SLF001
    preview.close()


def test_live_snapshot_delivers_framebuffer_pixels_without_grabbing_the_hidden_view() -> None:
    """Decode the completed WebGL framebuffer rather than QWidget presentation state."""
    application()
    renderer = SnapshotRenderer()
    preview = NativeLivePreview(background_factory=lambda: renderer)
    preview._scene_asset_id = "room-v1"  # noqa: SLF001

    renderer.snapshot_ready.emit("room-v1", 4, "background", renderer.payload)

    assert preview._latest_snapshot_revision == 4  # noqa: SLF001
    assert preview._surface._background_revision == 1  # noqa: SLF001
    preview.close()


def test_dof_snapshot_does_not_replace_the_sharp_harmonization_reference() -> None:
    """Keep subject appearance evidence stable across background-only DOF revisions."""
    application()
    renderer = SnapshotRenderer()
    preview = NativeLivePreview(background_factory=lambda: renderer)
    preview._scene_asset_id = "room-v1"  # noqa: SLF001

    renderer.snapshot_ready.emit("room-v1", 5, "harmonization", renderer.payload)
    reference_revision = preview._surface._harmonization_revision  # noqa: SLF001
    renderer.snapshot_ready.emit("room-v1", 5, "background", renderer.payload)

    assert preview._latest_harmonization_revision == 5  # noqa: SLF001
    assert preview._latest_snapshot_revision == 5  # noqa: SLF001
    assert preview._surface._harmonization_revision == reference_revision  # noqa: SLF001
    assert preview._surface._background_revision == 1  # noqa: SLF001
    preview.close()
