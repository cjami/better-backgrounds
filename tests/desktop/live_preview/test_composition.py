"""Focused tests for latest-frame live composition coordination."""

from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.live_preview.composition import CompositionCoordinator
from better_backgrounds.desktop.live_preview.surface import PreparedComposite

if TYPE_CHECKING:
    from better_backgrounds.desktop.live_preview.surface import NativeCompositeSurface
    from better_backgrounds.matting.compositor import LiveComposite
    from better_backgrounds.matting.engine import CompletedMatte


class RecordingSurface:
    """Record which queued matte reaches asynchronous preparation."""

    def __init__(self) -> None:
        """Create an empty call log."""
        self.calls: list[object] = []

    def prepare_matte(self, completed: CompletedMatte) -> PreparedComposite:
        """Return a minimal prepared frame for the supplied token."""
        self.calls.append(completed)
        pixels = np.zeros((1, 1, 3), dtype=np.uint8)
        alpha = np.zeros((1, 1), dtype=np.uint8)
        return PreparedComposite(
            completed,
            pixels,
            alpha,
            cast("LiveComposite", object()),
        )


def test_composition_queue_keeps_latest_pending_frame_and_reset_discards_it() -> None:
    """Replace stale pending work and invalidate all ownership on reset."""
    app = QApplication.instance() or QApplication([])
    surface = RecordingSurface()
    coordinator = CompositionCoordinator(cast("NativeCompositeSurface", surface))
    first = cast("CompletedMatte", object())
    latest = cast("CompletedMatte", object())
    coordinator._inflight = True  # noqa: SLF001

    coordinator.submit(first)
    coordinator.submit(latest)

    assert coordinator.presentation_drops == 1
    coordinator._inflight = False  # noqa: SLF001
    coordinator._accept(0, _prepared(first))  # noqa: SLF001
    ready = coordinator.take_ready()
    assert ready is not None
    assert ready.completed is latest
    for _attempt in range(20):
        app.processEvents()
        if surface.calls:
            break
    assert surface.calls == [latest]

    coordinator.reset()
    assert coordinator.presentation_drops == 0
    assert coordinator.take_ready() is None
    coordinator.close()


def test_composition_queue_replaces_an_unpainted_prepared_frame() -> None:
    """Expose only the newest completed preparation to the Qt presenter."""
    surface = RecordingSurface()
    coordinator = CompositionCoordinator(cast("NativeCompositeSurface", surface))
    first = cast("CompletedMatte", object())
    latest = cast("CompletedMatte", object())

    coordinator._accept(0, _prepared(first))  # noqa: SLF001
    coordinator._accept(0, _prepared(latest))  # noqa: SLF001

    prepared = coordinator.take_ready()
    assert prepared is not None
    assert prepared.completed is latest
    assert coordinator.presentation_drops == 1
    coordinator.close()


def _prepared(completed: CompletedMatte) -> PreparedComposite:
    pixels = np.zeros((1, 1, 3), dtype=np.uint8)
    alpha = np.zeros((1, 1), dtype=np.uint8)
    return PreparedComposite(
        completed,
        pixels,
        alpha,
        cast("LiveComposite", object()),
    )
