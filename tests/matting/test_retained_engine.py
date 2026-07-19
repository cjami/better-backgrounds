"""Feature-first: Tests for retained MatAnyone generation ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from better_backgrounds.matting.engine import ProcessMattingEngine

if TYPE_CHECKING:
    from pathlib import Path

    from better_backgrounds.matting.ring import SharedFrameRing


class RecordingRing:
    """Record shared-memory unlink requests without allocating native blocks."""

    def __init__(self) -> None:
        """Create one open fake ring."""
        self.closed = False

    def close(self, *, unlink: bool = False) -> None:
        """Record final owner cleanup."""
        self.closed = unlink


def test_retired_rings_wait_for_their_detaching_generation(tmp_path: Path) -> None:
    """Keep newer queued seed memory alive when an older reset is acknowledged."""
    engine = ProcessMattingEngine(tmp_path / "checkpoint.pth")
    old = RecordingRing()
    newer = RecordingRing()
    engine._retired_rings = [  # noqa: SLF001
        (2, cast("SharedFrameRing", old)),
        (4, cast("SharedFrameRing", newer)),
    ]

    engine._close_retired_rings(through_generation=2)  # noqa: SLF001

    assert old.closed
    assert not newer.closed
    assert engine._retired_rings == [(4, newer)]  # noqa: SLF001
