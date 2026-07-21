"""Feature-first: Tests for low-friction person acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from better_backgrounds.desktop.live_preview.preview import NativeLivePreview
from better_backgrounds.desktop.live_preview.surface import NativeCompositeSurface
from better_backgrounds.matting.seed import PersonCandidate

if TYPE_CHECKING:
    from better_backgrounds.matting.contracts import MattingConfig
    from better_backgrounds.matting.engine import ProcessMattingEngine


class RecordingEngine:
    """Record lifecycle commands without starting native model dependencies."""

    def __init__(self) -> None:
        """Create empty lifecycle recordings."""
        self.preparations: list[MattingConfig] = []
        self.initializations: list[tuple[np.ndarray, np.ndarray]] = []
        self.initialization_configs: list[MattingConfig] = []
        self.resets = 0
        self.closes = 0
        self.dropped_frames = 0

    def prepare(self, config: MattingConfig) -> None:
        """Record model preparation."""
        self.preparations.append(config)

    def initialize(self, frame, mask, config, _pipeline) -> None:  # noqa: ANN001
        """Record one selected frame and mask."""
        self.initializations.append((frame, mask))
        self.initialization_configs.append(config)

    def reset(self) -> None:
        """Record tracking reset without discarding the fake model."""
        self.resets += 1

    def close(self) -> None:
        """Accept final shutdown."""
        self.closes += 1

    def set_live_background(self, *_args: object, **_kwargs: object) -> None:
        """Accept room synchronization."""

    def configure_harmonization(self, _settings) -> None:  # noqa: ANN001
        """Accept appearance synchronization."""


def application() -> QApplication:
    """Return the shared Qt application used by native widget tests."""
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def candidate(candidate_id: int, left: int) -> PersonCandidate:
    """Create one plausible disconnected person region."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, left : left + 5] = 255
    return PersonCandidate(candidate_id, mask, (left, 3, 5, 14), 0.175)


def test_live_preview_forces_highest_matting_quality() -> None:
    """Never trade MatAnyone inference resolution for startup calibration speed."""
    application()
    engine = RecordingEngine()
    preview = NativeLivePreview(
        background_factory=QWidget,
        engine_factory=lambda: cast("ProcessMattingEngine", engine),
    )

    preview.prepare_matting()

    assert len(engine.preparations) == 1
    assert engine.preparations[0].internal_size == 540
    assert not engine.preparations[0].calibrate
    preview.close()


def test_single_person_initializes_without_confirmation() -> None:
    """Make the normal one-person webcam path zero-click."""
    application()
    engine = RecordingEngine()
    preview = NativeLivePreview(
        background_factory=QWidget,
        engine_factory=lambda: cast("ProcessMattingEngine", engine),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    preview._set_state("seeding", "Finding you")  # noqa: SLF001

    preview._accept_seed(frame, (candidate(1, 4),))  # noqa: SLF001

    assert len(engine.initializations) == 1
    assert engine.initialization_configs[0].internal_size == 540
    assert not engine.initialization_configs[0].calibrate
    assert preview._state == "initializing"  # noqa: SLF001
    preview.close()


def test_multiple_people_wait_for_an_explicit_candidate() -> None:
    """Avoid silently choosing the largest person in a shared camera frame."""
    application()
    engine = RecordingEngine()
    preview = NativeLivePreview(
        background_factory=QWidget,
        engine_factory=lambda: cast("ProcessMattingEngine", engine),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    preview._set_state("seeding", "Finding you")  # noqa: SLF001

    preview._accept_seed(  # noqa: SLF001
        frame,
        (candidate(1, 2), candidate(2, 13)),
    )
    assert preview._state == "choose-person"  # noqa: SLF001
    assert not engine.initializations

    preview.select_person_candidate(2)

    assert len(engine.initializations) == 1
    assert np.array_equal(engine.initializations[0][1], candidate(2, 13).mask)
    preview.close()


def test_reselection_resets_tracking_without_closing_the_model() -> None:
    """Keep retained network weights alive while the user chooses themselves again."""
    application()
    engine = RecordingEngine()
    preview = NativeLivePreview(
        background_factory=QWidget,
        engine_factory=lambda: cast("ProcessMattingEngine", engine),
    )
    preview.prepare_matting()

    preview.reselect_person()

    assert engine.resets == 1
    assert engine.closes == 0
    preview.close()


def test_candidate_hit_testing_accounts_for_mirroring_and_cover_crop() -> None:
    """Map preview clicks to the same mirrored person outline users can see."""
    application()
    surface = NativeCompositeSurface()
    surface.resize(800, 400)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 5:20] = 255
    person = PersonCandidate(1, mask, (5, 40, 15, 20), 0.03)

    surface.set_seed_candidates(frame, (person,))

    assert surface._candidate_at(720, 200) == 1  # noqa: SLF001
    assert surface._candidate_at(80, 200) is None  # noqa: SLF001
    surface.close()
