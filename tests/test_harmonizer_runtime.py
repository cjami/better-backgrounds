"""Verify the external Harmonizer inference, smoothing, and fallback boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from better_backgrounds._vendor.harmonizer import HarmonizerInferenceModel
from better_backgrounds.harmonization import HarmonizationSettings
from better_backgrounds.harmonizer_runtime import (
    HARMONIZER_CHECKPOINT_ENV,
    HarmonizerAppearanceHarmonizer,
    checkpoint_from_environment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


class StubHarmonizerModel(HarmonizerInferenceModel):
    """Expose changing global arguments without constructing the real network."""

    def __init__(self) -> None:
        """Track per-frame predictions through the model boundary."""
        nn.Module.__init__(self)
        self.marker = nn.Parameter(torch.zeros(1))
        self.predictions = 0

    def predict_arguments(
        self,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Jump from a dark argument to a bright argument after the first frame."""
        assert mask.ndim == 4
        self.predictions += 1
        value = 0.2 if self.predictions == 1 else 0.8
        return [composite.new_full((1, 1), value)]

    def restore_image(
        self,
        composite: torch.Tensor,
        mask: torch.Tensor,
        arguments: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Render the smoothed argument while retaining the exact background."""
        foreground = torch.ones_like(composite) * arguments[0].flatten()[0]
        return foreground * mask + composite * (1.0 - mask)


def test_external_checkpoint_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require an explicit external path instead of bundling non-commercial weights."""
    monkeypatch.delenv(HARMONIZER_CHECKPOINT_ENV, raising=False)
    assert checkpoint_from_environment() is None

    monkeypatch.setenv(HARMONIZER_CHECKPOINT_ENV, "models/harmonizer.pth")

    assert checkpoint_from_environment() == Path("models/harmonizer.pth").resolve()


def test_missing_checkpoint_falls_back_to_standard_composite(tmp_path: Path) -> None:
    """Keep the live preview usable when the external checkpoint is absent."""
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    harmonizer = HarmonizerAppearanceHarmonizer(
        tmp_path / "missing.pth",
        preferred_device="cpu",
    )
    harmonizer.set_room(image, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))

    result = harmonizer.apply(image, alpha, image, captured_at=0.0)

    assert not result.applied
    assert result.image is None
    assert result.degraded_components == ("harmonizer",)
    assert harmonizer.error is not None
    assert "not found" in harmonizer.error


def test_global_arguments_are_predicted_each_frame_and_smoothed(tmp_path: Path) -> None:
    """Prevent a new global prediction from producing an abrupt colour jump."""
    checkpoint = tmp_path / "harmonizer.pth"
    model = StubHarmonizerModel()
    torch.save(model.state_dict(), checkpoint)
    harmonizer = HarmonizerAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    background = np.full((8, 8, 3), 40, dtype=np.uint8)
    source = np.full((8, 8, 3), 200, dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.uint8)
    alpha[:, 2:6] = 255
    harmonizer.set_room(background, revision=2)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))

    first = harmonizer.apply(source, alpha, background, captured_at=0.0)
    second = harmonizer.apply(source, alpha, background, captured_at=1_000.0 / 30.0)
    third = harmonizer.apply(source, alpha, background, captured_at=2_000.0 / 30.0)

    assert first.applied
    assert second.applied
    assert third.applied
    assert first.statistics_updated
    assert second.statistics_updated
    assert third.statistics_updated
    assert model.predictions == 3
    assert first.image is not None
    assert second.image is not None
    assert third.image is not None
    assert np.array_equal(first.image[:, :2], background[:, :2])
    assert np.all(first.image[:, 2:6] == 51)
    assert np.all(second.image[:, 2:6] == 56)
    assert np.all(third.image[:, 2:6] == 61)
