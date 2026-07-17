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
        zero = composite.new_zeros((1, 1))
        return [zero, composite.new_full((1, 1), value), zero, zero, zero, zero]


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


def test_global_arguments_are_predicted_at_ten_hz_and_smoothed_each_frame(
    tmp_path: Path,
) -> None:
    """Reduce model work without introducing abrupt global colour jumps."""
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
    harmonizer.apply(source, alpha, background, captured_at=5_000.0 - 1_000.0 / 30.0)
    fourth = harmonizer.apply(source, alpha, background, captured_at=5_000.0)
    fifth = harmonizer.apply(source, alpha, background, captured_at=5_000.0 + 1_000.0 / 30.0)

    assert first.applied
    assert second.applied
    assert third.applied
    assert fourth.applied
    assert fifth.applied
    assert model.predictions == 2
    assert first.image is not None
    assert second.image is not None
    assert third.image is not None
    assert fourth.image is not None
    assert fifth.image is not None
    assert np.array_equal(first.image[:, :2], background[:, :2])
    first_value = int(first.image[0, 2, 0])
    assert np.all(second.image[:, 2:6] == first.image[:, 2:6])
    assert np.all(third.image[:, 2:6] == first.image[:, 2:6])
    assert np.all(fourth.image[:, 2:6] > first_value)
    assert np.all(fifth.image[:, 2:6] >= fourth.image[:, 2:6])
