"""Feature-first: Verify opt-in PIH inference and exact-frame fallback behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional

import better_backgrounds.harmonization.pih as pih_runtime
from better_backgrounds._vendor.pih import apply_rgb_curves
from better_backgrounds.harmonization import HarmonizationSettings
from better_backgrounds.harmonization.pih import (
    CURVE_SAMPLE_COUNT,
    DEFAULT_CURVE_STRENGTH,
    GAIN_NEW_FRAME_WEIGHT,
    HARMONIZATION_BACKEND_ENV,
    PIH_CHECKPOINT_ENV,
    PIH_CURVE_STRENGTH_ENV,
    PihAppearanceHarmonizer,
    create_appearance_harmonizer,
    pih_checkpoint_from_environment,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class StubPihModel(nn.Module):
    """Return identity color curves and a fixed local darkening map."""

    def __init__(self) -> None:
        """Expose one checkpoint-compatible marker."""
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.input_shapes: tuple[tuple[int, ...], ...] | None = None
        self.curve_calls = 0
        self.gain_calls = 0
        self.gain_values: list[float] = []

    def predict_curves(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Record inputs and return identity-like global curves."""
        self.input_shapes = (tuple(background.shape), tuple(composite.shape), tuple(mask.shape))
        self.curve_calls += 1
        controls = torch.linspace(0.0, 1.0, 32, device=composite.device, dtype=composite.dtype)
        return controls.expand(composite.shape[0], 3, 32)

    def predict_gain(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
        curves: torch.Tensor,
    ) -> torch.Tensor:
        """Return configured frame-local darkening."""
        del background, mask, curves
        index = min(self.gain_calls, len(self.gain_values) - 1)
        value = self.gain_values[index] if self.gain_values else 0.75
        self.gain_calls += 1
        return composite.new_full((composite.shape[0], 1, 512, 512), value)

    def predict_parameters(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record low-resolution inputs and darken the foreground locally."""
        curves = self.predict_curves(background, composite, mask)
        return curves, self.predict_gain(background, composite, mask, curves)


def test_compact_curve_renderer_matches_the_official_expanded_lut() -> None:
    """Preserve Adobe's align-corners-false curve interpolation without a 3D table."""
    generator = torch.Generator().manual_seed(7)
    image = torch.rand((1, 3, 8, 12), generator=generator)
    curves = torch.rand((1, 3, 32), generator=generator)
    table = torch.cat(
        (
            curves[:, 0, None, None, :][:, None].expand(1, 1, 32, 32, 32),
            curves[:, 1, None, :, None][:, None].expand(1, 1, 32, 32, 32),
            curves[:, 2, :, None, None][:, None].expand(1, 1, 32, 32, 32),
        ),
        dim=1,
    )
    grid = (image * 2.0 - 1.0).permute(0, 2, 3, 1).contiguous().unsqueeze(1)
    expected = functional.grid_sample(
        table,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(2)

    actual = apply_rgb_curves(image, curves)

    assert torch.allclose(actual, expected, atol=2e-7, rtol=0.0)


def test_pih_checkpoint_is_external_and_selects_the_available_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a prepared development checkpoint without transient shell state."""
    development_checkpoint = tmp_path / "development" / "ckpt_g39.pth"
    monkeypatch.setattr(
        pih_runtime,
        "DEVELOPMENT_PIH_CHECKPOINT",
        development_checkpoint,
    )
    monkeypatch.delenv(PIH_CHECKPOINT_ENV, raising=False)
    monkeypatch.delenv(HARMONIZATION_BACKEND_ENV, raising=False)
    monkeypatch.delenv(PIH_CURVE_STRENGTH_ENV, raising=False)

    assert pih_checkpoint_from_environment() is None
    assert create_appearance_harmonizer().__class__.__name__ == "HarmonizerAppearanceHarmonizer"

    development_checkpoint.parent.mkdir()
    development_checkpoint.touch()
    monkeypatch.setenv(PIH_CURVE_STRENGTH_ENV, "0.5")

    harmonizer = create_appearance_harmonizer()
    assert isinstance(harmonizer, PihAppearanceHarmonizer)
    assert harmonizer.checkpoint == development_checkpoint
    assert harmonizer.curve_strength == 0.5
    assert PihAppearanceHarmonizer().curve_strength == DEFAULT_CURVE_STRENGTH

    explicit_checkpoint = tmp_path / "explicit.pth"
    monkeypatch.setenv(PIH_CHECKPOINT_ENV, str(explicit_checkpoint))
    monkeypatch.setenv(HARMONIZATION_BACKEND_ENV, "pih")

    assert pih_checkpoint_from_environment() == explicit_checkpoint.resolve()
    assert PihAppearanceHarmonizer().checkpoint == explicit_checkpoint.resolve()


def test_missing_pih_checkpoint_falls_back_to_standard_composite(tmp_path: Path) -> None:
    """Keep live composition usable when PIH cannot be prepared."""
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    harmonizer = PihAppearanceHarmonizer(
        tmp_path / "missing.pth",
        preferred_device="cpu",
    )
    harmonizer.set_room(image, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))

    result = harmonizer.apply(image, alpha, image, captured_at=0.0)

    assert not result.applied
    assert result.image is None
    assert result.degraded_components == ("pih",)
    assert harmonizer.error is not None
    assert "not found" in harmonizer.error


def test_pih_predicts_current_frame_parameters_and_preserves_background(tmp_path: Path) -> None:
    """Apply current-pose shading without changing pixels outside the matte."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
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

    first = harmonizer.apply(source, alpha, background, captured_at=1_000.0)
    result = harmonizer.apply(source, alpha, background, captured_at=2_000.0)

    assert first.applied
    assert result.applied
    assert result.image is not None
    assert result.image.flags.c_contiguous
    assert model.input_shapes == ((1, 3, 512, 512), (1, 3, 512, 512), (1, 1, 512, 512))
    assert np.array_equal(result.image[:, :2], background[:, :2])
    assert np.all(result.image[:, 2:6] < source[:, 2:6])
    assert result.degraded_components == ()
    assert harmonizer.backend_name == "PIH/CPU"
    assert len(harmonizer._background_tensors) == 1  # noqa: SLF001


def test_pih_resets_temporal_curves_for_a_changed_room(tmp_path: Path) -> None:
    """Never carry smoothed appearance parameters across room identities."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    background = np.full((8, 8, 3), 40, dtype=np.uint8)
    source = np.full((8, 8, 3), 200, dtype=np.uint8)
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    harmonizer.set_room(background, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))
    assert harmonizer.apply(source, alpha, background, captured_at=0.0).applied

    harmonizer.set_room(background, revision=2)

    assert harmonizer.previous_curves is None


def test_pih_locks_global_curves_after_startup_samples(tmp_path: Path) -> None:
    """Avoid both global pumping and repeated ResNet inference in steady state."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    alpha = np.full((2, 2), 255, dtype=np.uint8)
    harmonizer.set_room(image, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))

    for index in range(CURVE_SAMPLE_COUNT + 2):
        assert harmonizer.apply(image, alpha, image, captured_at=index * 100.0).applied

    assert model.curve_calls == CURVE_SAMPLE_COUNT
    assert model.gain_calls == CURVE_SAMPLE_COUNT + 2


def test_pih_recalibrates_when_returning_to_a_room(tmp_path: Path) -> None:
    """Never preserve a potentially poor calibration across room changes."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    first_room = np.full((2, 2, 3), 40, dtype=np.uint8)
    second_room = np.full((2, 2, 3), 80, dtype=np.uint8)
    alpha = np.full((2, 2), 255, dtype=np.uint8)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))
    harmonizer.set_room(first_room, revision=1)
    for index in range(CURVE_SAMPLE_COUNT):
        harmonizer.apply(first_room, alpha, first_room, captured_at=index * 100.0)
    initial_curve_calls = model.curve_calls

    harmonizer.set_room(second_room, revision=2)
    harmonizer.apply(second_room, alpha, second_room, captured_at=1_000.0)
    returned_room = np.full_like(first_room, 41)
    harmonizer.set_room(returned_room, revision=3)
    result = harmonizer.apply(returned_room, alpha, returned_room, captured_at=2_000.0)

    assert result.applied
    assert model.curve_calls == initial_curve_calls + 2


def test_pih_stabilizes_gain_energy_without_reusing_a_spatial_map(tmp_path: Path) -> None:
    """Suppress brightness pumping while every pixel uses current-frame shading."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    model.gain_values = [0.6, 1.0]
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    alpha = np.full((2, 2), 255, dtype=np.uint8)
    harmonizer.set_room(image, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))
    harmonizer.apply(image, alpha, image, captured_at=0.0)

    assert harmonizer.apply(image, alpha, image, captured_at=1_000.0).applied

    expected = 0.6 + (1.0 - 0.6) * GAIN_NEW_FRAME_WEIGHT
    previous_gain_mean = harmonizer._previous_gain_mean  # noqa: SLF001
    assert previous_gain_mean is not None
    assert torch.allclose(previous_gain_mean, torch.tensor(expected))


def test_pih_applies_a_soft_matte_once(tmp_path: Path) -> None:
    """Do not attenuate Adobe's already-composited soft boundary a second time."""
    checkpoint = tmp_path / "pih.pth"
    model = StubPihModel()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    harmonizer = PihAppearanceHarmonizer(
        checkpoint,
        preferred_device="cpu",
        model_factory=lambda: model,
    )
    background = np.full((2, 2, 3), 40, dtype=np.uint8)
    source = np.full((2, 2, 3), 200, dtype=np.uint8)
    alpha = np.full((2, 2), 128, dtype=np.uint8)
    harmonizer.set_room(background, revision=1)
    harmonizer.configure(HarmonizationSettings(global_harmonization=True))
    harmonizer.apply(source, alpha, background, captured_at=0.0)

    result = harmonizer.apply(source, alpha, background, captured_at=1_000.0)

    assert result.image is not None
    composite = harmonizer._prepare_composite(source, alpha, background)  # noqa: SLF001
    composite_tensor = harmonizer._image_tensor(composite, torch.device("cpu"))  # noqa: SLF001
    mask_tensor = harmonizer._mask_tensor(alpha, torch.device("cpu"))  # noqa: SLF001
    curves, gain = model.predict_parameters(
        harmonizer._image_tensor(background, torch.device("cpu")),  # noqa: SLF001
        functional.interpolate(composite_tensor, (512, 512)),
        functional.interpolate(mask_tensor, (512, 512)),
    )
    corrected = apply_rgb_curves(composite_tensor, curves)
    corrected = torch.lerp(composite_tensor, corrected, harmonizer.curve_strength)
    expected = corrected * functional.interpolate(gain, (2, 2)) * mask_tensor
    expected += (1.0 - mask_tensor) * harmonizer._image_tensor(  # noqa: SLF001
        background,
        torch.device("cpu"),
    )
    expected_image = (
        expected[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    )
    assert np.array_equal(result.image, expected_image)


def test_pih_binds_the_resolved_current_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept fused tensors on cuda:0 instead of treating cuda as a different device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    harmonizer = PihAppearanceHarmonizer(preferred_device="cuda")

    assert harmonizer._select_device() == torch.device("cuda:0")  # noqa: SLF001
