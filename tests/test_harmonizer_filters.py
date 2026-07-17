"""Verify the native Harmonizer filters against the learned filter contract."""

from __future__ import annotations

import numpy as np
import torch

from better_backgrounds.harmonizer_filters import (
    HarmonizerFilterRenderer,
    render_harmonizer_filters,
)


def _reference_filters(
    composite: np.ndarray,
    alpha: np.ndarray,
    arguments: tuple[float, ...],
) -> np.ndarray:
    """Reproduce the checkpoint's original sequential Torch filter equations."""
    image = torch.from_numpy(composite).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
    matte = torch.from_numpy(alpha).to(torch.float32).unsqueeze(0).unsqueeze(0) / 255
    temperature, brightness, contrast, saturation_argument, highlight, shadow = (
        torch.tensor(value).clamp(-1, 1).view(1, 1, 1, 1) for value in arguments
    )

    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    means = [channel.mean(dim=(2, 3), keepdim=True) for channel in (red, green, blue)]
    gray = sum(means) / 3
    biases = [1 - gray / (mean + 1e-6) for mean in means]
    positive = (temperature > 0).float()
    negative = (temperature < 0).float()
    zero = (temperature == 0).float()
    targets = (
        means[0] + temperature * torch.sign(temperature) * negative,
        means[1] + temperature * torch.sign(temperature) * 0.5 * (1 - zero),
        means[2] + temperature * torch.sign(temperature) * positive,
    )
    target_gray = sum(targets) / 3
    image = torch.cat(
        tuple(
            (target_gray / (target + 1e-6) + bias) * channel
            for target, bias, channel in zip(
                targets,
                biases,
                (red, green, blue),
                strict=True,
            )
        ),
        dim=1,
    ).clamp(0, 1)

    positive = (brightness >= 0).float()
    brightness_scale = (1 / (1 - brightness + 1e-6)) * positive
    brightness_scale += (brightness + 1) * (1 - positive)
    image = (image * brightness_scale).clamp(0, 1)

    threshold = image.mean(dim=(1, 2, 3), keepdim=True)
    positive = (contrast > 0).float()
    adjusted = 255 / (256 - torch.floor(contrast * 255)) - 1
    adjusted = contrast * (1 - positive) + adjusted * positive
    image = (image + (image - threshold) * adjusted).clamp(0, 1)

    minimum = image.min(dim=1, keepdim=True)[0]
    maximum = image.max(dim=1, keepdim=True)[0]
    spread = maximum - minimum
    total = maximum + minimum
    mean = total / 2
    positive = (saturation_argument >= 0).float()
    lower_half = (mean < 0.5).float()
    saturation = spread / (total + 1e-6) * lower_half
    saturation += spread / (2 - total + 1e-6) * (1 - lower_half)
    clipped = ((saturation_argument + saturation) > 1).float()
    positive_alpha = saturation * clipped + (1 - saturation_argument) * (1 - clipped)
    positive_alpha = 1 / (positive_alpha + 1e-6) - 1
    saturation_alpha = positive_alpha * positive + (1 + saturation_argument) * (1 - positive)
    image = (image * positive + mean * (1 - positive) + (image - mean) * saturation_alpha).clamp(
        0, 1
    )

    image = (1 - torch.pow(1 - image + 1e-9, highlight + 1)).clamp(0, 1)
    image = torch.pow(image + 1e-9, -shadow + 1).clamp(0, 1)
    original = torch.from_numpy(composite).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
    image = image * matte + original * (1 - matte)
    return image.squeeze(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()


def test_native_filters_match_the_checkpoint_filter_equations() -> None:
    """Keep optimization from changing the meaning of predicted arguments."""
    composite = np.random.default_rng(4).integers(16, 240, (9, 11, 3), dtype=np.uint8)
    alpha = np.tile(np.linspace(0, 255, 11, dtype=np.uint8), (9, 1))
    argument_sets = (
        (0.08, 0.12, 0.16, -0.14, 0.09, -0.07),
        (-0.06, -0.1, -0.12, 0.18, -0.08, 0.11),
    )

    for arguments in argument_sets:
        actual = render_harmonizer_filters(composite, alpha, arguments)
        expected = _reference_filters(composite, alpha, arguments)

        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        assert int(difference.max()) <= 1
        assert np.array_equal(actual[:, 0], composite[:, 0])


def test_compiled_renderer_stays_close_to_the_reference_equations() -> None:
    """Keep session LUT compilation visually equivalent to the native reference."""
    composite = np.random.default_rng(8).integers(16, 240, (90, 110, 3), dtype=np.uint8)
    alpha = np.tile(np.linspace(0, 255, 110, dtype=np.uint8), (90, 1))
    arguments = (0.08, 0.12, 0.16, 0.14, 0.09, -0.07)
    renderer = HarmonizerFilterRenderer.compile(composite, arguments)

    actual = renderer.render(composite, alpha)
    expected = _reference_filters(composite, alpha, arguments)

    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(difference.max()) <= 3
    assert float(difference.mean()) < 0.6
    assert np.array_equal(actual[:, 0], composite[:, 0])
