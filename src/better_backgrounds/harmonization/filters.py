"""Faithful native renderer for Harmonizer's learned global controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

GLOBAL_ARGUMENT_COUNT = 6
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3


@dataclass(frozen=True, slots=True)
class HarmonizerFilterRenderer:
    """Apply session-compiled equivalents of the six learned filters."""

    appearance_lookup: NDArray[np.float32]
    saturation_scale: float
    tone_lookup: NDArray[np.uint8]

    @classmethod
    def compile(
        cls,
        reference: NDArray[np.uint8],
        arguments: Sequence[float],
    ) -> HarmonizerFilterRenderer:
        """Compile fixed session controls using representative image statistics."""
        if len(arguments) != GLOBAL_ARGUMENT_COUNT:
            msg = "Harmonizer must predict six global arguments"
            raise ValueError(msg)
        values = np.clip(np.asarray(arguments, dtype=np.float32), -1.0, 1.0)
        temperature, brightness, contrast, saturation, highlight, shadow = values
        image = reference.astype(np.float32) / 255.0
        coefficients = _temperature_coefficients(image, float(temperature))
        brightness_scale = _brightness_scale(float(brightness))
        representative = image * coefficients
        np.clip(representative, 0.0, 1.0, out=representative)
        representative *= brightness_scale
        np.clip(representative, 0.0, 1.0, out=representative)
        threshold = sum(cv2.mean(representative)[:RGB_CHANNELS]) / RGB_CHANNELS
        contrast_scale, contrast_offset = _contrast_transform(float(contrast), threshold)

        channel_values = np.arange(256, dtype=np.float32)[:, None] / 255.0
        appearance = channel_values * coefficients.reshape(1, RGB_CHANNELS)
        np.clip(appearance, 0.0, 1.0, out=appearance)
        appearance *= brightness_scale
        np.clip(appearance, 0.0, 1.0, out=appearance)
        appearance *= contrast_scale
        appearance += contrast_offset
        appearance_lookup = np.clip(appearance, 0.0, 1.0).reshape(
            256,
            1,
            RGB_CHANNELS,
        )
        tone_lookup = _tone_lookup(float(highlight), float(shadow))
        saturation_scale = (
            1.0 / (1.0 - float(saturation) + 1e-6) if saturation >= 0 else 1.0 + float(saturation)
        )
        return cls(appearance_lookup, saturation_scale, tone_lookup)

    def render(
        self,
        composite: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        *,
        transition: float = 1.0,
    ) -> NDArray[np.uint8]:
        """Render one frame with a bounded automatic startup transition."""
        transformed = cv2.LUT(composite, self.appearance_lookup)
        hls = cast(
            "NDArray[np.float32]",
            cv2.cvtColor(transformed, cv2.COLOR_RGB2HLS),
        )
        hls[..., 2] *= self.saturation_scale
        np.clip(hls[..., 2], 0.0, 1.0, out=hls[..., 2])
        transformed = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        transformed = np.rint(np.clip(transformed, 0.0, 1.0) * 255.0).astype(np.uint8)
        transformed = cv2.LUT(transformed, self.tone_lookup)
        weight = alpha.astype(np.float32) * float(np.clip(transition, 0.0, 1.0)) / 255.0
        restored = cast(
            "NDArray[np.uint8]",
            cv2.blendLinear(transformed, composite, weight, 1.0 - weight),
        )
        cv2.copyTo(composite, np.equal(alpha, 0).astype(np.uint8), restored)
        return restored


def render_harmonizer_filters(
    composite: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    arguments: Sequence[float],
) -> NDArray[np.uint8]:
    """Execute the checkpoint's six filters in their trained order."""
    if len(arguments) != GLOBAL_ARGUMENT_COUNT:
        msg = "Harmonizer must predict six global arguments"
        raise ValueError(msg)
    if (
        composite.dtype != np.uint8
        or composite.ndim != RGB_DIMENSIONS
        or composite.shape[2] != RGB_CHANNELS
    ):
        msg = "composite must be uint8 RGB"
        raise ValueError(msg)
    if alpha.dtype != np.uint8 or alpha.shape != composite.shape[:2]:
        msg = "alpha must be uint8 and match composite"
        raise ValueError(msg)

    values = np.clip(np.asarray(arguments, dtype=np.float32), -1.0, 1.0)
    temperature, brightness, contrast, saturation, highlight, shadow = values
    image = _temperature(composite.astype(np.float32) / 255.0, float(temperature))
    image = _brightness(image, float(brightness))
    image = _contrast(image, float(contrast))
    image = _saturation(image, float(saturation))
    transformed = _highlight_and_shadow(image, float(highlight), float(shadow))

    matte = alpha.astype(np.float32) / 255.0
    restored = cast(
        "NDArray[np.uint8]",
        cv2.blendLinear(transformed, composite, matte, 1.0 - matte),
    )
    cv2.copyTo(composite, np.equal(alpha, 0).astype(np.uint8), restored)
    return restored


def _temperature(image: NDArray[np.float32], argument: float) -> NDArray[np.float32]:
    coefficients = _temperature_coefficients(image, argument)
    cv2.transform(image, np.diag(coefficients), dst=image)
    return np.clip(image, 0.0, 1.0, out=image)


def _temperature_coefficients(
    image: NDArray[np.float32],
    argument: float,
) -> NDArray[np.float32]:
    means = np.asarray(cv2.mean(image)[:RGB_CHANNELS], dtype=np.float32).reshape(1, 1, 3)
    gray = float(means.sum()) / RGB_CHANNELS
    bias = 1.0 - gray / (means + 1e-6)
    magnitude = argument * np.sign(argument)
    targets = means.copy()
    if argument < 0:
        targets[..., 0] += magnitude
    if argument != 0:
        targets[..., 1] += magnitude * 0.5
    if argument > 0:
        targets[..., 2] += magnitude
    target_gray = float(targets.sum()) / RGB_CHANNELS
    return cast(
        "NDArray[np.float32]",
        (target_gray / (targets + 1e-6) + bias).reshape(RGB_CHANNELS),
    )


def _brightness(image: NDArray[np.float32], argument: float) -> NDArray[np.float32]:
    image *= _brightness_scale(argument)
    return np.clip(image, 0.0, 1.0, out=image)


def _brightness_scale(argument: float) -> float:
    return 1.0 / (1.0 - argument + 1e-6) if argument >= 0 else argument + 1.0


def _contrast(image: NDArray[np.float32], argument: float) -> NDArray[np.float32]:
    threshold = sum(cv2.mean(image)[:RGB_CHANNELS]) / RGB_CHANNELS
    scale, offset = _contrast_transform(argument, threshold)
    matrix = np.concatenate(
        (
            np.eye(RGB_CHANNELS, dtype=np.float32) * scale,
            np.full((RGB_CHANNELS, 1), offset, dtype=np.float32),
        ),
        axis=1,
    )
    cv2.transform(image, matrix, dst=image)
    return np.clip(image, 0.0, 1.0, out=image)


def _contrast_transform(argument: float, threshold: float) -> tuple[float, float]:
    adjusted = 255.0 / (256.0 - np.floor(argument * 255.0)) - 1.0 if argument > 0 else argument
    return 1.0 + float(adjusted), -threshold * float(adjusted)


def _saturation(image: NDArray[np.float32], argument: float) -> NDArray[np.float32]:
    hls = cast("NDArray[np.float32]", cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
    channel = hls[..., 2]
    if argument >= 0:
        channel /= 1.0 - argument + 1e-6
    else:
        channel *= 1.0 + argument
    np.clip(channel, 0.0, 1.0, out=channel)
    return cast("NDArray[np.float32]", cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))


def _highlight_and_shadow(
    image: NDArray[np.float32],
    highlight: float,
    shadow: float,
) -> NDArray[np.uint8]:
    lookup = _tone_lookup(highlight, shadow)
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cast("NDArray[np.uint8]", cv2.LUT(pixels, lookup))


def _tone_lookup(highlight: float, shadow: float) -> NDArray[np.uint8]:
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    values = 1.0 - np.power(1.0 - values + 1e-9, highlight + 1.0)
    values = np.power(np.clip(values, 0.0, 1.0) + 1e-9, -shadow + 1.0)
    return cast(
        "NDArray[np.uint8]",
        np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8),
    )
