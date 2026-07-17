"""Native exact-frame alpha composition for the live preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from better_backgrounds.harmonization import AppearanceHarmonizer
    from better_backgrounds.live_matting import FramePacket, MatteResult

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
MINIMUM_BACKGROUND_BRIGHTNESS_RANGE = 12


@dataclass(frozen=True, slots=True)
class LiveComposite:
    """Retain the evidence used to produce one displayed composite."""

    frame_id: int
    source: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    image: NDArray[np.uint8]
    standard_image: NDArray[np.uint8]
    background_revision: int
    harmonized: bool
    harmonization_ms: float = 0.0
    harmonization_degraded: tuple[str, ...] = ()


def background_has_content(background: NDArray[np.uint8]) -> bool:
    """Reject uniform renderer clear frames without rejecting dark room detail."""
    if (
        background.dtype != np.uint8
        or background.ndim != RGB_DIMENSIONS
        or background.shape[2] != RGB_CHANNELS
        or background.size == 0
    ):
        return False
    brightness = background.sum(axis=2, dtype=np.uint16)
    return int(brightness.max()) - int(brightness.min()) > MINIMUM_BACKGROUND_BRIGHTNESS_RANGE


def compose_live_frame(
    packet: FramePacket,
    matte: MatteResult,
    source: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    background: NDArray[np.uint8],
    *,
    revision: int,
    harmonizer: AppearanceHarmonizer | None = None,
) -> LiveComposite:
    """Blend one matching source/matte pair against an immutable background."""
    if (
        packet.frame_id != matte.frame_id
        or packet.captured_at != matte.captured_at
        or packet.shared_slot != matte.alpha_slot
    ):
        msg = "source and matte must describe the same source frame"
        raise ValueError(msg)
    expected_source = (packet.height, packet.width, RGB_CHANNELS)
    expected_alpha = (packet.height, packet.width)
    if source.dtype != np.uint8 or source.shape != expected_source:
        msg = f"source must be {expected_source} uint8 RGB"
        raise ValueError(msg)
    if alpha.dtype != np.uint8 or alpha.shape != expected_alpha:
        msg = f"alpha must be {expected_alpha} uint8"
        raise ValueError(msg)
    if (
        background.dtype != np.uint8
        or background.ndim != RGB_DIMENSIONS
        or background.shape[2] != RGB_CHANNELS
    ):
        msg = "background must be uint8 RGB"
        raise ValueError(msg)
    if background.shape != expected_source:
        background = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                background,
                (packet.width, packet.height),
                interpolation=cv2.INTER_LINEAR,
            ),
        )
    weight = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    weighted_source = cv2.multiply(source, weight, dtype=cv2.CV_16U)
    weighted_background = cv2.multiply(
        background,
        cv2.bitwise_not(weight),
        dtype=cv2.CV_16U,
    )
    standard_image = cast(
        "NDArray[np.uint8]",
        cv2.convertScaleAbs(
            cv2.add(weighted_source, weighted_background),
            alpha=1 / 255,
        ),
    )
    image = standard_image
    harmonization_ms = 0.0
    harmonization_degraded: tuple[str, ...] = ()
    harmonized = harmonizer is not None and harmonizer.active
    if harmonized and harmonizer is not None:
        result = harmonizer.apply(
            source,
            alpha,
            background,
            captured_at=packet.captured_at,
        )
        if result.image is not None:
            image = result.image
        harmonization_ms = result.processing_ms
        harmonization_degraded = result.degraded_components
        harmonized = result.applied
    return LiveComposite(
        frame_id=packet.frame_id,
        source=source,
        alpha=alpha,
        image=image,
        standard_image=standard_image,
        background_revision=revision,
        harmonized=harmonized,
        harmonization_ms=harmonization_ms,
        harmonization_degraded=harmonization_degraded,
    )
