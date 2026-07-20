"""Native exact-frame alpha composition for the live preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import cv2
import numpy as np

from better_backgrounds.matting.refinement import add_light_wrap, compose_linear_light

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from better_backgrounds.harmonization import HarmonizationResult
    from better_backgrounds.matting.contracts import FramePacket, MatteResult

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
MINIMUM_BACKGROUND_BRIGHTNESS_RANGE = 12


class LiveHarmonizer(Protocol):
    """Describe the exact-frame appearance boundary used by the compositor."""

    @property
    def active(self) -> bool:
        """Return whether this stage was requested."""
        ...

    def apply(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        *,
        captured_at: float,
        reference_background: NDArray[np.uint8] | None = None,
    ) -> HarmonizationResult:
        """Process one exact frame and matte pair."""
        ...


@dataclass(frozen=True, slots=True)
class LiveComposite:
    """Retain the evidence used to produce one displayed composite."""

    frame_id: int
    source: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    image: NDArray[np.uint8]
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
    foreground: NDArray[np.uint8] | None = None,
    harmonizer: LiveHarmonizer | None = None,
    harmonization_background: NDArray[np.uint8] | None = None,
) -> LiveComposite:
    """Blend one matching source/matte pair against an immutable background."""
    _validate_frame_identity(packet, matte)
    expected_source = (packet.height, packet.width, RGB_CHANNELS)
    expected_alpha = (packet.height, packet.width)
    _validate_frame_buffers(source, alpha, foreground, expected_source, expected_alpha)
    _validate_background(background)
    if harmonization_background is not None:
        _validate_background(harmonization_background)
    if background.shape != expected_source:
        background = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                background,
                (packet.width, packet.height),
                interpolation=cv2.INTER_LINEAR,
            ),
        )
    if harmonization_background is not None and harmonization_background.shape != expected_source:
        harmonization_background = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                harmonization_background,
                (packet.width, packet.height),
                interpolation=cv2.INTER_LINEAR,
            ),
        )
    image: NDArray[np.uint8] | None = None
    harmonization_ms = 0.0
    harmonization_degraded: tuple[str, ...] = ()
    harmonized = harmonizer is not None and harmonizer.active
    render_source = source if foreground is None else foreground
    if harmonized and harmonizer is not None:
        result = harmonizer.apply(
            render_source,
            alpha,
            background,
            captured_at=packet.captured_at,
            reference_background=harmonization_background,
        )
        if result.image is not None:
            image = add_light_wrap(result.image, alpha, background)
        harmonization_ms = result.processing_ms
        harmonization_degraded = result.degraded_components
        harmonized = result.applied
    if image is None:
        image = _standard_composite(render_source, alpha, background)
    return LiveComposite(
        frame_id=packet.frame_id,
        source=source,
        alpha=alpha,
        image=image,
        background_revision=revision,
        harmonized=harmonized,
        harmonization_ms=harmonization_ms,
        harmonization_degraded=harmonization_degraded,
    )


def _validate_frame_identity(packet: FramePacket, matte: MatteResult) -> None:
    if (
        packet.frame_id != matte.frame_id
        or packet.captured_at != matte.captured_at
        or packet.shared_slot != matte.alpha_slot
    ):
        msg = "source and matte must describe the same source frame"
        raise ValueError(msg)


def _validate_frame_buffers(
    source: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    foreground: NDArray[np.uint8] | None,
    expected_source: tuple[int, int, int],
    expected_alpha: tuple[int, int],
) -> None:
    if source.dtype != np.uint8 or source.shape != expected_source:
        msg = f"source must be {expected_source} uint8 RGB"
        raise ValueError(msg)
    if alpha.dtype != np.uint8 or alpha.shape != expected_alpha:
        msg = f"alpha must be {expected_alpha} uint8"
        raise ValueError(msg)
    if foreground is not None and (
        foreground.dtype != np.uint8 or foreground.shape != expected_source
    ):
        msg = f"foreground must be {expected_source} uint8 RGB"
        raise ValueError(msg)


def _validate_background(background: NDArray[np.uint8]) -> None:
    if (
        background.dtype != np.uint8
        or background.ndim != RGB_DIMENSIONS
        or background.shape[2] != RGB_CHANNELS
    ):
        msg = "background must be uint8 RGB"
        raise ValueError(msg)


def _standard_composite(
    source: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Blend the plain composite used as output when harmonization is unavailable."""
    return compose_linear_light(source, alpha, background)
