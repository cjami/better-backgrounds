"""Persistent CUDA refinement, composition, and PIH rendering."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch.nn import functional

from better_backgrounds.harmonization.pih import PihAppearanceHarmonizer
from better_backgrounds.matting.refinement import (
    BACKGROUND_ESTIMATE_KERNEL,
    BACKGROUND_ESTIMATE_SCALE,
    BOUNDARY_ALPHA_MAXIMUM,
    BOUNDARY_ALPHA_MINIMUM,
    DECONTAMINATION_STRENGTH,
    LIGHT_WRAP_STRENGTH,
    MAXIMUM_STABILIZATION_GAP_MS,
    MAXIMUM_STABILIZED_DELTA,
    MINIMUM_RECOVERY_ALPHA,
    SRGB_DECODE_DIVISOR,
    SRGB_DECODE_THRESHOLD,
    SRGB_ENCODE_MULTIPLIER,
    SRGB_ENCODE_THRESHOLD,
    SRGB_GAMMA,
    SRGB_OFFSET,
    SRGB_SCALE,
    TARGET_FRAME_INTERVAL_MS,
    TEMPORAL_RETENTION,
    light_wrap_radius,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from better_backgrounds.matting.compositor import LiveHarmonizer

TENSOR_DIMENSIONS = 4
RGB_DIMENSIONS = 3
ALPHA_DIMENSIONS = 2
RGB_CHANNELS = 3
BACKGROUND_CACHE_LIMIT = 2


@dataclass(frozen=True, slots=True)
class CudaCompositeResult:
    """Return final host pixels after one synchronized CUDA readback."""

    image: NDArray[np.uint8]
    standard_image: NDArray[np.uint8]
    harmonized: bool
    harmonization_ms: float
    harmonization_degraded: tuple[str, ...]
    post_processing_ms: float
    readback_ms: float


@dataclass(frozen=True, slots=True)
class CudaCompositionOptions:
    """Group frame-local CUDA composition choices."""

    captured_at: float
    harmonizer: LiveHarmonizer | None
    reference_background: NDArray[np.uint8] | None
    retain_standard: bool


@dataclass(slots=True)
class TensorAlphaStabilizer:
    """Preserve the current temporal edge policy without leaving CUDA."""

    _previous: torch.Tensor | None = None
    _captured_at: float | None = None

    def apply(self, alpha: torch.Tensor, *, captured_at: float) -> torch.Tensor:
        """Stabilize one B1HW uint8 alpha tensor on its current device."""
        if alpha.dtype != torch.uint8 or alpha.ndim != TENSOR_DIMENSIONS or alpha.shape[1] != 1:
            msg = "tensor alpha must be B1HW uint8"
            raise ValueError(msg)
        if not math.isfinite(captured_at) or captured_at < 0:
            msg = "capture timestamp must be finite and non-negative"
            raise ValueError(msg)
        previous = self._previous
        previous_at = self._captured_at
        elapsed = None if previous_at is None else captured_at - previous_at
        if (
            previous is None
            or previous.shape != alpha.shape
            or elapsed is None
            or elapsed <= 0
            or elapsed > MAXIMUM_STABILIZATION_GAP_MS
        ):
            return self._remember(alpha, captured_at=captured_at)
        difference = torch.abs(alpha.to(torch.int16) - previous.to(torch.int16))
        boundary = ((alpha >= BOUNDARY_ALPHA_MINIMUM) & (alpha <= BOUNDARY_ALPHA_MAXIMUM)) | (
            (previous >= BOUNDARY_ALPHA_MINIMUM) & (previous <= BOUNDARY_ALPHA_MAXIMUM)
        )
        stable_boundary = boundary & (difference <= MAXIMUM_STABILIZED_DELTA)
        elapsed_frames = max(elapsed / TARGET_FRAME_INTERVAL_MS, 1.0)
        retention = TEMPORAL_RETENTION**elapsed_frames
        blended = (
            torch.lerp(
                alpha.to(torch.float32),
                previous.to(torch.float32),
                retention,
            )
            .round()
            .to(torch.uint8)
        )
        return self._remember(
            torch.where(stable_boundary, blended, alpha),
            captured_at=captured_at,
        )

    def _remember(self, alpha: torch.Tensor, *, captured_at: float) -> torch.Tensor:
        remembered = alpha.clone()
        self._previous = remembered
        self._captured_at = captured_at
        return remembered


class CudaLiveEngine:
    """Retain room tensors and fuse full-resolution post-processing on CUDA."""

    def __init__(self) -> None:
        """Bind the engine to the current CUDA device for its process lifetime."""
        if not torch.cuda.is_available():
            msg = "CUDA is unavailable"
            raise RuntimeError(msg)
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._backgrounds: dict[
            tuple[int, tuple[int, ...]],
            tuple[NDArray[np.uint8], torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def available() -> bool:
        """Return whether a fused CUDA session can be created."""
        return torch.cuda.is_available()

    @staticmethod
    def supports(harmonizer: LiveHarmonizer | None) -> bool:
        """Use tensor PIH or the standard path; retain other portable backends."""
        return (
            harmonizer is None
            or not harmonizer.active
            or isinstance(harmonizer, PihAppearanceHarmonizer)
        )

    def clear_backgrounds(self) -> None:
        """Release tensors invalidated by a renderer revision or output size change."""
        self._backgrounds.clear()

    def compose(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        options: CudaCompositionOptions,
    ) -> CudaCompositeResult:
        """Upload exact-frame inputs once and read back only requested final images."""
        source_u8 = self.upload_frame(source)
        alpha_u8 = self.upload_alpha(alpha)
        return self.compose_uploaded(
            source_u8,
            alpha_u8,
            background,
            options,
        )

    def compose_uploaded(
        self,
        source_u8: torch.Tensor,
        alpha_u8: torch.Tensor,
        background: NDArray[np.uint8],
        options: CudaCompositionOptions,
    ) -> CudaCompositeResult:
        """Fuse post-processing from live tensors already owned by this CUDA process."""
        if not self.supports(options.harmonizer):
            msg = "configured harmonizer does not expose a CUDA tensor entry point"
            raise RuntimeError(msg)
        if source_u8.device != self._device or alpha_u8.device != self._device:
            msg = "live tensors must be on the CUDA engine device"
            raise ValueError(msg)
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        background_float, background_linear, wrapped_light = self._background_tensor(background)
        source_float = source_u8.to(torch.float32).div_(255.0)
        alpha_float = alpha_u8.to(torch.float32).div_(255.0)
        foreground_float = self._decontaminate(source_float, alpha_float)
        standard = self._standard_composite(
            foreground_float,
            alpha_float,
            background_linear,
            wrapped_light,
        )
        primary = standard
        harmonizer = options.harmonizer
        harmonized = harmonizer is not None and harmonizer.active
        harmonization_ms = 0.0
        degraded: tuple[str, ...] = ()
        if harmonized and isinstance(harmonizer, PihAppearanceHarmonizer):
            reference_float = (
                background_float
                if options.reference_background is None
                else self._background_tensor(options.reference_background)[0]
            )
            result = harmonizer.apply_tensors(
                foreground_float,
                alpha_float,
                background_float,
                captured_at=options.captured_at,
                reference_background=reference_float,
            )
            harmonization_ms = result.processing_ms
            degraded = result.degraded_components
            harmonized = result.applied
            if result.image is not None:
                primary = (
                    self._add_light_wrap(result.image, alpha_float, wrapped_light)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                )
        torch.cuda.synchronize(self._device)
        post_processing_ms = (time.perf_counter() - started) * 1_000.0
        readback_started = time.perf_counter()
        image = self._rgb_array(primary)
        standard_image = self._rgb_array(standard) if options.retain_standard else image
        readback_ms = (time.perf_counter() - readback_started) * 1_000.0
        return CudaCompositeResult(
            image=image,
            standard_image=standard_image,
            harmonized=harmonized,
            harmonization_ms=harmonization_ms,
            harmonization_degraded=degraded,
            post_processing_ms=post_processing_ms,
            readback_ms=readback_ms,
        )

    def upload_frame(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Upload a full-resolution source once for matting and composition."""
        return self._rgb_tensor(image)

    def upload_alpha(self, alpha: NDArray[np.uint8]) -> torch.Tensor:
        """Upload alpha for compatibility callers using the NumPy matting API."""
        return self._alpha_tensor(alpha)

    def _background_tensor(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (id(image), image.shape)
        cached = self._backgrounds.get(key)
        if cached is not None and cached[0] is image:
            return cached[1], cached[2], cached[3]
        if cached is not None:
            self._backgrounds.pop(key)
        floating = self._rgb_tensor(image).to(torch.float32).div_(255.0)
        linear = self._decode_srgb(floating)
        wrapped_light = self._blur_background(linear)
        while len(self._backgrounds) >= BACKGROUND_CACHE_LIMIT:
            oldest = next(iter(self._backgrounds))
            self._backgrounds.pop(oldest)
        self._backgrounds[key] = (image, floating, linear, wrapped_light)
        return floating, linear, wrapped_light

    def _rgb_tensor(self, image: NDArray[np.uint8]) -> torch.Tensor:
        if (
            image.dtype != np.uint8
            or image.ndim != RGB_DIMENSIONS
            or image.shape[2] != RGB_CHANNELS
        ):
            msg = "CUDA RGB input must be HWC uint8"
            raise ValueError(msg)
        contiguous = np.ascontiguousarray(image)
        return torch.from_numpy(contiguous).to(self._device).permute(2, 0, 1).unsqueeze(0)

    def _alpha_tensor(self, alpha: NDArray[np.uint8]) -> torch.Tensor:
        if alpha.dtype != np.uint8 or alpha.ndim != ALPHA_DIMENSIONS:
            msg = "CUDA alpha input must be HW uint8"
            raise ValueError(msg)
        contiguous = np.ascontiguousarray(alpha)
        return torch.from_numpy(contiguous).to(self._device).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _decontaminate(source: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        uncertain = (alpha >= BOUNDARY_ALPHA_MINIMUM / 255.0) & (
            alpha <= BOUNDARY_ALPHA_MAXIMUM / 255.0
        )
        if not bool(uncertain.any()):
            return source
        height, width = alpha.shape[-2:]
        estimate_size = (
            max(1, round(height * BACKGROUND_ESTIMATE_SCALE)),
            max(1, round(width * BACKGROUND_ESTIMATE_SCALE)),
        )
        linear_source = CudaLiveEngine._decode_srgb(source)
        small_source = functional.interpolate(linear_source, estimate_size, mode="area")
        small_alpha = functional.interpolate(alpha, estimate_size, mode="area")
        weight = (1.0 - small_alpha).pow(4.0)
        kernel_height, kernel_width = BACKGROUND_ESTIMATE_KERNEL
        padding = (
            kernel_width // 2,
            kernel_width // 2,
            kernel_height // 2,
            kernel_height // 2,
        )
        numerator = functional.avg_pool2d(
            functional.pad(small_source * weight, padding, mode="replicate"),
            BACKGROUND_ESTIMATE_KERNEL,
            stride=1,
            divisor_override=1,
        )
        denominator = functional.avg_pool2d(
            functional.pad(weight, padding, mode="replicate"),
            BACKGROUND_ESTIMATE_KERNEL,
            stride=1,
            divisor_override=1,
        )
        estimate = (numerator / denominator.clamp_min(1e-4)).clamp(0.0, 1.0)
        estimate = functional.interpolate(
            estimate,
            (height, width),
            mode="bilinear",
            align_corners=False,
        )
        recovered = (linear_source - (1.0 - alpha) * estimate) / alpha.clamp_min(
            MINIMUM_RECOVERY_ALPHA,
        )
        recovered = recovered.clamp(0.0, 1.0)
        cleaned = torch.lerp(linear_source, recovered, DECONTAMINATION_STRENGTH)
        cleaned = torch.where(uncertain.expand_as(source), cleaned, linear_source)
        encoded = CudaLiveEngine._encode_srgb(cleaned)
        return encoded.mul(255.0).round().div_(255.0)

    @staticmethod
    def _standard_composite(
        source: torch.Tensor,
        alpha: torch.Tensor,
        background_linear: torch.Tensor,
        wrapped_light: torch.Tensor,
    ) -> torch.Tensor:
        source_linear = CudaLiveEngine._decode_srgb(source)
        composite = source_linear * alpha + background_linear * (1.0 - alpha)
        composite = CudaLiveEngine._apply_light_wrap(composite, alpha, wrapped_light)
        return CudaLiveEngine._encode_uint8(composite)

    @staticmethod
    def _decode_srgb(image: torch.Tensor) -> torch.Tensor:
        return torch.where(
            image <= SRGB_DECODE_THRESHOLD,
            image / SRGB_DECODE_DIVISOR,
            ((image + SRGB_OFFSET) / SRGB_SCALE).pow(SRGB_GAMMA),
        )

    @staticmethod
    def _encode_srgb(image: torch.Tensor) -> torch.Tensor:
        clipped = image.clamp(0.0, 1.0)
        return torch.where(
            clipped <= SRGB_ENCODE_THRESHOLD,
            clipped * SRGB_ENCODE_MULTIPLIER,
            SRGB_SCALE * clipped.pow(1.0 / SRGB_GAMMA) - SRGB_OFFSET,
        )

    @staticmethod
    def _blur_background(background: torch.Tensor) -> torch.Tensor:
        height, width = background.shape[-2:]
        radius = light_wrap_radius(height, width)
        kernel_size = radius * 2 + 1
        padding = (radius, radius, radius, radius)
        return functional.avg_pool2d(
            functional.pad(background, padding, mode="replicate"),
            kernel_size,
            stride=1,
        )

    @staticmethod
    def _apply_light_wrap(
        composite: torch.Tensor,
        alpha: torch.Tensor,
        wrapped_light: torch.Tensor,
    ) -> torch.Tensor:
        boundary = 4.0 * alpha * (1.0 - alpha) * LIGHT_WRAP_STRENGTH
        screened = composite + (1.0 - composite) * wrapped_light
        return torch.lerp(composite, screened, boundary).clamp(0.0, 1.0)

    @staticmethod
    def _add_light_wrap(
        composite: torch.Tensor,
        alpha: torch.Tensor,
        wrapped_light: torch.Tensor,
    ) -> torch.Tensor:
        linear = CudaLiveEngine._decode_srgb(composite)
        linear = CudaLiveEngine._apply_light_wrap(linear, alpha, wrapped_light)
        return CudaLiveEngine._encode_srgb(linear)

    @staticmethod
    def _encode_uint8(image: torch.Tensor) -> torch.Tensor:
        return CudaLiveEngine._encode_srgb(image).mul(255.0).round().to(torch.uint8)

    @staticmethod
    def _rgb_array(tensor: torch.Tensor) -> NDArray[np.uint8]:
        return cast(
            "NDArray[np.uint8]",
            tensor[0].permute(1, 2, 0).contiguous().cpu().numpy(),
        )
