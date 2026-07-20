"""Opt-in Adobe PIH runtime for frame-local appearance harmonization."""

from __future__ import annotations

import optparse
import os
import pickle
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.serialization import safe_globals

from better_backgrounds._vendor.pih import PihInferenceModel, apply_rgb_curves
from better_backgrounds.harmonization import HarmonizationResult, HarmonizationSettings
from better_backgrounds.harmonization.runtime import (
    SESSION_TRANSITION_MS,
    HarmonizerAppearanceHarmonizer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

HARMONIZATION_BACKEND_ENV = "BETTER_BACKGROUNDS_HARMONIZATION_BACKEND"
PIH_CHECKPOINT_ENV = "BETTER_BACKGROUNDS_PIH_CHECKPOINT"
PIH_DEVICE_ENV = "BETTER_BACKGROUNDS_PIH_DEVICE"
PIH_CURVE_STRENGTH_ENV = "BETTER_BACKGROUNDS_PIH_CURVE_STRENGTH"
DEVELOPMENT_PIH_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / ".tools" / "pih_bench" / "ckpt_g39.pth"
)
DEFAULT_CURVE_STRENGTH = 0.65
CURVE_BLACK_LIFT_RETENTION = 0.8
CURVE_SAMPLE_COUNT = 5
CURVE_SAMPLE_INTERVAL_MS = 100.0
CURVE_RECALIBRATION_INTERVAL_MS = 1_000.0
CURVE_ADAPTATION_TIME_MS = 2_000.0
GAIN_NEW_FRAME_WEIGHT = 0.15
MINIMUM_GAIN = 0.6
MAXIMUM_GAIN = 1.0
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
TENSOR_DIMENSIONS = 4


@dataclass(frozen=True, slots=True)
class TensorHarmonizationResult:
    """Return PIH output on its execution device without forcing a readback."""

    image: torch.Tensor | None
    processing_ms: float
    degraded_components: tuple[str, ...]
    applied: bool


class _PihPredictor(Protocol):
    def predict_curves(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def predict_gain(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
        curves: torch.Tensor,
    ) -> torch.Tensor: ...

    def predict_parameters(
        self,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def pih_checkpoint_from_environment() -> Path | None:
    """Resolve the explicitly supplied PIH checkpoint path."""
    configured = os.environ.get(PIH_CHECKPOINT_ENV)
    return None if not configured else Path(configured).expanduser().resolve()


def create_appearance_harmonizer() -> HarmonizerAppearanceHarmonizer | PihAppearanceHarmonizer:
    """Create the selected backend, preferring a prepared local PIH experiment."""
    checkpoint = pih_checkpoint_from_environment()
    if checkpoint is None and DEVELOPMENT_PIH_CHECKPOINT.is_file():
        checkpoint = DEVELOPMENT_PIH_CHECKPOINT
    configured_backend = os.environ.get(HARMONIZATION_BACKEND_ENV)
    backend = (
        configured_backend.strip().lower()
        if configured_backend is not None
        else ("pih" if checkpoint is not None else "harmonizer")
    )
    if backend == "harmonizer":
        return HarmonizerAppearanceHarmonizer()
    if backend == "pih":
        return PihAppearanceHarmonizer(
            checkpoint,
            preferred_device=os.environ.get(PIH_DEVICE_ENV),
            curve_strength=_curve_strength_from_environment(),
        )
    msg = f"Unsupported harmonization backend: {backend}"
    raise ValueError(msg)


def _curve_strength_from_environment() -> float:
    configured = os.environ.get(PIH_CURVE_STRENGTH_ENV)
    if configured is None:
        return DEFAULT_CURVE_STRENGTH
    try:
        strength = float(configured)
    except ValueError as error:
        msg = f"{PIH_CURVE_STRENGTH_ENV} must be between 0 and 1"
        raise ValueError(msg) from error
    if not 0.0 <= strength <= 1.0:
        msg = f"{PIH_CURVE_STRENGTH_ENV} must be between 0 and 1"
        raise ValueError(msg)
    return strength


class PihAppearanceHarmonizer:
    """Predict PIH curves and local shading for each exact source/matte pair."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        *,
        preferred_device: str | None = None,
        curve_strength: float = DEFAULT_CURVE_STRENGTH,
        model_factory: Callable[[], nn.Module] = PihInferenceModel,
    ) -> None:
        """Configure lazy inference from an externally supplied official checkpoint."""
        self.settings = HarmonizationSettings()
        self.checkpoint = checkpoint or pih_checkpoint_from_environment()
        self.preferred_device = preferred_device
        if not 0.0 <= curve_strength <= 1.0:
            msg = "PIH curve strength must be between 0 and 1"
            raise ValueError(msg)
        self.curve_strength = curve_strength
        self._model_factory = model_factory
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._curve_samples: list[torch.Tensor] = []
        self._locked_curves: torch.Tensor | None = None
        self._previous_curves: torch.Tensor | None = None
        self._last_curve_sample_at: float | None = None
        self._previous_gain_mean: torch.Tensor | None = None
        self._gain_designs: dict[
            tuple[torch.device, torch.dtype, int, int],
            torch.Tensor,
        ] = {}
        self._background_tensors: dict[
            int,
            tuple[NDArray[np.uint8], torch.Tensor, torch.Tensor],
        ] = {}
        self._transition_started_at: float | None = None
        self._room_revision: int | None = None
        self._error: str | None = None
        self._model_lock = threading.Lock()
        self._preparing = False

    @property
    def active(self) -> bool:
        """Return whether the user requested global harmonization."""
        return self.settings.global_harmonization

    @property
    def backend_name(self) -> str:
        """Describe the active PIH accelerator."""
        device = "UNAVAILABLE" if self._device is None else str(self._device).upper()
        return f"PIH/{device}"

    @property
    def error(self) -> str | None:
        """Expose the bounded reason for the latest standard-composite fallback."""
        return self._error

    @property
    def previous_curves(self) -> torch.Tensor | None:
        """Expose whether this camera/room has temporal curve state."""
        return self._previous_curves

    def prepare(self) -> None:
        """Load and warm the configured checkpoint before live composition."""
        self._preparing = True
        try:
            model, device = self._ensure_model()
            self._warm_model(model, device)
        except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
            self._error = str(error)[:240]
        else:
            self._error = None
        finally:
            self._preparing = False

    def configure(self, settings: HarmonizationSettings) -> None:
        """Apply the room-scoped global harmonization switch."""
        self.settings = settings
        if not settings.global_harmonization:
            self.reset_camera()

    def set_room(
        self,
        background: NDArray[np.uint8],
        *,
        revision: int,
    ) -> None:
        """Reset temporal parameters for a changed room snapshot."""
        self._validate_rgb(background, name="background")
        self._room_revision = revision
        self._background_tensors.clear()
        self._reset_temporal_parameters()

    def clear_room(self) -> None:
        """Discard the active room and temporal appearance parameters."""
        self._room_revision = None
        self._background_tensors.clear()
        self._reset_temporal_parameters()

    def reset_camera(self) -> None:
        """Discard parameters that must not cross camera identities."""
        self._reset_temporal_parameters()

    def _reset_temporal_parameters(self) -> None:
        self._curve_samples.clear()
        self._locked_curves = None
        self._previous_curves = None
        self._last_curve_sample_at = None
        self._previous_gain_mean = None
        self._transition_started_at = None

    def apply(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        *,
        captured_at: float,
        reference_background: NDArray[np.uint8] | None = None,
    ) -> HarmonizationResult:
        """Predict and render PIH parameters for one exact frame."""
        started = time.perf_counter()
        degraded = self._unavailable_component()
        if degraded is not None:
            return self._fallback(started, degraded)
        try:
            model, device = self._ensure_model()
            reference = background if reference_background is None else reference_background
            prediction_composite = self._prepare_composite(source, alpha, reference)
            display_composite = (
                prediction_composite
                if reference is background
                else self._prepare_composite(source, alpha, background)
            )
            prediction_image = self._image_tensor(prediction_composite, device)
            full_image = self._image_tensor(display_composite, device)
            full_mask = self._mask_tensor(alpha, device)
            _, low_background = self._background_tensor_pair(reference, device)
            full_background, _ = self._background_tensor_pair(background, device)
            low_image = functional.interpolate(
                prediction_image,
                PihInferenceModel.input_size,
                mode="bilinear",
                align_corners=False,
            )
            low_mask = functional.interpolate(
                full_mask,
                PihInferenceModel.input_size,
                mode="bilinear",
                align_corners=False,
            )
            predictor = cast("_PihPredictor", model)
            with torch.inference_mode():
                curves = self._session_curves(
                    predictor,
                    low_background,
                    low_image,
                    low_mask,
                    captured_at=captured_at,
                )
                gain = predictor.predict_gain(low_background, low_image, low_mask, curves)
                gain = self._regularize_gain_field(gain, low_mask)
                gain = self._smooth_gain(gain, low_mask)
                image = self._render(
                    (full_image, full_background, full_mask),
                    (curves, gain),
                    captured_at=captured_at,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._error = str(error)[:240]
            return self._fallback(started, ("pih",))
        self._error = None
        return HarmonizationResult(
            image=image,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            degraded_components=(),
            applied=True,
        )

    def apply_tensors(
        self,
        source: torch.Tensor,
        alpha: torch.Tensor,
        background: torch.Tensor,
        *,
        captured_at: float,
        reference_background: torch.Tensor | None = None,
    ) -> TensorHarmonizationResult:
        """Predict and render PIH without intermediate device-to-host transfers."""
        started = time.perf_counter()
        degraded = self._unavailable_component()
        if degraded is not None:
            return TensorHarmonizationResult(
                image=None,
                processing_ms=(time.perf_counter() - started) * 1_000.0,
                degraded_components=degraded,
                applied=False,
            )
        try:
            model, device = self._ensure_model()
            self._validate_frame_tensors(source, alpha, background, device=device)
            reference = background if reference_background is None else reference_background
            self._validate_frame_tensors(source, alpha, reference, device=device)
            prediction_image = source * alpha + reference * (1.0 - alpha)
            display_image = (
                prediction_image
                if reference_background is None
                else source * alpha + background * (1.0 - alpha)
            )
            low_image = functional.interpolate(
                prediction_image,
                PihInferenceModel.input_size,
                mode="bilinear",
                align_corners=False,
            )
            low_mask = functional.interpolate(
                alpha,
                PihInferenceModel.input_size,
                mode="bilinear",
                align_corners=False,
            )
            low_background = functional.interpolate(
                reference,
                PihInferenceModel.input_size,
                mode="bilinear",
                align_corners=False,
            )
            predictor = cast("_PihPredictor", model)
            with torch.inference_mode():
                curves = self._session_curves(
                    predictor,
                    low_background,
                    low_image,
                    low_mask,
                    captured_at=captured_at,
                )
                gain = predictor.predict_gain(low_background, low_image, low_mask, curves)
                gain = self._regularize_gain_field(gain, low_mask)
                gain = self._smooth_gain(gain, low_mask)
                image = self._render_tensor(
                    (display_image, background, alpha),
                    (curves, gain),
                    captured_at=captured_at,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._error = str(error)[:240]
            return TensorHarmonizationResult(
                image=None,
                processing_ms=(time.perf_counter() - started) * 1_000.0,
                degraded_components=("pih",),
                applied=False,
            )
        self._error = None
        return TensorHarmonizationResult(
            image=image,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            degraded_components=(),
            applied=True,
        )

    def _session_curves(
        self,
        predictor: _PihPredictor,
        background: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor,
        *,
        captured_at: float,
    ) -> torch.Tensor:
        locked = self._locked_curves
        last_sample_at = self._last_curve_sample_at
        sample_interval = (
            CURVE_SAMPLE_INTERVAL_MS if locked is None else CURVE_RECALIBRATION_INTERVAL_MS
        )
        if last_sample_at is not None and captured_at - last_sample_at < sample_interval:
            previous = self._previous_curves
            if previous is None:
                msg = "PIH curve sampling state is incomplete"
                raise RuntimeError(msg)
            return previous
        sample = self._anchor_curves(
            predictor.predict_curves(background, composite, mask),
        )
        self._last_curve_sample_at = captured_at
        self._curve_samples.append(sample)
        if len(self._curve_samples) > CURVE_SAMPLE_COUNT:
            self._curve_samples.pop(0)
        samples = torch.stack(self._curve_samples)
        curves = samples.mean(dim=0)
        if len(self._curve_samples) == CURVE_SAMPLE_COUNT:
            curves = self._coherent_curves(samples)
        if locked is not None:
            elapsed = sample_interval if last_sample_at is None else captured_at - last_sample_at
            update_weight = 1.0 - float(np.exp(-elapsed / CURVE_ADAPTATION_TIME_MS))
            curves = torch.lerp(locked, curves, update_weight)
        if len(self._curve_samples) == CURVE_SAMPLE_COUNT:
            self._locked_curves = curves.clone()
        self._previous_curves = curves.clone()
        return curves

    @staticmethod
    def _anchor_curves(curves: torch.Tensor) -> torch.Tensor:
        black = curves[:, :, :1]
        white = curves[:, :, -1:]
        span = (white - black).clamp_min(torch.finfo(curves.dtype).eps)
        retained_black = black * CURVE_BLACK_LIFT_RETENTION
        scale = (white - retained_black) / span
        offset = retained_black - black * scale
        return (curves * scale + offset).clamp(0.0, 1.0)

    @staticmethod
    def _coherent_curves(samples: torch.Tensor) -> torch.Tensor:
        median = samples.median(dim=0).values
        distances = (samples - median).square().flatten(start_dim=1).mean(dim=1)
        return samples[distances.argmin()]

    def _regularize_gain_field(self, gain: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = gain.shape
        dtype = gain.dtype
        device = gain.device
        key = (device, dtype, height, width)
        design = self._gain_designs.get(key)
        if design is None:
            vertical, horizontal = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device),
                torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device),
                indexing="ij",
            )
            design = torch.stack(
                (torch.ones_like(horizontal), horizontal, vertical),
            ).reshape(1, 3, -1)
            self._gain_designs[key] = design
        design = design.expand(batch, -1, -1)
        weight = mask.to(dtype).reshape(batch, 1, -1)
        weighted_design = design * weight
        normal = weighted_design @ design.transpose(1, 2)
        scale = normal.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0)
        regularizer = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        normal = normal + regularizer * (scale * torch.finfo(dtype).eps)[:, None, None]
        target = weighted_design @ gain.reshape(batch, 1, -1).transpose(1, 2)
        coefficients = torch.linalg.solve(normal, target)
        projected = (coefficients.transpose(1, 2) @ design).reshape(batch, 1, height, width)
        return projected.clamp(MINIMUM_GAIN, MAXIMUM_GAIN)

    def _smooth_gain(self, gain: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(gain.dtype)
        current_mean = (gain * weight).sum() / weight.sum().clamp_min(1.0)
        previous_mean = self._previous_gain_mean
        stable_mean = (
            current_mean
            if previous_mean is None
            else torch.lerp(previous_mean, current_mean, GAIN_NEW_FRAME_WEIGHT)
        )
        self._previous_gain_mean = stable_mean.clone()
        scale = stable_mean / current_mean.clamp_min(torch.finfo(gain.dtype).eps)
        return (gain * scale).clamp(MINIMUM_GAIN, MAXIMUM_GAIN)

    def _render(
        self,
        frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        parameters: tuple[torch.Tensor, torch.Tensor],
        *,
        captured_at: float,
    ) -> NDArray[np.uint8]:
        output = self._render_tensor(frame, parameters, captured_at=captured_at)
        return cast(
            "NDArray[np.uint8]",
            output[0]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy(),
        )

    def _render_tensor(
        self,
        frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        parameters: tuple[torch.Tensor, torch.Tensor],
        *,
        captured_at: float,
    ) -> torch.Tensor:
        composite, background, mask = frame
        curves, gain = parameters
        predicted_color = apply_rgb_curves(composite, curves)
        corrected = torch.lerp(composite, predicted_color, self.curve_strength)
        full_gain = functional.interpolate(
            gain,
            composite.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        harmonized = corrected * full_gain * mask + (1.0 - mask) * background
        if self._transition_started_at is None:
            self._transition_started_at = captured_at
        elapsed = captured_at - self._transition_started_at
        transition = float(np.clip(elapsed / SESSION_TRANSITION_MS, 0.0, 1.0))
        return torch.lerp(composite, harmonized, transition)

    @staticmethod
    def _validate_frame_tensors(
        source: torch.Tensor,
        alpha: torch.Tensor,
        background: torch.Tensor,
        *,
        device: torch.device,
    ) -> None:
        expected_alpha = (source.shape[0], 1, *source.shape[-2:])
        if source.ndim != TENSOR_DIMENSIONS or source.shape[1] != RGB_CHANNELS:
            msg = "source tensor must be BCHW RGB"
            raise ValueError(msg)
        if alpha.shape != expected_alpha:
            msg = "alpha tensor must be B1HW and match source"
            raise ValueError(msg)
        if background.shape != source.shape:
            msg = "background tensor must match source"
            raise ValueError(msg)
        tensors = (source, alpha, background)
        if any(tensor.device != device or tensor.dtype != torch.float32 for tensor in tensors):
            msg = "PIH tensors must be FP32 on the model device"
            raise ValueError(msg)

    def _ensure_model(self) -> tuple[nn.Module, torch.device]:
        if self._model is not None and self._device is not None:
            return self._model, self._device
        with self._model_lock:
            if self._model is not None and self._device is not None:
                return self._model, self._device
            checkpoint = self.checkpoint
            if checkpoint is None:
                msg = f"set {PIH_CHECKPOINT_ENV} to the external ckpt_g39.pth checkpoint"
                raise RuntimeError(msg)
            if not checkpoint.is_file():
                msg = f"PIH checkpoint not found: {checkpoint}"
                raise RuntimeError(msg)
            device = self._select_device()
            with safe_globals([optparse.Values]):
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("state_dict") if isinstance(payload, Mapping) else None
            if state is None and isinstance(payload, Mapping):
                state = payload
            if not isinstance(state, Mapping) or not state:
                msg = "PIH checkpoint does not contain a state dictionary"
                raise TypeError(msg)
            model = self._model_factory()
            model.load_state_dict(state, strict=True)
            model.eval().to(device)
            self._model = model
            self._device = device
            return model, device

    @staticmethod
    def _warm_model(model: nn.Module, device: torch.device) -> None:
        predictor = cast("_PihPredictor", model)
        image = torch.zeros((1, 3, *PihInferenceModel.input_size), device=device)
        mask = torch.zeros((1, 1, *PihInferenceModel.input_size), device=device)
        with torch.inference_mode():
            predictor.predict_parameters(image, image, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _select_device(self) -> torch.device:
        requested = self.preferred_device
        if requested is None:
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            msg = "CUDA was requested for PIH but is unavailable"
            raise RuntimeError(msg)
        if requested == "mps" and not torch.backends.mps.is_available():
            msg = "Metal was requested for PIH but is unavailable"
            raise RuntimeError(msg)
        if requested not in {"cpu", "cuda", "mps"}:
            msg = f"Unsupported PIH device: {requested}"
            raise RuntimeError(msg)
        if requested == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device(requested)

    def _prepare_composite(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        self._validate_rgb(source, name="source")
        if alpha.dtype != np.uint8 or alpha.shape != source.shape[:2]:
            msg = "alpha must be uint8 and match source"
            raise ValueError(msg)
        if background.dtype != np.uint8 or background.shape != source.shape:
            msg = "background must be uint8 RGB and match source"
            raise ValueError(msg)
        weight = alpha.astype(np.float32) / 255.0
        composite = cv2.blendLinear(source, background, weight, 1.0 - weight)
        cv2.copyTo(background, np.equal(alpha, 0).astype(np.uint8), composite)
        cv2.copyTo(source, np.equal(alpha, 255).astype(np.uint8), composite)
        return cast("NDArray[np.uint8]", composite)

    @staticmethod
    def _image_tensor(image: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
        return (
            torch.from_numpy(np.require(image, requirements=("C", "W")))
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255.0)
        )

    def _background_tensor_pair(
        self,
        image: NDArray[np.uint8],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(image)
        cached = self._background_tensors.get(key)
        if cached is not None and cached[0] is image:
            return cached[1], cached[2]
        full = self._image_tensor(image, device)
        low = functional.interpolate(
            full,
            PihInferenceModel.input_size,
            mode="bilinear",
            align_corners=False,
        )
        self._background_tensors[key] = (image, full, low)
        return full, low

    @staticmethod
    def _mask_tensor(mask: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
        return (
            torch.from_numpy(np.require(mask, requirements=("C", "W")))
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .div(255.0)
        )

    @staticmethod
    def _validate_rgb(image: NDArray[np.uint8], *, name: str) -> None:
        if (
            image.dtype != np.uint8
            or image.ndim != RGB_DIMENSIONS
            or image.shape[2] != RGB_CHANNELS
        ):
            msg = f"{name} must be uint8 RGB"
            raise ValueError(msg)

    def _unavailable_component(self) -> tuple[str, ...] | None:
        if not self.settings.global_harmonization:
            return ()
        if self._room_revision is None:
            return ("room",)
        if self._preparing:
            return ("pih_loading",)
        if self._error is not None and self._model is None:
            return ("pih",)
        return None

    @staticmethod
    def _fallback(started: float, degraded: tuple[str, ...]) -> HarmonizationResult:
        return HarmonizationResult(
            image=None,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            degraded_components=degraded,
            applied=False,
        )
