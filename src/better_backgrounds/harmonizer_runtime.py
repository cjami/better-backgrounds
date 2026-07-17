"""Global Harmonizer runtime for stable non-commercial webcam compositing."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import torch

from better_backgrounds._vendor.harmonizer import HarmonizerInferenceModel
from better_backgrounds.harmonization import HarmonizationResult, HarmonizationSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

HARMONIZER_CHECKPOINT_ENV = "BETTER_BACKGROUNDS_HARMONIZER_CHECKPOINT"
HARMONIZER_TARGET_FPS = 30.0
HARMONIZER_FRAME_INTERVAL_MS = 1_000.0 / HARMONIZER_TARGET_FPS
HARMONIZER_FRAME_RETENTION = 1.0 - 1.0 / HARMONIZER_TARGET_FPS
HARMONIZER_PREDICTION_INTERVAL_MS = 5_000.0
EDGE_DECONTAMINATION_STRENGTH = 0.12
MINIMUM_GLOBAL_SCALE = 0.75
MAXIMUM_GLOBAL_SCALE = 1.5
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
GLOBAL_ARGUMENT_COUNT = 6
OPAQUE_ALPHA = 255
EDGE_ALPHA_MINIMUM = 16
EDGE_ALPHA_MAXIMUM = 239
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def checkpoint_from_environment() -> Path | None:
    """Resolve the explicitly supplied non-commercial checkpoint path."""
    configured = os.environ.get(HARMONIZER_CHECKPOINT_ENV)
    return None if not configured else Path(configured).expanduser().resolve()


class HarmonizerAppearanceHarmonizer:
    """Predict stable Harmonizer controls and apply a fast global transform."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        *,
        preferred_device: str | None = None,
        model_factory: Callable[[], HarmonizerInferenceModel] = HarmonizerInferenceModel,
    ) -> None:
        """Configure lazy inference from an externally supplied official checkpoint."""
        self.settings = HarmonizationSettings()
        self.checkpoint = checkpoint or checkpoint_from_environment()
        self.preferred_device = preferred_device
        self._model_factory = model_factory
        self._model: HarmonizerInferenceModel | None = None
        self._device: torch.device | None = None
        self._arguments: list[float] | None = None
        self._target_arguments: list[float] | None = None
        self._last_frame_at: float | None = None
        self._last_prediction_at: float | None = None
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
        """Describe the accelerator used only for global-control prediction."""
        return "unavailable" if self._device is None else str(self._device).upper()

    @property
    def error(self) -> str | None:
        """Expose the bounded reason for the latest standard-composite fallback."""
        return self._error

    def prepare(self) -> None:
        """Load the configured checkpoint before it reaches the live frame callback."""
        self._preparing = True
        try:
            self._ensure_model()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
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

    def set_room(self, background: NDArray[np.uint8], *, revision: int) -> None:
        """Reset global controls for a changed room snapshot."""
        self._validate_rgb(background, name="background")
        self._room_revision = revision
        self.reset_camera()

    def clear_room(self) -> None:
        """Discard the active room and live global controls."""
        self._room_revision = None
        self.reset_camera()

    def reset_camera(self) -> None:
        """Discard global controls that must not cross camera identities."""
        self._arguments = None
        self._target_arguments = None
        self._last_frame_at = None
        self._last_prediction_at = None

    def apply(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        *,
        captured_at: float,
    ) -> HarmonizationResult:
        """Predict, smooth, and apply global controls to one exact frame."""
        started = time.perf_counter()
        degraded = self._unavailable_component()
        if degraded is not None:
            return self._fallback(started, degraded)
        try:
            composite = self._prepare_composite(source, alpha, background)
            if self._prediction_due(captured_at):
                self._target_arguments = self._predict_arguments(composite, alpha)
                self._last_prediction_at = captured_at
            arguments = self._smooth_arguments(captured_at)
            host = self._restore_global(composite, alpha, arguments)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._error = str(error)[:240]
            return self._fallback(started, ("harmonizer",))
        self._error = None
        return HarmonizationResult(
            image=host,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            degraded_components=(),
            applied=True,
        )

    def _unavailable_component(self) -> tuple[str, ...] | None:
        if not self.settings.global_harmonization:
            return ()
        if self._room_revision is None:
            return ("room",)
        if self._preparing and self._model is None:
            return ("harmonizer_loading",)
        if self._error is not None and self._model is None:
            return ("harmonizer",)
        return None

    def _predict_arguments(
        self,
        composite: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
    ) -> list[float]:
        model, device = self._ensure_model()
        image_tensor = torch.from_numpy(np.ascontiguousarray(composite)).to(
            device=device,
            dtype=torch.float32,
        )
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).div(255.0)
        matte_tensor = torch.from_numpy(np.ascontiguousarray(alpha)).to(
            device=device,
            dtype=torch.float32,
        )
        matte_tensor = matte_tensor.unsqueeze(0).unsqueeze(0).div(255.0)
        with torch.inference_mode():
            return [
                float(argument.item())
                for argument in model.predict_arguments(image_tensor, matte_tensor)
            ]

    def _ensure_model(self) -> tuple[HarmonizerInferenceModel, torch.device]:
        if self._model is not None and self._device is not None:
            return self._model, self._device
        with self._model_lock:
            if self._model is not None and self._device is not None:
                return self._model, self._device
            checkpoint = self.checkpoint
            if checkpoint is None:
                msg = f"set {HARMONIZER_CHECKPOINT_ENV} to the external harmonizer.pth checkpoint"
                raise RuntimeError(msg)
            if not checkpoint.is_file():
                msg = f"Harmonizer checkpoint not found: {checkpoint}"
                raise RuntimeError(msg)
            device = self._select_device()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or not payload:
                msg = "Harmonizer checkpoint is not a state dictionary"
                raise TypeError(msg)
            model = self._model_factory()
            model.load_state_dict(payload, strict=True)
            model.eval().to(device)
            self._model = model
            self._device = device
            return model, device

    def _select_device(self) -> torch.device:
        requested = self.preferred_device
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" or (requested is None and torch.cuda.is_available()):
            if not torch.cuda.is_available():
                msg = "CUDA was requested but is unavailable"
                raise RuntimeError(msg)
            return torch.device("cuda")
        if requested == "mps" or (requested is None and torch.backends.mps.is_available()):
            if not torch.backends.mps.is_available():
                msg = "Metal was requested but is unavailable"
                raise RuntimeError(msg)
            return torch.device("mps")
        msg = "Harmonizer requires CUDA or Metal acceleration"
        raise RuntimeError(msg)

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
        clean_source = source.copy()
        uncertain = (alpha >= EDGE_ALPHA_MINIMUM) & (alpha <= EDGE_ALPHA_MAXIMUM)
        if np.any(uncertain):
            weights = alpha[uncertain].astype(np.float32) / 255.0
            uncertainty = weights * (1.0 - weights) * 4.0 * EDGE_DECONTAMINATION_STRENGTH
            source_edge = source[uncertain].astype(np.float32)
            background_edge = background[uncertain].astype(np.float32)
            cleaned = source_edge + (source_edge - background_edge) * uncertainty[:, None]
            clean_source[uncertain] = np.rint(np.clip(cleaned, 0, 255)).astype(np.uint8)
        weight = alpha.astype(np.float32) / 255.0
        composite = cv2.blendLinear(clean_source, background, weight, 1.0 - weight)
        cv2.copyTo(background, np.equal(alpha, 0).astype(np.uint8), composite)
        cv2.copyTo(
            clean_source,
            np.equal(alpha, OPAQUE_ALPHA).astype(np.uint8),
            composite,
        )
        return cast("NDArray[np.uint8]", composite)

    @staticmethod
    def _restore_global(
        composite: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        arguments: list[float],
    ) -> NDArray[np.uint8]:
        if len(arguments) != GLOBAL_ARGUMENT_COUNT:
            msg = "Harmonizer must predict six global arguments"
            raise ValueError(msg)
        temperature, brightness, contrast, saturation, highlight, shadow = np.clip(
            arguments,
            -1.0,
            1.0,
        )
        temperature_gains = np.array(
            [1.0 + 0.15 * temperature, 1.0, 1.0 - 0.15 * temperature],
            dtype=np.float32,
        )
        brightness_scale = 1.0 / (1.0 - brightness + 1e-6) if brightness >= 0 else 1.0 + brightness
        brightness_scale = float(
            np.clip(brightness_scale, MINIMUM_GLOBAL_SCALE, MAXIMUM_GLOBAL_SCALE),
        )
        contrast_scale = (
            255.0 / (256.0 - np.floor(contrast * 255.0)) if contrast > 0 else 1.0 + contrast
        )
        contrast_scale = float(
            np.clip(contrast_scale, MINIMUM_GLOBAL_SCALE, MAXIMUM_GLOBAL_SCALE),
        )
        saturation_scale = float(
            np.clip(1.0 + saturation, MINIMUM_GLOBAL_SCALE, MAXIMUM_GLOBAL_SCALE),
        )
        saturation_matrix = np.eye(RGB_CHANNELS, dtype=np.float32) * saturation_scale + np.outer(
            np.ones(RGB_CHANNELS, dtype=np.float32), LUMINANCE_WEIGHTS
        ) * (1.0 - saturation_scale)
        matrix = saturation_matrix @ np.diag(temperature_gains) * brightness_scale * contrast_scale
        offset = np.full(
            (RGB_CHANNELS, 1),
            127.5 * (1.0 - contrast_scale),
            dtype=np.float32,
        )
        transformed = cv2.transform(composite, np.concatenate((matrix, offset), axis=1))
        values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        values = 1.0 - np.power(1.0 - values + 1e-9, highlight + 1.0)
        values = np.power(values + 1e-9, -shadow + 1.0)
        lookup = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        transformed = cv2.LUT(transformed, lookup)
        weight = alpha.astype(np.float32) / 255.0
        restored = cv2.blendLinear(transformed, composite, weight, 1.0 - weight)
        cv2.copyTo(composite, np.equal(alpha, 0).astype(np.uint8), restored)
        cv2.copyTo(
            transformed,
            np.equal(alpha, OPAQUE_ALPHA).astype(np.uint8),
            restored,
        )
        return cast("NDArray[np.uint8]", restored)

    def _prediction_due(self, captured_at: float) -> bool:
        previous_at = self._last_prediction_at
        return (
            self._target_arguments is None
            or previous_at is None
            or captured_at - previous_at >= HARMONIZER_PREDICTION_INTERVAL_MS
        )

    def _smooth_arguments(self, captured_at: float) -> list[float]:
        target = self._target_arguments
        if target is None:
            msg = "Harmonizer did not predict global arguments"
            raise RuntimeError(msg)
        previous = self._arguments
        previous_at = self._last_frame_at
        if previous is None or previous_at is None:
            self._arguments = target
            self._last_frame_at = captured_at
            return target
        elapsed_frames = max(captured_at - previous_at, 0.0) / HARMONIZER_FRAME_INTERVAL_MS
        retention = HARMONIZER_FRAME_RETENTION**elapsed_frames
        smoothed = [
            retention * old + (1.0 - retention) * new
            for old, new in zip(previous, target, strict=True)
        ]
        self._arguments = smoothed
        self._last_frame_at = captured_at
        return smoothed

    @staticmethod
    def _validate_rgb(image: NDArray[np.uint8], *, name: str) -> None:
        if (
            image.dtype != np.uint8
            or image.ndim != RGB_DIMENSIONS
            or image.shape[2] != RGB_CHANNELS
        ):
            msg = f"{name} must be uint8 RGB"
            raise ValueError(msg)

    @staticmethod
    def _fallback(started: float, degraded: tuple[str, ...]) -> HarmonizationResult:
        return HarmonizationResult(
            image=None,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            degraded_components=degraded,
            applied=False,
        )
