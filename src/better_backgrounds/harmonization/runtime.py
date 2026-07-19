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
from better_backgrounds.harmonization.filters import HarmonizerFilterRenderer

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

HARMONIZER_CHECKPOINT_ENV = "BETTER_BACKGROUNDS_HARMONIZER_CHECKPOINT"
SESSION_PREDICTION_SAMPLE_COUNT = 3
SESSION_PREDICTION_SAMPLE_INTERVAL_MS = 100.0
SESSION_TRANSITION_MS = 750.0
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3


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
        self._renderer: HarmonizerFilterRenderer | None = None
        self._prediction_samples: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = []
        self._last_sample_at: float | None = None
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

    def set_room(
        self,
        background: NDArray[np.uint8],
        *,
        revision: int,
    ) -> None:
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
        self._renderer = None
        self._prediction_samples.clear()
        self._last_sample_at = None
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
        """Predict, smooth, and apply global controls to one exact frame."""
        started = time.perf_counter()
        degraded = self._unavailable_component()
        if degraded is not None:
            return self._fallback(started, degraded)
        try:
            if self._renderer is None:
                self._ensure_model()
            composite = self._prepare_composite(source, alpha, background)
            if self._renderer is None:
                reference = background if reference_background is None else reference_background
                prediction_composite = self._prepare_composite(source, alpha, reference)
                self._collect_prediction_sample(prediction_composite, alpha, captured_at)
                if len(self._prediction_samples) < SESSION_PREDICTION_SAMPLE_COUNT:
                    return self._fallback(started, ())
                arguments = self._predict_arguments(self._prediction_samples)
                reference = np.median(
                    np.stack([sample for sample, _alpha in self._prediction_samples]),
                    axis=0,
                ).astype(np.uint8)
                self._renderer = HarmonizerFilterRenderer.compile(reference, arguments)
                self._prediction_samples.clear()
                self._transition_started_at = captured_at
            transition_started_at = self._transition_started_at
            transition = (
                1.0
                if transition_started_at is None
                else (captured_at - transition_started_at) / SESSION_TRANSITION_MS
            )
            host = self._renderer.render(composite, alpha, transition=transition)
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
        samples: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    ) -> list[float]:
        model, device = self._ensure_model()
        composites = np.stack([composite for composite, _alpha in samples])
        alphas = np.stack([alpha for _composite, alpha in samples])
        image_tensor = torch.from_numpy(np.ascontiguousarray(composites)).to(
            device=device,
            dtype=torch.float32,
        )
        image_tensor = image_tensor.permute(0, 3, 1, 2).div(255.0)
        matte_tensor = torch.from_numpy(np.ascontiguousarray(alphas)).to(
            device=device,
            dtype=torch.float32,
        )
        matte_tensor = matte_tensor.unsqueeze(1).div(255.0)
        with torch.inference_mode():
            return [
                float(torch.median(argument.flatten()).item())
                for argument in model.predict_arguments(image_tensor, matte_tensor)
            ]

    def _collect_prediction_sample(
        self,
        composite: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        captured_at: float,
    ) -> None:
        previous_at = self._last_sample_at
        if (
            previous_at is not None
            and captured_at - previous_at < SESSION_PREDICTION_SAMPLE_INTERVAL_MS
        ):
            return
        width, height = HarmonizerInferenceModel.input_size
        sample_size = (width, height)
        self._prediction_samples.append(
            (
                cast(
                    "NDArray[np.uint8]",
                    cv2.resize(composite, sample_size, interpolation=cv2.INTER_LINEAR),
                ),
                cast(
                    "NDArray[np.uint8]",
                    cv2.resize(alpha, sample_size, interpolation=cv2.INTER_LINEAR),
                ),
            ),
        )
        self._last_sample_at = captured_at

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
        if requested == "cuda":
            if not torch.cuda.is_available():
                msg = "CUDA was requested but is unavailable"
                raise RuntimeError(msg)
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                msg = "Metal was requested but is unavailable"
                raise RuntimeError(msg)
            return torch.device("mps")
        if requested is None or requested == "cpu":
            return torch.device("cpu")
        msg = f"Unsupported Harmonizer device: {requested}"
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
        weight = alpha.astype(np.float32) / 255.0
        composite = cv2.blendLinear(source, background, weight, 1.0 - weight)
        cv2.copyTo(background, np.equal(alpha, 0).astype(np.uint8), composite)
        cv2.copyTo(source, np.equal(alpha, 255).astype(np.uint8), composite)
        return cast("NDArray[np.uint8]", composite)

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
