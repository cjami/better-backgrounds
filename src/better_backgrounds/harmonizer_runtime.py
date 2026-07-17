"""Global Harmonizer runtime for stable non-commercial webcam compositing."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from better_backgrounds._vendor.harmonizer import HarmonizerInferenceModel
from better_backgrounds.harmonization import (
    EDGE_DECONTAMINATION_STRENGTH,
    AppearanceParameters,
    HarmonizationResult,
    HarmonizationSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

HARMONIZER_CHECKPOINT_ENV = "BETTER_BACKGROUNDS_HARMONIZER_CHECKPOINT"
HARMONIZER_TARGET_FPS = 30.0
HARMONIZER_FRAME_INTERVAL_MS = 1_000.0 / HARMONIZER_TARGET_FPS
HARMONIZER_FRAME_RETENTION = 1.0 - 1.0 / HARMONIZER_TARGET_FPS
RGB_CHANNELS = 3
RGB_DIMENSIONS = 3


def checkpoint_from_environment() -> Path | None:
    """Resolve the explicitly supplied non-commercial checkpoint path."""
    configured = os.environ.get(HARMONIZER_CHECKPOINT_ENV)
    return None if not configured else Path(configured).expanduser().resolve()


class HarmonizerAppearanceHarmonizer:
    """Apply stable global Harmonizer arguments to every exact foreground frame."""

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
        self._arguments: list[torch.Tensor] | None = None
        self._last_argument_at: float | None = None
        self._background: torch.Tensor | None = None
        self._background_key: tuple[int, int, int] | None = None
        self._room_revision: int | None = None
        self._error: str | None = None

    @property
    def active(self) -> bool:
        """Return whether the user requested global harmonization."""
        return self.settings.global_harmonization

    @property
    def backend_name(self) -> str:
        """Describe the selected accelerator or unavailable fallback."""
        return "unavailable" if self._device is None else str(self._device).upper()

    @property
    def error(self) -> str | None:
        """Expose the bounded reason for the latest standard-composite fallback."""
        return self._error

    def configure(self, settings: HarmonizationSettings) -> None:
        """Apply the room-scoped global harmonization switch."""
        self.settings = settings
        if not settings.global_harmonization:
            self.reset_camera()

    def set_room(self, background: NDArray[np.uint8], *, revision: int) -> None:
        """Reset cached pixels and arguments for a changed room snapshot."""
        if (
            background.dtype != np.uint8
            or background.ndim != RGB_DIMENSIONS
            or background.shape[2] != RGB_CHANNELS
        ):
            msg = "background must be uint8 RGB"
            raise ValueError(msg)
        self._room_revision = revision
        self._background = None
        self._background_key = None
        self.reset_camera()

    def clear_room(self) -> None:
        """Discard cached room pixels and live arguments."""
        self._room_revision = None
        self._background = None
        self._background_key = None
        self.reset_camera()

    def reset_camera(self) -> None:
        """Discard global arguments that must not cross camera identities."""
        self._arguments = None
        self._last_argument_at = None

    def apply(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        *,
        captured_at: float,
    ) -> HarmonizationResult:
        """Predict, smooth, and apply global arguments to one exact frame."""
        started = time.perf_counter()
        if not self.settings.global_harmonization:
            return self._fallback(source, started, ())
        if self._room_revision is None:
            return self._fallback(source, started, ("room",))
        try:
            model, device = self._ensure_model()
            composite, matte = self._prepare_inputs(source, alpha, background, device)
            with torch.inference_mode():
                predicted = model.predict_arguments(composite, matte)
                arguments = self._smooth_arguments(predicted, captured_at)
                output = model.restore_image(composite, matte, arguments)
            pixels = output.squeeze(0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
            host = cast("NDArray[np.uint8]", pixels.cpu().numpy())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._error = str(error)[:240]
            return self._fallback(source, started, ("harmonizer",))
        self._error = None
        return HarmonizationResult(
            foreground=None,
            image=host,
            parameters=AppearanceParameters(),
            statistics_updated=True,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            statistics_ms=0.0,
            degraded_components=(),
            applied=True,
        )

    def _ensure_model(self) -> tuple[HarmonizerInferenceModel, torch.device]:
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

    def _prepare_inputs(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            source.dtype != np.uint8
            or source.ndim != RGB_DIMENSIONS
            or source.shape[2] != RGB_CHANNELS
        ):
            msg = "source must be uint8 RGB"
            raise ValueError(msg)
        if alpha.dtype != np.uint8 or alpha.shape != source.shape[:2]:
            msg = "alpha must be uint8 and match source"
            raise ValueError(msg)
        if background.dtype != np.uint8 or background.shape != source.shape:
            msg = "background must be uint8 RGB and match source"
            raise ValueError(msg)
        source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to(
            device=device,
            dtype=torch.float32,
        )
        source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).div(255.0)
        matte = torch.from_numpy(np.ascontiguousarray(alpha)).to(device=device, dtype=torch.float32)
        matte = matte.unsqueeze(0).unsqueeze(0).div(255.0)
        background_tensor = self._cached_background(background, device)
        uncertainty = matte * (1.0 - matte) * 4.0
        clean_source = (
            source_tensor
            + (source_tensor - background_tensor) * uncertainty * EDGE_DECONTAMINATION_STRENGTH
        )
        clean_source.clamp_(0.0, 1.0)
        composite = clean_source * matte + background_tensor * (1.0 - matte)
        return composite, matte

    def _cached_background(
        self,
        background: NDArray[np.uint8],
        device: torch.device,
    ) -> torch.Tensor:
        revision = self._room_revision
        if revision is None:
            msg = "Harmonizer background cache requires a room revision"
            raise RuntimeError(msg)
        key = (revision, background.shape[0], background.shape[1])
        if self._background is not None and self._background_key == key:
            return self._background
        tensor = torch.from_numpy(np.ascontiguousarray(background)).to(
            device=device,
            dtype=torch.float32,
        )
        self._background = tensor.permute(2, 0, 1).unsqueeze(0).div(255.0)
        self._background_key = key
        return self._background

    def _smooth_arguments(
        self,
        predicted: list[torch.Tensor],
        captured_at: float,
    ) -> list[torch.Tensor]:
        detached = [argument.detach() for argument in predicted]
        previous = self._arguments
        previous_at = self._last_argument_at
        if previous is None or previous_at is None:
            self._arguments = detached
            self._last_argument_at = captured_at
            return detached
        elapsed_frames = max(captured_at - previous_at, 0.0) / HARMONIZER_FRAME_INTERVAL_MS
        retention = HARMONIZER_FRAME_RETENTION**elapsed_frames
        smoothed = [
            retention * old + (1.0 - retention) * new
            for old, new in zip(previous, detached, strict=True)
        ]
        self._arguments = smoothed
        self._last_argument_at = captured_at
        return smoothed

    @staticmethod
    def _fallback(
        source: NDArray[np.uint8],
        started: float,
        degraded: tuple[str, ...],
    ) -> HarmonizationResult:
        return HarmonizationResult(
            foreground=source,
            image=None,
            parameters=AppearanceParameters(),
            statistics_updated=False,
            processing_ms=(time.perf_counter() - started) * 1_000.0,
            statistics_ms=0.0,
            degraded_components=degraded,
            applied=False,
        )
