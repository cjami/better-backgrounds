"""Pinned MatAnyone 2 assets and stateful per-frame inference adapter."""

from __future__ import annotations

import hashlib
import importlib
import platform
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from better_backgrounds.matting.contracts import MattingCapabilities, MattingConfig

if TYPE_CHECKING:
    from types import ModuleType

    import torch
    from numpy.typing import NDArray

MATANYONE2_REVISION = "d3bb5a1ebedf259a5453c6d168e6840fff85581e"
RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
TENSOR_DIMENSIONS = 4
DeviceRequest = Literal["auto", "cuda", "mps", "cpu"]
DeviceType = Literal["cuda", "mps", "cpu"]


class CheckpointMetadata(BaseModel):
    """Describe one immutable bundled model checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal["matanyone2.pth"]
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MatAnyoneAssetManifest(BaseModel):
    """Record the exact non-commercial upstream runtime and checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    name: Literal["MatAnyone 2"]
    version: str
    upstream_revision: Literal["d3bb5a1ebedf259a5453c6d168e6840fff85581e"]
    source: HttpUrl
    license: Literal["S-Lab-1.0-NC"]
    checkpoint: CheckpointMetadata


def load_matanyone_asset_manifest() -> MatAnyoneAssetManifest:
    """Load the package-owned MatAnyone 2 asset manifest."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/matanyone2/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    return MatAnyoneAssetManifest.model_validate_json(content)


def packaged_checkpoint_path(*, verify: bool = True) -> Path:
    """Return the filesystem checkpoint path after optional integrity checking."""
    manifest = load_matanyone_asset_manifest()
    resource = files("better_backgrounds").joinpath(
        "assets",
        "matanyone2",
        manifest.checkpoint.path,
    )
    path = Path(str(resource))
    if not path.is_file():
        msg = "Bundled MatAnyone 2 checkpoint is missing"
        raise FileNotFoundError(msg)
    if verify:
        verify_checkpoint_path(path)
    return path


def verify_checkpoint_path(path: Path) -> None:
    """Verify a checkpoint in a background worker before Torch deserializes it."""
    manifest = load_matanyone_asset_manifest()
    hasher = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while block := checkpoint.read(1024 * 1024):
            hasher.update(block)
    digest = hasher.hexdigest()
    if path.stat().st_size != manifest.checkpoint.size or digest != manifest.checkpoint.sha256:
        msg = "Bundled MatAnyone 2 checkpoint failed integrity verification"
        raise ValueError(msg)


def _vendor_root() -> Path:
    """Return the package-level directory containing pinned upstream code."""
    return Path(str(files("better_backgrounds").joinpath("_vendor"))).resolve()


def ensure_vendored_matanyone() -> ModuleType:
    """Load the pinned vendored package under its upstream import name."""
    vendor_root = _vendor_root()
    existing = sys.modules.get("matanyone2")
    if existing is not None:
        module_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if vendor_root not in module_file.parents:
            msg = "A non-vendored matanyone2 package is already loaded"
            raise RuntimeError(msg)
        return existing
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    return importlib.import_module("matanyone2")


def resolve_device(requested: DeviceRequest = "auto") -> DeviceType:
    """Choose CUDA, Apple MPS, then the functional CPU fallback."""
    torch = importlib.import_module("torch")
    available = {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(torch.backends.mps.is_built() and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "cuda":
        if available["cuda"]:
            return "cuda"
        msg = "Requested MatAnyone 2 device is unavailable: cuda"
        raise RuntimeError(msg)
    if requested == "mps":
        if available["mps"]:
            return "mps"
        msg = "Requested MatAnyone 2 device is unavailable: mps"
        raise RuntimeError(msg)
    if requested == "cpu":
        return "cpu"
    if available["cuda"]:
        return "cuda"
    if available["mps"]:
        return "mps"
    return "cpu"


class MatAnyoneRuntime:
    """Adapt the upstream stateful step API to RGB NumPy frames."""

    def __init__(self, checkpoint: Path, config: MattingConfig) -> None:
        """Load one checkpoint and initialize an empty temporal memory."""
        ensure_vendored_matanyone()
        torch = importlib.import_module("torch")
        hydra = importlib.import_module("hydra")
        omegaconf = importlib.import_module("omegaconf")
        model_module = importlib.import_module("matanyone2.model.matanyone2")
        inference_module = importlib.import_module("matanyone2.inference.inference_core")
        self._torch = torch
        self.device = resolve_device(config.device)
        config_dir = _vendor_root() / "matanyone2" / "config"
        with hydra.initialize_config_dir(
            version_base="1.3.2",
            config_dir=str(config_dir),
            job_name="better_backgrounds_live",
        ):
            upstream_config = hydra.compose(config_name="eval_matanyone_config")
        with omegaconf.open_dict(upstream_config):
            upstream_config.weights = str(checkpoint)
            upstream_config.max_internal_size = -1
            upstream_config.model.pretrained_resnet = False
        network = model_module.MatAnyone2(upstream_config, single_object=True)
        weights = torch.load(checkpoint, map_location=self.device, weights_only=True)
        network.load_weights(weights)
        network.to(self.device).eval()
        self._network = network
        self._inference_class = inference_module.InferenceCore
        self._upstream_config = upstream_config
        self._core: Any = self._new_core()
        self.config = config
        self.capabilities = MattingCapabilities(
            device_type=self.device,
            accelerated=self.device in {"cuda", "mps"},
        )

    def calibration_device_identity(self) -> tuple[str, str, str]:
        """Return stable local runtime fields used to invalidate calibration."""
        torch = self._torch
        if self.device == "cuda":
            device_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
            accelerator_version = str(torch.version.cuda or "unknown")
        elif self.device == "mps":
            device_name = f"Apple MPS ({platform.machine()})"
            accelerator_version = platform.mac_ver()[0] or "unknown"
        else:
            device_name = platform.processor() or platform.machine() or "CPU"
            accelerator_version = "cpu"
        return device_name, str(torch.__version__), accelerator_version

    def reconfigure(self, config: MattingConfig) -> None:
        """Clear temporal memory while retaining the loaded network weights."""
        self.config = config
        self._core = self._new_core()

    def initialize(self, frame: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Seed the target person and repeat the first frame for temporal warm-up."""
        if mask.dtype != np.uint8 or mask.shape != frame.shape[:2]:
            msg = "seed mask must match the RGB frame"
            raise ValueError(msg)
        original_size = (frame.shape[1], frame.shape[0])
        resized_frame, resized_mask = self._resize_for_inference(frame, mask)
        if resized_mask is None:
            msg = "seed mask was lost while preparing inference"
            raise RuntimeError(msg)
        image = self._image_tensor(resized_frame)
        seed = self._torch.from_numpy(np.ascontiguousarray(resized_mask)).float().to(self.device)
        with self._torch.inference_mode():
            output = self._core.step(image, seed, objects=[1])
            for _index in range(self.config.warmup_iterations):
                output = self._core.step(image, first_frame_pred=True)
        return self._alpha_array(output, output_size=original_size)

    def step(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Produce one full-resolution alpha matte using retained temporal memory."""
        original_size = (frame.shape[1], frame.shape[0])
        frame, _mask = self._resize_for_inference(frame)
        with self._torch.inference_mode():
            output = self._core.step(self._image_tensor(frame))
        return self._alpha_array(output, output_size=original_size)

    def initialize_tensor(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> torch.Tensor:
        """Seed temporal memory and return a full-resolution device alpha tensor."""
        if mask.dtype != np.uint8 or mask.shape != frame.shape[:2]:
            msg = "seed mask must match the RGB frame"
            raise ValueError(msg)
        output_size = (frame.shape[0], frame.shape[1])
        resized_frame, resized_mask = self._resize_for_inference(frame, mask)
        if resized_mask is None:
            msg = "seed mask was lost while preparing inference"
            raise RuntimeError(msg)
        image = self._image_tensor(resized_frame)
        seed = self._torch.from_numpy(np.ascontiguousarray(resized_mask)).float().to(self.device)
        with self._torch.inference_mode():
            output = self._core.step(image, seed, objects=[1])
            for _index in range(self.config.warmup_iterations):
                output = self._core.step(image, first_frame_pred=True)
        return self._alpha_tensor(output, output_size=output_size)

    def step_tensor(self, frame: NDArray[np.uint8]) -> torch.Tensor:
        """Return full-resolution uint8 alpha on the execution device."""
        output_size = (frame.shape[0], frame.shape[1])
        inference_frame, _mask = self._resize_for_inference(frame)
        with self._torch.inference_mode():
            output = self._core.step(self._image_tensor(inference_frame))
        return self._alpha_tensor(output, output_size=output_size)

    def step_uploaded(self, source: torch.Tensor) -> torch.Tensor:
        """Run one live step from an already-uploaded full-resolution RGB tensor."""
        if (
            source.device.type != self.device
            or source.dtype != self._torch.uint8
            or source.ndim != TENSOR_DIMENSIONS
            or source.shape[0] != 1
            or source.shape[1] != RGB_CHANNELS
        ):
            msg = "uploaded MatAnyone source must be BCHW RGB uint8 on the runtime device"
            raise ValueError(msg)
        output_size = (int(source.shape[-2]), int(source.shape[-1]))
        image = source.to(dtype=self._torch.float32).div_(255.0)
        shortest = min(output_size)
        if shortest > self.config.internal_size:
            scale = self.config.internal_size / shortest
            inference_size = (
                round(output_size[0] * scale),
                round(output_size[1] * scale),
            )
            image = self._torch.nn.functional.interpolate(
                image,
                inference_size,
                mode="area",
            )
        with self._torch.inference_mode():
            output = self._core.step(image[0])
        return self._alpha_tensor(output, output_size=output_size)

    def synchronize(self) -> None:
        """Wait for asynchronous accelerator work before recording latency."""
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    def close(self) -> None:
        """Release model memory owned by the worker process."""
        self._core = None
        self._network = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()

    def _image_tensor(self, frame: NDArray[np.uint8]):  # noqa: ANN202
        if (
            frame.dtype != np.uint8
            or frame.ndim != RGB_DIMENSIONS
            or frame.shape[2] != RGB_CHANNELS
        ):
            msg = "MatAnyone 2 input must be uint8 RGB"
            raise ValueError(msg)
        contiguous = np.ascontiguousarray(frame)
        return (
            self._torch.from_numpy(contiguous)
            .permute(2, 0, 1)
            .to(self.device, dtype=self._torch.float32)
            .div_(255.0)
        )

    def _new_core(self):  # noqa: ANN202
        return self._inference_class(
            self._network,
            cfg=self._upstream_config,
            device=self.device,
        )

    def _resize_for_inference(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None]:
        height, width = frame.shape[:2]
        shortest = min(height, width)
        if shortest <= self.config.internal_size:
            return frame, mask
        scale = self.config.internal_size / shortest
        resized_size = (round(width * scale), round(height * scale))
        resized_frame = cast(
            "NDArray[np.uint8]",
            cv2.resize(frame, resized_size, interpolation=cv2.INTER_AREA),
        )
        resized_mask = (
            None
            if mask is None
            else cast(
                "NDArray[np.uint8]",
                cv2.resize(mask, resized_size, interpolation=cv2.INTER_NEAREST),
            )
        )
        return resized_frame, resized_mask

    def _alpha_array(
        self,
        output,  # noqa: ANN001
        *,
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        matte = self._core.output_prob_to_mask(output)
        alpha = matte.detach().float().clamp(0.0, 1.0).mul(255.0).round().byte()
        array = cast("NDArray[np.uint8]", alpha.cpu().numpy())
        if (array.shape[1], array.shape[0]) != output_size:
            array = cast(
                "NDArray[np.uint8]",
                cv2.resize(array, output_size, interpolation=cv2.INTER_LINEAR),
            )
        return array

    def _alpha_tensor(
        self,
        output: object,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        matte = self._core.output_prob_to_mask(output)
        alpha = matte.detach().float().clamp(0.0, 1.0)
        if tuple(alpha.shape[-2:]) != output_size:
            alpha = self._torch.nn.functional.interpolate(
                alpha.reshape(1, 1, *alpha.shape[-2:]),
                output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return alpha.mul(255.0).round().to(self._torch.uint8)
