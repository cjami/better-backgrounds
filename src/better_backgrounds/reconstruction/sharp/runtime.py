"""Pinned Apple SHARP inference adapter for the dedicated build worker."""

from __future__ import annotations

import importlib
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy.typing import NDArray
    from torch import Tensor

SharpDevice = Literal["cuda", "mps", "cpu"]
SharpDeviceRequest = Literal["auto", "cuda", "mps", "cpu"]


def _ply_safe_opacities(opacities: Tensor) -> Tensor:
    """Keep probability endpoints finite when SHARP exports them as logits."""
    import torch  # noqa: PLC0415

    epsilon = torch.finfo(opacities.dtype).eps
    return opacities.clamp(min=epsilon, max=1.0 - epsilon)


def ensure_vendored_sharp() -> ModuleType:
    """Load only the pinned application-owned SHARP package."""
    vendor_root = Path(str(files("better_backgrounds").joinpath("_vendor"))).resolve()
    expected = (vendor_root / "sharp").resolve()
    existing = sys.modules.get("sharp")
    if existing is not None:
        loaded = Path(str(getattr(existing, "__file__", ""))).resolve()
        if not loaded.is_relative_to(expected):
            msg = "A non-vendored sharp package is already loaded"
            raise RuntimeError(msg)
        return existing
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    module = importlib.import_module("sharp")
    loaded = Path(str(getattr(module, "__file__", ""))).resolve()
    if not loaded.is_relative_to(expected):
        msg = "The pinned SHARP runtime could not be loaded"
        raise RuntimeError(msg)
    return module


def resolve_sharp_device(requested: SharpDeviceRequest = "auto") -> SharpDevice:
    """Resolve an explicit supported device without inventing a fallback."""
    import torch  # noqa: PLC0415

    cuda_available = torch.cuda.is_available()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)
    if requested == "mps" and not mps_available:
        msg = "MPS was requested but is not available"
        raise RuntimeError(msg)
    return requested


def synchronize_sharp_device(device: SharpDevice) -> None:
    """Synchronize accelerator work for defensible inference timing."""
    import torch  # noqa: PLC0415

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run_sharp_inference(
    image_path: Path,
    focal_length_px: float,
    checkpoint_path: Path,
    device: SharpDevice,
    output_path: Path,
    model_loaded: Callable[[], None],
) -> float:
    """Run the pinned official predictor and write one compatible binary PLY."""
    ensure_vendored_sharp()
    import torch  # noqa: PLC0415
    from torch.nn import functional  # noqa: PLC0415

    models = importlib.import_module("sharp.models")
    gaussian_utils = importlib.import_module("sharp.utils.gaussians")

    with Image.open(image_path) as opened:
        image = cast("NDArray[np.uint8]", np.asarray(opened.convert("RGB"), dtype=np.uint8))
    height, width = image.shape[:2]
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor = models.create_predictor(models.PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    model_loaded()

    internal_shape = (1536, 1536)
    image_tensor = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([focal_length_px / width], device=device)
    resized = functional.interpolate(
        image_tensor[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    synchronize_sharp_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        gaussians_ndc = predictor(resized, disparity_factor)
        intrinsics = torch.tensor(
            [
                [focal_length_px, 0, width / 2, 0],
                [0, focal_length_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        intrinsics[0] *= internal_shape[0] / width
        intrinsics[1] *= internal_shape[1] / height
        gaussians = gaussian_utils.unproject_gaussians(
            gaussians_ndc,
            torch.eye(4, device=device),
            intrinsics,
            internal_shape,
        )
        gaussians = gaussian_utils.Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=_ply_safe_opacities(gaussians.opacities),
        )
    synchronize_sharp_device(device)
    inference_ms = (time.perf_counter() - started) * 1000.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gaussian_utils.save_ply(gaussians, focal_length_px, (height, width), output_path)
    return inference_ms
