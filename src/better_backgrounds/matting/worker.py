"""Spawn-safe MatAnyone 2 worker process using the shared frame ring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch.nn import functional

from better_backgrounds.matting.contracts import (
    INTERNAL_SIZES,
    FramePacket,
    LivePipelineConfig,
    MatteResult,
    MattingCapabilities,
    MattingConfig,
    ProcessedResult,
    StageTimings,
    calibration_p95_latency,
    choose_internal_size,
)
from better_backgrounds.matting.refinement import TemporalAlphaStabilizer
from better_backgrounds.matting.ring import FrameRingDescriptor, SharedFrameRing

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from pathlib import Path

    from better_backgrounds.harmonization import HarmonizationSettings

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
ALPHA_MIDPOINT = 128


@dataclass(frozen=True, slots=True)
class WorkerReady:
    """Report successful model loading and seed warm-up."""

    capabilities: MattingCapabilities
    initialization_ms: float
    selected_internal_size: int
    fused: bool = False
    real_time_error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Report one bounded worker error without a traceback or source pixels."""

    message: str
    frame_id: int | None = None


@dataclass(frozen=True, slots=True)
class StopWorker:
    """Request orderly model and shared-memory teardown."""


@dataclass(frozen=True, slots=True)
class SetLiveBackground:
    """Replace immutable output-sized room tensors between frame commands."""

    background: np.ndarray | None
    reference: np.ndarray | None
    revision: int


@dataclass(frozen=True, slots=True)
class ConfigurePresentation:
    """Update cheap session presentation choices without reseeding."""

    mirrored: bool
    retain_standard: bool
    revision: int


@dataclass(frozen=True, slots=True)
class ConfigureHarmonization:
    """Apply room-scoped appearance settings inside the persistent worker."""

    settings: HarmonizationSettings


@dataclass(frozen=True, slots=True)
class ConfigureGeometry:
    """Change output aspect and dimensions without resetting temporal memory."""

    width: int
    height: int
    aspect_ratio: float
    revision: int


@dataclass(slots=True)
class FittedLiveBackgrounds:
    """Fit immutable room evidence once for the active output geometry."""

    config: LivePipelineConfig
    _background_source: np.ndarray | None = field(default=None, repr=False)
    _reference_source: np.ndarray | None = field(default=None, repr=False)
    _background: np.ndarray = field(init=False, repr=False)
    _reference: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Fit the initial empty room at the configured output size."""
        self._refit()

    @property
    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stable arrays reused for every live frame."""
        return self._background, self._reference

    def replace(
        self,
        background: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> None:
        """Replace room sources and fit them to the current output once."""
        self._background_source = background
        self._reference_source = reference
        self._refit()

    def configure(self, config: LivePipelineConfig) -> None:
        """Refit retained sources after an output geometry change."""
        self.config = config
        self._refit()

    def _refit(self) -> None:
        self._background = _fit_background(self._background_source, self.config)
        self._reference = (
            self._background
            if self._reference_source is None
            else _fit_background(self._reference_source, self.config)
        )


WorkerCommand = (
    FramePacket
    | StopWorker
    | SetLiveBackground
    | ConfigurePresentation
    | ConfigureHarmonization
    | ConfigureGeometry
)
WorkerEvent = WorkerReady | MatteResult | ProcessedResult | WorkerFailure


def matting_worker_main(  # noqa: PLR0912
    descriptor: FrameRingDescriptor,
    checkpoint: Path,
    config: MattingConfig,
    seed_packet: FramePacket,
    commands: Queue,
    events: Queue,
    pipeline_config: LivePipelineConfig | None = None,
) -> None:
    """Load MatAnyone 2 once, then process sequential exact-frame commands."""
    ring = SharedFrameRing.attach(descriptor)
    runtime = None
    cuda_engine = None
    tensor_stabilizer = None
    try:
        from better_backgrounds.matting.runtime import MatAnyoneRuntime  # noqa: PLC0415

        started = time.perf_counter()
        runtime = MatAnyoneRuntime(checkpoint, config)
        seed_frame = ring.read_frame(seed_packet.shared_slot)
        seed_mask = ring.read_alpha(seed_packet.shared_slot)
        selected_size = config.internal_size
        if config.calibrate:
            latencies: dict[int, float] = {}
            for size in INTERNAL_SIZES:
                trial = config.model_copy(
                    update={
                        "internal_size": size,
                        "calibrate": False,
                    },
                )
                runtime.reconfigure(trial)
                runtime.initialize(seed_frame, seed_mask)
                timings = []
                for _index in range(config.calibration_frames):
                    runtime.synchronize()
                    frame_started = time.perf_counter()
                    runtime.step(seed_frame)
                    runtime.synchronize()
                    timings.append((time.perf_counter() - frame_started) * 1000.0)
                latencies[size] = calibration_p95_latency(timings)
            if runtime.capabilities.accelerated and not any(
                latency <= config.latency_budget_ms for latency in latencies.values()
            ):
                _fail_latency_gate(latencies, budget_ms=config.latency_budget_ms)
            selected_size = choose_internal_size(
                latencies,
                budget_ms=config.latency_budget_ms,
            )
            config = config.model_copy(
                update={"internal_size": selected_size, "calibrate": False},
            )
            runtime.reconfigure(config)
        seed_alpha = runtime.initialize(seed_frame, seed_mask)
        ring.write_alpha(seed_packet.shared_slot, seed_alpha)
        stabilizer = TemporalAlphaStabilizer()
        fused = bool(
            pipeline_config is not None
            and pipeline_config.prefer_cuda
            and runtime.capabilities.device_type == "cuda"
        )
        real_time_error = None
        harmonizer = None
        if fused:
            try:
                from better_backgrounds.harmonization.pih import (  # noqa: PLC0415
                    create_appearance_harmonizer,
                )
                from better_backgrounds.matting.accelerated import (  # noqa: PLC0415
                    CudaCompositionOptions,
                    CudaLiveEngine,
                    TensorAlphaStabilizer,
                )

                cuda_engine = CudaLiveEngine()
                tensor_stabilizer = TensorAlphaStabilizer()
                harmonizer = create_appearance_harmonizer()
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                fused = False
                real_time_error = _safe_message(error)
        elif pipeline_config is not None:
            real_time_error = "real-time 1080p support requires Windows CUDA"
        fitted_backgrounds = (
            None if pipeline_config is None else FittedLiveBackgrounds(pipeline_config)
        )
        background_revision = 0
        mirrored = True if pipeline_config is None else pipeline_config.mirrored
        retain_standard = False if pipeline_config is None else pipeline_config.retain_standard
        pipeline_revision = 0 if pipeline_config is None else pipeline_config.revision
        events.put(
            WorkerReady(
                capabilities=runtime.capabilities,
                initialization_ms=(time.perf_counter() - started) * 1000.0,
                selected_internal_size=selected_size,
                fused=fused,
                real_time_error=real_time_error,
            ),
        )
        while True:
            try:
                command = commands.get(timeout=0.25)
            except Empty:
                continue
            if isinstance(command, StopWorker):
                return
            if isinstance(command, SetLiveBackground):
                if fitted_backgrounds is not None:
                    fitted_backgrounds.replace(command.background, command.reference)
                background_revision = command.revision
                if cuda_engine is not None:
                    cuda_engine.clear_backgrounds()
                if harmonizer is not None:
                    if command.reference is None:
                        harmonizer.clear_room()
                    else:
                        harmonizer.set_room(command.reference, revision=command.revision)
                continue
            if isinstance(command, ConfigurePresentation):
                mirrored = command.mirrored
                retain_standard = command.retain_standard
                pipeline_revision = command.revision
                continue
            if isinstance(command, ConfigureHarmonization):
                if harmonizer is not None:
                    harmonizer.configure(command.settings)
                    if command.settings.active:
                        harmonizer.prepare()
                    if cuda_engine is not None:
                        fused = cuda_engine.supports(harmonizer)
                continue
            if isinstance(command, ConfigureGeometry):
                if pipeline_config is not None:
                    pipeline_config = pipeline_config.model_copy(
                        update={
                            "output_width": command.width,
                            "output_height": command.height,
                            "aspect_ratio": command.aspect_ratio,
                            "revision": command.revision,
                        },
                    )
                    pipeline_revision = command.revision
                    if fitted_backgrounds is not None:
                        fitted_backgrounds.configure(pipeline_config)
                    if cuda_engine is not None:
                        cuda_engine.clear_backgrounds()
                continue
            if not isinstance(command, FramePacket):
                events.put(WorkerFailure(message="Unknown matting worker command"))
                continue
            inference_started = time.perf_counter()
            try:
                source_frame = ring.read_frame(command.shared_slot)
                if (
                    fused
                    and cuda_engine is not None
                    and tensor_stabilizer is not None
                    and pipeline_config is not None
                ):
                    source_tensor = cuda_engine.upload_frame(source_frame)
                    alpha_tensor = runtime.step_uploaded(source_tensor)
                    alpha_tensor = tensor_stabilizer.apply(
                        alpha_tensor.unsqueeze(0).unsqueeze(0),
                        captured_at=command.captured_at,
                    )
                    runtime.synchronize()
                    matting_ms = (time.perf_counter() - inference_started) * 1_000.0
                    normalization_started = time.perf_counter()
                    fitted_source, fitted_alpha = _fit_output_tensors(
                        source_tensor,
                        alpha_tensor,
                        pipeline_config,
                        mirrored=mirrored,
                    )
                    room, room_reference = _require_fitted_backgrounds(fitted_backgrounds)
                    normalization_ms = (time.perf_counter() - normalization_started) * 1_000.0
                    accelerated = cuda_engine.compose_uploaded(
                        fitted_source,
                        fitted_alpha,
                        room,
                        CudaCompositionOptions(
                            captured_at=command.captured_at,
                            harmonizer=harmonizer,
                            reference_background=room_reference,
                            retain_standard=retain_standard,
                        ),
                    )
                    ring.write_output(command.shared_slot, accelerated.image)
                    ring.write_output(
                        command.shared_slot,
                        accelerated.standard_image,
                        standard=True,
                    )
                    diagnostic_started = time.perf_counter()
                    preview_width = min(160, pipeline_config.output_width)
                    preview_height = max(
                        1,
                        round(preview_width / pipeline_config.aspect_ratio),
                    )
                    mask_preview = (
                        functional.interpolate(
                            fitted_alpha.to(torch.float32),
                            (preview_height, preview_width),
                            mode="area",
                        )[0, 0]
                        .round()
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    occupancy = float(
                        (fitted_alpha >= ALPHA_MIDPOINT).to(torch.float32).mean().item(),
                    )
                    diagnostic_readback_ms = (time.perf_counter() - diagnostic_started) * 1_000.0
                    events.put(
                        ProcessedResult(
                            frame_id=command.frame_id,
                            captured_at=command.captured_at,
                            alpha_slot=command.shared_slot,
                            output_width=pipeline_config.output_width,
                            output_height=pipeline_config.output_height,
                            background_revision=background_revision,
                            occupancy=occupancy,
                            mask_preview=mask_preview,
                            timings=StageTimings(
                                normalization_ms=normalization_ms,
                                queue_ms=max(
                                    0.0,
                                    inference_started * 1_000.0 - command.captured_at,
                                ),
                                matting_ms=matting_ms,
                                post_processing_ms=accelerated.post_processing_ms,
                                readback_ms=(accelerated.readback_ms + diagnostic_readback_ms),
                            ),
                            standard_retained=retain_standard,
                            pipeline_revision=pipeline_revision,
                            harmonized=accelerated.harmonized,
                            harmonization_ms=accelerated.harmonization_ms,
                            harmonization_degraded=accelerated.harmonization_degraded,
                        ),
                    )
                    continue
                alpha = runtime.step(source_frame)
                runtime.synchronize()
                alpha = stabilizer.apply(alpha, captured_at=command.captured_at)
                ring.write_alpha(command.shared_slot, alpha)
            except Exception as error:  # noqa: BLE001
                events.put(
                    WorkerFailure(
                        message=_safe_message(error),
                        frame_id=command.frame_id,
                    ),
                )
                return
            matting_ms = (time.perf_counter() - inference_started) * 1000.0
            events.put(
                MatteResult(
                    frame_id=command.frame_id,
                    captured_at=command.captured_at,
                    alpha_slot=command.shared_slot,
                    inference_ms=matting_ms,
                ),
            )
    except Exception as error:  # noqa: BLE001
        events.put(WorkerFailure(message=_safe_message(error)))
    finally:
        if runtime is not None:
            runtime.close()
        ring.close()


def _fit_output_tensors(
    source: torch.Tensor,
    alpha: torch.Tensor,
    config: LivePipelineConfig,
    *,
    mirrored: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mirrored:
        source = torch.flip(source, dims=(3,))
        alpha = torch.flip(alpha, dims=(3,))
    target_width = config.output_width
    target_height = config.output_height
    if source.shape[2] != target_height:
        resized_width = round(source.shape[3] * target_height / source.shape[2])
        source = (
            functional.interpolate(
                source.to(torch.float32),
                (target_height, resized_width),
                mode="area",
            )
            .round()
            .to(torch.uint8)
        )
        alpha = (
            functional.interpolate(
                alpha.to(torch.float32),
                (target_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
            .round()
            .to(torch.uint8)
        )
    if source.shape[3] >= target_width:
        left = (source.shape[3] - target_width) // 2
        return (
            source[:, :, :, left : left + target_width],
            alpha[:, :, :, left : left + target_width],
        )
    left = (target_width - source.shape[3]) // 2
    fitted_source = torch.zeros(
        (1, 3, target_height, target_width),
        dtype=torch.uint8,
        device=source.device,
    )
    fitted_alpha = torch.zeros(
        (1, 1, target_height, target_width),
        dtype=torch.uint8,
        device=alpha.device,
    )
    fitted_source[:, :, :, left : left + source.shape[3]] = source
    fitted_alpha[:, :, :, left : left + source.shape[3]] = alpha
    return fitted_source, fitted_alpha


def _require_fitted_backgrounds(
    backgrounds: FittedLiveBackgrounds | None,
) -> tuple[np.ndarray, np.ndarray]:
    if backgrounds is None:
        msg = "fused live output requires fitted room evidence"
        raise RuntimeError(msg)
    return backgrounds.frames


def _fit_background(
    background: np.ndarray | None,
    config: LivePipelineConfig,
) -> np.ndarray:
    shape = (config.output_height, config.output_width, 3)
    if background is None:
        return np.zeros(shape, dtype=np.uint8)
    if (
        background.dtype != np.uint8
        or background.ndim != RGB_DIMENSIONS
        or background.shape[2] != RGB_CHANNELS
    ):
        msg = "worker background must be uint8 RGB"
        raise ValueError(msg)
    if background.shape != shape:
        return cv2.resize(
            background,
            (config.output_width, config.output_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return background


def _safe_message(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:300]


def _fail_latency_gate(latencies: dict[int, float], *, budget_ms: float) -> None:
    measurements = ", ".join(f"{size}p={latencies[size]:.1f} ms" for size in INTERNAL_SIZES)
    msg = f"MatAnyone 2 missed the {budget_ms:.1f} ms inference gate: {measurements}"
    raise RuntimeError(msg)
