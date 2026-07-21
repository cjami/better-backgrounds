"""Spawn-safe MatAnyone 2 worker process using the shared frame ring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from better_backgrounds.matting.contracts import (
    INTERNAL_SIZES,
    FramePacket,
    InternalSize,
    LivePipelineConfig,
    MatteResult,
    MattingCapabilities,
    MattingConfig,
    ProcessedResult,
    StageTimings,
    calibration_p95_latency,
)
from better_backgrounds.matting.refinement import TemporalAlphaStabilizer
from better_backgrounds.matting.ring import FrameRingDescriptor, SharedFrameRing

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from pathlib import Path

    import torch

    from better_backgrounds.harmonization import HarmonizationSettings
    from better_backgrounds.matting.calibration import CalibrationProfileStore
    from better_backgrounds.matting.runtime import MatAnyoneRuntime

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
ALPHA_MIDPOINT = 128


@dataclass(frozen=True, slots=True)
class WorkerPrepared:
    """Report that checkpoint verification and model loading completed."""

    capabilities: MattingCapabilities
    preparation_ms: float


@dataclass(frozen=True, slots=True)
class WorkerProgress:
    """Report one truthful model preparation or initialization stage."""

    stage: str
    message: str
    completed: int = 0
    total: int = 0
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerReady:
    """Report successful model loading and seed warm-up."""

    capabilities: MattingCapabilities
    initialization_ms: float
    selected_internal_size: int
    fused: bool = False
    real_time_error: str | None = None
    generation: int = 0


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Report one bounded worker error without a traceback or source pixels."""

    message: str
    frame_id: int | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class StopWorker:
    """Request orderly model and shared-memory teardown."""


@dataclass(frozen=True, slots=True)
class InitializeTracking:
    """Attach a new camera ring and seed retained MatAnyone weights."""

    descriptor: FrameRingDescriptor
    config: MattingConfig
    seed_packet: FramePacket
    pipeline_config: LivePipelineConfig | None
    generation: int


@dataclass(frozen=True, slots=True)
class ResetTracking:
    """Release camera memory while retaining the loaded network."""

    generation: int


@dataclass(frozen=True, slots=True)
class WorkerReset:
    """Acknowledge that an older camera ring is no longer attached."""

    generation: int


@dataclass(frozen=True, slots=True)
class ProcessFrame:
    """Bind a frame packet to the tracking generation that submitted it."""

    packet: FramePacket
    generation: int


@dataclass(frozen=True, slots=True)
class WorkerMatte:
    """Bind a portable matte result to its tracking generation."""

    result: MatteResult
    generation: int


@dataclass(frozen=True, slots=True)
class WorkerProcessed:
    """Bind a fused output result to its tracking generation."""

    result: ProcessedResult
    generation: int


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
    ProcessFrame
    | InitializeTracking
    | ResetTracking
    | StopWorker
    | SetLiveBackground
    | ConfigurePresentation
    | ConfigureHarmonization
    | ConfigureGeometry
)
WorkerEvent = (
    WorkerPrepared
    | WorkerProgress
    | WorkerReady
    | WorkerReset
    | WorkerMatte
    | WorkerProcessed
    | WorkerFailure
)


def matting_worker_main(  # noqa: PLR0912
    checkpoint: Path,
    config: MattingConfig,
    commands: Queue,
    events: Queue,
    calibration_path: Path | None = None,
) -> None:
    """Load MatAnyone once, then accept repeated camera and person seeds."""
    ring = None
    runtime = None
    cuda_engine = None
    tensor_stabilizer = None
    try:
        import torch  # noqa: PLC0415
        from torch.nn import functional  # noqa: PLC0415

        from better_backgrounds.matting.calibration import (  # noqa: PLC0415
            CalibrationProfileStore,
        )
        from better_backgrounds.matting.runtime import (  # noqa: PLC0415
            MatAnyoneRuntime,
            verify_checkpoint_path,
        )

        started = time.perf_counter()
        events.put(WorkerProgress("verifying", "Verifying background removal model"))
        verify_checkpoint_path(checkpoint)
        events.put(WorkerProgress("loading", "Loading background removal model"))
        runtime = MatAnyoneRuntime(checkpoint, config)
        events.put(
            WorkerPrepared(
                capabilities=runtime.capabilities,
                preparation_ms=(time.perf_counter() - started) * 1_000.0,
            ),
        )
        profile_store = CalibrationProfileStore(calibration_path)
        session_sizes: dict[tuple[int, int, float], int] = {}
        current_generation = 0
        selected_size = config.internal_size
        stabilizer = TemporalAlphaStabilizer()
        fused = False
        real_time_error = None
        harmonizer = None
        pipeline_config = None
        fitted_backgrounds = None
        background_revision = 0
        mirrored = True
        pipeline_revision = 0
        while True:
            try:
                command = commands.get(timeout=0.25)
            except Empty:
                continue
            if isinstance(command, StopWorker):
                return
            if isinstance(command, ResetTracking):
                if ring is not None:
                    ring.close()
                    ring = None
                current_generation = command.generation
                runtime.reconfigure(config)
                stabilizer = TemporalAlphaStabilizer()
                tensor_stabilizer = None
                events.put(WorkerReset(command.generation))
                continue
            if isinstance(command, InitializeTracking):
                initialization_started = time.perf_counter()
                if ring is not None:
                    ring.close()
                ring = SharedFrameRing.attach(command.descriptor)
                current_generation = command.generation
                pipeline_config = command.pipeline_config
                seed_frame = ring.read_frame(command.seed_packet.shared_slot)
                seed_mask = ring.read_alpha(command.seed_packet.shared_slot)
                events.put(
                    WorkerProgress(
                        "optimizing",
                        "Optimising background removal for this computer",
                        generation=current_generation,
                    ),
                )
                config, selected_size = _calibrated_config(
                    runtime,
                    command.config,
                    seed_frame,
                    seed_mask,
                    command.descriptor,
                    profile_store,
                    session_sizes,
                    events,
                    current_generation,
                )
                events.put(
                    WorkerProgress(
                        "warming",
                        "Starting background removal",
                        generation=current_generation,
                    ),
                )
                runtime.reconfigure(config)
                seed_alpha = runtime.initialize(seed_frame, seed_mask)
                ring.write_alpha(command.seed_packet.shared_slot, seed_alpha)
                stabilizer = TemporalAlphaStabilizer()
                requested_fused = bool(
                    pipeline_config is not None
                    and pipeline_config.prefer_cuda
                    and runtime.capabilities.device_type == "cuda"
                )
                real_time_error = None
                if requested_fused and cuda_engine is None:
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
                        harmonizer = create_appearance_harmonizer()
                        tensor_stabilizer = TensorAlphaStabilizer()
                    except (OSError, RuntimeError, TypeError, ValueError) as error:
                        real_time_error = _safe_message(error)
                elif requested_fused:
                    from better_backgrounds.matting.accelerated import (  # noqa: PLC0415
                        TensorAlphaStabilizer,
                    )

                    tensor_stabilizer = TensorAlphaStabilizer()
                fused = requested_fused and cuda_engine is not None
                if pipeline_config is not None and not requested_fused:
                    real_time_error = "real-time 1080p support requires Windows CUDA"
                fitted_backgrounds = (
                    None if pipeline_config is None else FittedLiveBackgrounds(pipeline_config)
                )
                background_revision = 0
                mirrored = True if pipeline_config is None else pipeline_config.mirrored
                pipeline_revision = 0 if pipeline_config is None else pipeline_config.revision
                events.put(
                    WorkerReady(
                        capabilities=runtime.capabilities,
                        initialization_ms=(time.perf_counter() - initialization_started) * 1_000.0,
                        selected_internal_size=selected_size,
                        fused=fused,
                        real_time_error=real_time_error,
                        generation=current_generation,
                    ),
                )
                continue
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
            if not isinstance(command, ProcessFrame):
                events.put(WorkerFailure(message="Unknown matting worker command"))
                continue
            if command.generation != current_generation or ring is None:
                continue
            packet = command.packet
            inference_started = time.perf_counter()
            try:
                source_frame = ring.read_frame(packet.shared_slot)
                if (
                    fused
                    and cuda_engine is not None
                    and tensor_stabilizer is not None
                    and pipeline_config is not None
                ):
                    matting_start = torch.cuda.Event(enable_timing=True)
                    matting_done = torch.cuda.Event(enable_timing=True)
                    normalization_done = torch.cuda.Event(enable_timing=True)
                    matting_start.record()
                    source_tensor = cuda_engine.upload_frame(source_frame)
                    alpha_tensor = runtime.step_uploaded(source_tensor)
                    alpha_tensor = tensor_stabilizer.apply(
                        alpha_tensor.unsqueeze(0).unsqueeze(0),
                        captured_at=packet.captured_at,
                    )
                    matting_done.record()
                    fitted_source, fitted_alpha = _fit_output_tensors(
                        source_tensor,
                        alpha_tensor,
                        pipeline_config,
                        mirrored=mirrored,
                    )
                    room, room_reference = _require_fitted_backgrounds(fitted_backgrounds)
                    normalization_done.record()
                    accelerated = cuda_engine.compose_uploaded(
                        fitted_source,
                        fitted_alpha,
                        room,
                        CudaCompositionOptions(
                            captured_at=packet.captured_at,
                            harmonizer=harmonizer,
                            reference_background=room_reference,
                        ),
                    )
                    # compose_uploaded's readback drained the stream, so these events are
                    # complete and measurable here without a mid-frame host synchronize().
                    matting_ms = matting_start.elapsed_time(matting_done)
                    normalization_ms = matting_done.elapsed_time(normalization_done)
                    ring.write_output(packet.shared_slot, accelerated.image)
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
                        np.count_nonzero(mask_preview >= ALPHA_MIDPOINT) / mask_preview.size,
                    )
                    diagnostic_readback_ms = (time.perf_counter() - diagnostic_started) * 1_000.0
                    events.put(
                        WorkerProcessed(
                            ProcessedResult(
                                frame_id=packet.frame_id,
                                captured_at=packet.captured_at,
                                alpha_slot=packet.shared_slot,
                                output_width=pipeline_config.output_width,
                                output_height=pipeline_config.output_height,
                                background_revision=background_revision,
                                occupancy=occupancy,
                                mask_preview=mask_preview,
                                timings=StageTimings(
                                    normalization_ms=normalization_ms,
                                    queue_ms=max(
                                        0.0,
                                        inference_started * 1_000.0 - packet.captured_at,
                                    ),
                                    matting_ms=matting_ms,
                                    post_processing_ms=accelerated.post_processing_ms,
                                    readback_ms=(accelerated.readback_ms + diagnostic_readback_ms),
                                ),
                                pipeline_revision=pipeline_revision,
                                harmonized=accelerated.harmonized,
                                harmonization_ms=accelerated.harmonization_ms,
                                harmonization_degraded=accelerated.harmonization_degraded,
                            ),
                            current_generation,
                        ),
                    )
                    continue
                alpha = runtime.step(source_frame)
                runtime.synchronize()
                alpha = stabilizer.apply(alpha, captured_at=packet.captured_at)
                ring.write_alpha(packet.shared_slot, alpha)
            except Exception as error:  # noqa: BLE001
                events.put(
                    WorkerFailure(
                        message=_safe_message(error),
                        frame_id=packet.frame_id,
                        generation=current_generation,
                    ),
                )
                return
            matting_ms = (time.perf_counter() - inference_started) * 1000.0
            events.put(
                WorkerMatte(
                    MatteResult(
                        frame_id=packet.frame_id,
                        captured_at=packet.captured_at,
                        alpha_slot=packet.shared_slot,
                        inference_ms=matting_ms,
                    ),
                    current_generation,
                ),
            )
    except Exception as error:  # noqa: BLE001
        events.put(WorkerFailure(message=_safe_message(error)))
    finally:
        if runtime is not None:
            runtime.close()
        if ring is not None:
            ring.close()


def _calibrated_config(
    runtime: MatAnyoneRuntime,
    requested: MattingConfig,
    seed_frame: np.ndarray,
    seed_mask: np.ndarray,
    descriptor: FrameRingDescriptor,
    profile_store: CalibrationProfileStore,
    session_sizes: dict[tuple[int, int, float], int],
    events: Queue,
    generation: int,
) -> tuple[MattingConfig, int]:
    """Reuse a session/profile size or measure from highest quality downward."""
    if not requested.calibrate:
        return requested, requested.internal_size
    session_key = (
        descriptor.width,
        descriptor.height,
        requested.latency_budget_ms,
    )
    session_size = session_sizes.get(session_key)
    if session_size is not None:
        selected = cast("InternalSize", session_size)
        return (
            requested.model_copy(update={"internal_size": selected, "calibrate": False}),
            selected,
        )

    from better_backgrounds.matting.calibration import (  # noqa: PLC0415
        CalibrationIdentity,
        CalibrationProfile,
    )
    from better_backgrounds.matting.runtime import (  # noqa: PLC0415
        load_matanyone_asset_manifest,
    )

    manifest = load_matanyone_asset_manifest()
    device_name, torch_version, accelerator_version = runtime.calibration_device_identity()
    identity = CalibrationIdentity(
        checkpoint_sha256=manifest.checkpoint.sha256,
        upstream_revision=manifest.upstream_revision,
        device_type=runtime.capabilities.device_type,
        device_name=device_name,
        torch_version=torch_version,
        accelerator_version=accelerator_version,
        capture_width=descriptor.width,
        capture_height=descriptor.height,
        latency_budget_ms=requested.latency_budget_ms,
    )
    latencies: dict[int, float] = {}
    cached = profile_store.find(identity)
    selected: InternalSize | None = None
    if cached is not None:
        events.put(
            WorkerProgress(
                "validating",
                "Checking saved background removal settings",
                total=1,
                generation=generation,
            ),
        )
        cached_latency = _measure_size(
            runtime,
            requested,
            cached.selected_internal_size,
            seed_frame,
            seed_mask,
            frames=5,
        )
        latencies[cached.selected_internal_size] = cached_latency
        if (
            cached_latency <= requested.latency_budget_ms
            or cached.selected_internal_size == INTERNAL_SIZES[0]
        ):
            selected = cached.selected_internal_size

    if selected is None:
        descending = tuple(reversed(INTERNAL_SIZES))
        for index, size in enumerate(descending, start=1):
            if size in latencies:
                latency = latencies[size]
            else:
                events.put(
                    WorkerProgress(
                        "calibrating",
                        f"Testing background removal quality ({size}p)",
                        completed=index - 1,
                        total=len(descending),
                        generation=generation,
                    ),
                )
                latency = _measure_size(
                    runtime,
                    requested,
                    size,
                    seed_frame,
                    seed_mask,
                    frames=requested.calibration_frames,
                )
                latencies[size] = latency
            if latency <= requested.latency_budget_ms:
                selected = cast("InternalSize", size)
                break
        if selected is None:
            selected = INTERNAL_SIZES[0]

    selected_latency = latencies.get(selected, 0.0)
    profile_store.save(
        CalibrationProfile(
            identity=identity,
            selected_internal_size=selected,
            measured_p95_ms=selected_latency,
        ),
    )
    session_sizes[session_key] = selected
    return (
        requested.model_copy(update={"internal_size": selected, "calibrate": False}),
        selected,
    )


def _measure_size(
    runtime: MatAnyoneRuntime,
    requested: MattingConfig,
    size: int,
    seed_frame: np.ndarray,
    seed_mask: np.ndarray,
    *,
    frames: int,
) -> float:
    """Measure one resolution after recreating its temporal inference core."""
    trial = requested.model_copy(update={"internal_size": size, "calibrate": False})
    runtime.reconfigure(trial)
    runtime.initialize(seed_frame, seed_mask)
    timings = []
    for _index in range(frames):
        runtime.synchronize()
        frame_started = time.perf_counter()
        runtime.step(seed_frame)
        runtime.synchronize()
        timings.append((time.perf_counter() - frame_started) * 1_000.0)
    return calibration_p95_latency(timings)


def _fit_output_tensors(
    source: torch.Tensor,
    alpha: torch.Tensor,
    config: LivePipelineConfig,
    *,
    mirrored: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch  # noqa: PLC0415
    from torch.nn import functional  # noqa: PLC0415

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
