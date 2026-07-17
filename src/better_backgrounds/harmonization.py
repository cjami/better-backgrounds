"""Conservative cached appearance harmonization for exact-frame compositing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from types import ModuleType

    import torch
    from numpy.typing import NDArray

RGB_CHANNELS = 3
RGB_DIMENSIONS = 3
ANALYSIS_LONG_EDGE = 160
HIGH_CONFIDENCE_ALPHA = 230
MINIMUM_STATISTICS_PIXELS = 32
MINIMUM_LINEAR_VALUE = 1e-4
DEFAULT_STATISTICS_INTERVAL = 0.2
DEFAULT_SMOOTHING_TIME = 0.5
DEFAULT_FRAME_BUDGET_MS = 2.0
DEFAULT_OVERRUN_LIMIT = 3
MAX_EXPOSURE_STOPS = 0.5
MIN_CONTRAST = 0.85
MAX_CONTRAST = 1.15
MIN_CHANNEL_GAIN = 0.92
MAX_CHANNEL_GAIN = 1.08
MIN_SATURATION = 0.9
MAX_SATURATION = 1.1
MIN_DIRECTIONAL_LUMINANCE = 0.9
MAX_DIRECTIONAL_LUMINANCE = 1.1
MIN_DIRECTIONAL_CHANNEL = 0.97
MAX_DIRECTIONAL_CHANNEL = 1.03
EDGE_DECONTAMINATION_STRENGTH = 0.12
LIGHT_WRAP_STRENGTH = 0.08
MAX_DETAIL_ADJUSTMENT = 0.12
MAX_BLUR_SIGMA = 0.65
MAX_GRAIN = 1.5 / 255.0
SRGB_DECODE_THRESHOLD = 0.04045
SRGB_ENCODE_THRESHOLD = 0.0031308
DETAIL_EFFECT_THRESHOLD = 0.01
REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DEGRADATION_STEPS = (
    "statistics_frequency",
    "grain",
    "light_wrap",
    "directional_shading",
    "global_only",
    "standard_composite",
)
STATISTICS_DEGRADATION_STAGE = 1
GRAIN_DEGRADATION_STAGE = 2
LIGHT_WRAP_DEGRADATION_STAGE = 3
DIRECTIONAL_DEGRADATION_STAGE = 4
GLOBAL_ONLY_DEGRADATION_STAGE = 5
STANDARD_COMPOSITE_DEGRADATION_STAGE = 6


class HarmonizationSettings(BaseModel):
    """Independently switch every evidence-dependent appearance component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_appearance: bool = False
    directional_shading: bool = False
    edge_decontamination: bool = False
    light_wrap: bool = False
    detail_match: bool = False
    depth_effects: bool = False

    @property
    def active(self) -> bool:
        """Return whether any component was explicitly requested."""
        return any(
            (
                self.global_appearance,
                self.directional_shading,
                self.edge_decontamination,
                self.light_wrap,
                self.detail_match,
                self.depth_effects,
            ),
        )


@dataclass(frozen=True, slots=True)
class AppearanceStatistics:
    """Small robust colour and detail summary in linear RGB."""

    median_rgb: NDArray[np.float32]
    median_luminance: float
    contrast_range: float
    saturation: float
    detail: float


@dataclass(frozen=True, slots=True)
class RoomAppearance:
    """Room-scoped evidence cached outside the webcam frame loop."""

    revision: int
    image: NDArray[np.uint8]
    blurred_image: NDArray[np.uint8]
    statistics: AppearanceStatistics
    ambient_rgb: NDArray[np.float32]
    left_rgb: NDArray[np.float32]
    right_rgb: NDArray[np.float32]
    top_rgb: NDArray[np.float32]
    bottom_rgb: NDArray[np.float32]


def _identity_gains() -> NDArray[np.float32]:
    return np.ones(RGB_CHANNELS, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class AppearanceParameters:
    """Bounded values smoothed between low-frequency statistics updates."""

    exposure_stops: float = 0.0
    contrast: float = 1.0
    channel_gains: NDArray[np.float32] = field(default_factory=_identity_gains)
    saturation: float = 1.0
    detail_adjustment: float = 0.0
    blur_sigma: float = 0.0
    grain: float = 0.0


@dataclass(frozen=True, slots=True)
class HarmonizationResult:
    """Return processed pixels and measurable per-frame evidence."""

    foreground: NDArray[np.uint8] | None
    image: NDArray[np.uint8] | None
    parameters: AppearanceParameters
    statistics_updated: bool
    processing_ms: float
    statistics_ms: float
    degraded_components: tuple[str, ...]
    applied: bool


class HarmonizationBudget:
    """Degrade expensive components in the phase-defined order after overruns."""

    def __init__(
        self,
        budget_ms: float = DEFAULT_FRAME_BUDGET_MS,
        *,
        overrun_limit: int = DEFAULT_OVERRUN_LIMIT,
        enforced: bool = True,
    ) -> None:
        """Create a deterministic one-way fallback policy for one live session."""
        if budget_ms <= 0 or not math.isfinite(budget_ms):
            msg = "frame budget must be finite and positive"
            raise ValueError(msg)
        if overrun_limit < 1:
            msg = "overrun limit must be positive"
            raise ValueError(msg)
        self.budget_ms = budget_ms
        self.overrun_limit = overrun_limit
        self.enforced = enforced
        self.stage = 0
        self._consecutive_overruns = 0

    @property
    def degraded_components(self) -> tuple[str, ...]:
        """Describe every fallback applied at the current stage."""
        return DEGRADATION_STEPS[: self.stage]

    def observe(self, processing_ms: float) -> None:
        """Advance one stage only after sustained budget overruns."""
        if not math.isfinite(processing_ms) or processing_ms < 0:
            msg = "processing time must be finite and non-negative"
            raise ValueError(msg)
        if not self.enforced:
            return
        if processing_ms <= self.budget_ms:
            self._consecutive_overruns = 0
            return
        self._consecutive_overruns += 1
        if self._consecutive_overruns >= self.overrun_limit:
            self.stage = min(len(DEGRADATION_STEPS), self.stage + 1)
            self._consecutive_overruns = 0

    def reset(self) -> None:
        """Re-evaluate headroom for a changed room or camera source."""
        self.stage = 0
        self._consecutive_overruns = 0


def _validate_rgb(image: NDArray[np.uint8], *, name: str) -> None:
    if image.dtype != np.uint8 or image.ndim != RGB_DIMENSIONS or image.shape[2] != RGB_CHANNELS:
        msg = f"{name} must be uint8 RGB"
        raise ValueError(msg)
    if image.size == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)


def _linearize(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    encoded = image.astype(np.float32) / 255.0
    return cast(
        "NDArray[np.float32]",
        np.where(
            encoded <= SRGB_DECODE_THRESHOLD,
            encoded / 12.92,
            ((encoded + 0.055) / 1.055) ** 2.4,
        ),
    )


def _encode(linear: NDArray[np.float32]) -> NDArray[np.uint8]:
    bounded = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        bounded <= SRGB_ENCODE_THRESHOLD,
        bounded * 12.92,
        1.055 * np.power(bounded, 1.0 / 2.4) - 0.055,
    )
    return cast("NDArray[np.uint8]", np.rint(encoded * 255.0).astype(np.uint8))


def _analysis_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    scale = min(1.0, ANALYSIS_LONG_EDGE / max(height, width))
    if scale == 1.0:
        return image
    return cast(
        "NDArray[np.uint8]",
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
    )


def _statistics(pixels: NDArray[np.float32]) -> AppearanceStatistics:
    flattened = pixels.reshape(-1, RGB_CHANNELS)
    luminance = flattened @ REC709
    median_rgb = cast("NDArray[np.float32]", np.median(flattened, axis=0).astype(np.float32))
    low, median_luminance, high = np.percentile(luminance, (10, 50, 90))
    channel_max = flattened.max(axis=1)
    channel_min = flattened.min(axis=1)
    saturation = np.median(
        (channel_max - channel_min) / np.maximum(channel_max, MINIMUM_LINEAR_VALUE),
    )
    return AppearanceStatistics(
        median_rgb=median_rgb,
        median_luminance=float(median_luminance),
        contrast_range=float(max(high - low, MINIMUM_LINEAR_VALUE)),
        saturation=float(max(saturation, MINIMUM_LINEAR_VALUE)),
        detail=0.0,
    )


def _detail(linear: NDArray[np.float32]) -> float:
    luminance = cast("NDArray[np.float32]", linear @ REC709)
    laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(laplacian)))


def _masked_detail(
    linear: NDArray[np.float32],
    selected: NDArray[np.bool_],
) -> float:
    """Measure texture inside confident pixels without counting the matte boundary."""
    interior = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    if int(np.count_nonzero(interior)) < MINIMUM_STATISTICS_PIXELS:
        interior = selected
    luminance = cast("NDArray[np.float32]", linear @ REC709)
    laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(laplacian[interior])))


def _region_median(
    linear: NDArray[np.float32],
    rows: slice,
    columns: slice,
) -> NDArray[np.float32]:
    region = linear[rows, columns]
    return cast(
        "NDArray[np.float32]",
        np.median(region.reshape(-1, RGB_CHANNELS), axis=0).astype(np.float32),
    )


def preprocess_room(background: NDArray[np.uint8], *, revision: int) -> RoomAppearance:
    """Cache robust room colour, direction, blur, and detail evidence."""
    _validate_rgb(background, name="background")
    if revision < 0:
        msg = "background revision must be non-negative"
        raise ValueError(msg)
    image = background.copy()
    analysis = _analysis_image(image)
    linear = _linearize(analysis)
    height, width = analysis.shape[:2]
    statistics = _statistics(linear)
    statistics = AppearanceStatistics(
        median_rgb=statistics.median_rgb,
        median_luminance=statistics.median_luminance,
        contrast_range=statistics.contrast_range,
        saturation=statistics.saturation,
        detail=_detail(linear),
    )
    horizontal_band = max(1, width // 3)
    vertical_band = max(1, height // 3)
    left = _region_median(linear, slice(None), slice(0, horizontal_band))
    right = _region_median(linear, slice(None), slice(width - horizontal_band, width))
    top = _region_median(linear, slice(0, vertical_band), slice(None))
    bottom = _region_median(linear, slice(height - vertical_band, height), slice(None))
    ambient = cast(
        "NDArray[np.float32]",
        np.mean(np.stack((left, right, top, bottom)), axis=0).astype(np.float32),
    )
    blurred = cast(
        "NDArray[np.uint8]",
        cv2.GaussianBlur(image, (0, 0), sigmaX=max(2.0, min(height, width) / 24.0)),
    )
    for cached in (image, blurred, ambient, left, right, top, bottom, statistics.median_rgb):
        cached.setflags(write=False)
    return RoomAppearance(
        revision=revision,
        image=image,
        blurred_image=blurred,
        statistics=statistics,
        ambient_rgb=ambient,
        left_rgb=left,
        right_rgb=right,
        top_rgb=top,
        bottom_rgb=bottom,
    )


def _interpolate(
    current: AppearanceParameters,
    target: AppearanceParameters,
    amount: float,
) -> AppearanceParameters:
    mix = min(1.0, max(0.0, amount))

    def scalar(start: float, end: float) -> float:
        return start + (end - start) * mix

    return AppearanceParameters(
        exposure_stops=scalar(current.exposure_stops, target.exposure_stops),
        contrast=scalar(current.contrast, target.contrast),
        channel_gains=cast(
            "NDArray[np.float32]",
            (current.channel_gains + (target.channel_gains - current.channel_gains) * mix).astype(
                np.float32,
            ),
        ),
        saturation=scalar(current.saturation, target.saturation),
        detail_adjustment=scalar(current.detail_adjustment, target.detail_adjustment),
        blur_sigma=scalar(current.blur_sigma, target.blur_sigma),
        grain=scalar(current.grain, target.grain),
    )


class AppearanceHarmonizer:
    """Own cached room evidence and smoothed camera-specific estimates."""

    def __init__(
        self,
        settings: HarmonizationSettings | None = None,
        *,
        statistics_interval: float = DEFAULT_STATISTICS_INTERVAL,
        smoothing_time: float = DEFAULT_SMOOTHING_TIME,
        budget: HarmonizationBudget | None = None,
    ) -> None:
        """Start disabled and without scene or camera state."""
        if statistics_interval < 0 or not math.isfinite(statistics_interval):
            msg = "statistics interval must be finite and non-negative"
            raise ValueError(msg)
        if smoothing_time <= 0 or not math.isfinite(smoothing_time):
            msg = "smoothing time must be finite and positive"
            raise ValueError(msg)
        self.settings = settings or HarmonizationSettings()
        self.statistics_interval = statistics_interval
        self.smoothing_time = smoothing_time
        self._budget = budget or HarmonizationBudget()
        self._room: RoomAppearance | None = None
        self._parameters = AppearanceParameters()
        self._target = AppearanceParameters()
        self._last_statistics_at: float | None = None
        self._last_frame_at: float | None = None
        self._grain_cache: dict[tuple[int, int], NDArray[np.float32]] = {}

    @property
    def active(self) -> bool:
        """Return whether requested effects have cached room evidence."""
        return self.settings.active and self._room is not None

    @property
    def parameters(self) -> AppearanceParameters:
        """Expose current bounded values for diagnostics and fixed captures."""
        return self._parameters

    @property
    def room_revision(self) -> int | None:
        """Return the room snapshot revision owning current evidence."""
        return None if self._room is None else self._room.revision

    @property
    def degradation_stage(self) -> int:
        """Expose ordered performance fallback state for diagnostics."""
        return self._budget.stage

    def configure(self, settings: HarmonizationSettings) -> None:
        """Apply explicit component switches without changing cached room evidence."""
        self.settings = settings

    def set_room(self, background: NDArray[np.uint8], *, revision: int) -> None:
        """Preprocess a changed room snapshot and clear live estimates."""
        self._room = preprocess_room(background, revision=revision)
        self._grain_cache.clear()
        self.reset_camera()

    def clear_room(self) -> None:
        """Discard all scene and camera-scoped appearance evidence."""
        self._room = None
        self._grain_cache.clear()
        self.reset_camera()

    def reset_camera(self) -> None:
        """Prevent smoothed parameters leaking across camera sources."""
        self._parameters = AppearanceParameters()
        self._target = AppearanceParameters()
        self._last_statistics_at = None
        self._last_frame_at = None
        self._budget.reset()

    def apply(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        *,
        captured_at: float,
    ) -> HarmonizationResult:
        """Apply cached appearance evidence to one matching foreground and alpha."""
        started = time.perf_counter()
        self._validate_frame(source, alpha, background, captured_at)
        room = self._room
        if room is None or not self.settings.active:
            return HarmonizationResult(
                foreground=source,
                image=None,
                parameters=self._parameters,
                statistics_updated=False,
                processing_ms=(time.perf_counter() - started) * 1_000.0,
                statistics_ms=0.0,
                degraded_components=(),
                applied=False,
            )

        statistics_started = time.perf_counter()
        statistics_updated = self._update_target(source, alpha, captured_at)
        statistics_ms = (
            (time.perf_counter() - statistics_started) * 1_000.0 if statistics_updated else 0.0
        )
        self._smooth(captured_at)
        degraded = self._budget.degraded_components
        if self.settings.depth_effects:
            degraded = (*degraded, "depth_effects")
        stage = self._budget.stage
        supported_active = any(
            (
                self.settings.global_appearance,
                self.settings.directional_shading and stage < DIRECTIONAL_DEGRADATION_STAGE,
                self.settings.edge_decontamination and stage < GLOBAL_ONLY_DEGRADATION_STAGE,
                self.settings.light_wrap and stage < LIGHT_WRAP_DEGRADATION_STAGE,
                self.settings.detail_match and stage < GLOBAL_ONLY_DEGRADATION_STAGE,
            ),
        )
        if not supported_active or stage >= len(DEGRADATION_STEPS):
            processing_ms = (time.perf_counter() - started) * 1_000.0
            self._observe_budget(processing_ms)
            return HarmonizationResult(
                foreground=source,
                image=None,
                parameters=self._parameters,
                statistics_updated=statistics_updated,
                processing_ms=processing_ms,
                statistics_ms=statistics_ms,
                degraded_components=degraded,
                applied=False,
            )

        foreground, image = self._render(source, alpha, background, room, stage)
        processing_ms = (time.perf_counter() - started) * 1_000.0
        self._observe_budget(processing_ms)
        return HarmonizationResult(
            foreground=foreground,
            image=image,
            parameters=self._parameters,
            statistics_updated=statistics_updated,
            processing_ms=processing_ms,
            statistics_ms=statistics_ms,
            degraded_components=degraded,
            applied=True,
        )

    def _render(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        room: RoomAppearance,
        stage: int,
    ) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8]]:
        """Render one frame through the portable reference implementation."""
        source_linear = _linearize(source)
        background_linear = _linearize(background)
        matte = alpha.astype(np.float32)[..., None] / 255.0
        processed = source_linear.copy()
        if self.settings.global_appearance:
            processed = self._apply_global(
                processed,
                room,
                minimal=stage >= GLOBAL_ONLY_DEGRADATION_STAGE,
            )
        if self.settings.directional_shading and stage < DIRECTIONAL_DEGRADATION_STAGE:
            processed *= self._directional_field(room, source.shape[:2])
        uncertain = 4.0 * matte * (1.0 - matte)
        if self.settings.edge_decontamination and stage < GLOBAL_ONLY_DEGRADATION_STAGE:
            processed += (processed - background_linear) * uncertain * EDGE_DECONTAMINATION_STRENGTH
        if self.settings.light_wrap and stage < LIGHT_WRAP_DEGRADATION_STAGE:
            blurred = self._blurred_background(room, source.shape[:2])
            processed += (_linearize(blurred) - processed) * uncertain * LIGHT_WRAP_STRENGTH
        if self.settings.detail_match and stage < GLOBAL_ONLY_DEGRADATION_STAGE:
            processed = self._apply_detail(
                processed,
                matte,
                grain_enabled=stage < GRAIN_DEGRADATION_STAGE,
            )
        processed = np.clip(processed, 0.0, 1.0)
        processed = source_linear + matte * (processed - source_linear)
        foreground = _encode(processed)
        foreground[alpha == 0] = source[alpha == 0]
        composite_linear = processed * matte + background_linear * (1.0 - matte)
        image = _encode(composite_linear)
        image[alpha == 0] = background[alpha == 0]
        return foreground, image

    @staticmethod
    def _validate_frame(
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        captured_at: float,
    ) -> None:
        _validate_rgb(source, name="source")
        _validate_rgb(background, name="background")
        if background.shape != source.shape:
            msg = "background must match source dimensions"
            raise ValueError(msg)
        if alpha.dtype != np.uint8 or alpha.shape != source.shape[:2]:
            msg = "alpha must be uint8 and match source dimensions"
            raise ValueError(msg)
        if not math.isfinite(captured_at) or captured_at < 0:
            msg = "capture timestamp must be finite and non-negative"
            raise ValueError(msg)

    def _update_target(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        captured_at: float,
    ) -> bool:
        last = self._last_statistics_at
        interval = self.statistics_interval * (
            2.5 if self._budget.stage >= STATISTICS_DEGRADATION_STAGE else 1.0
        )
        if last is not None and captured_at - last < interval:
            return False
        analysis_source = _analysis_image(source)
        analysis_alpha = cast(
            "NDArray[np.uint8]",
            cv2.resize(
                alpha,
                (analysis_source.shape[1], analysis_source.shape[0]),
                interpolation=cv2.INTER_AREA,
            ),
        )
        selected = analysis_alpha >= HIGH_CONFIDENCE_ALPHA
        if int(np.count_nonzero(selected)) < MINIMUM_STATISTICS_PIXELS:
            return False
        linear = _linearize(analysis_source)
        foreground = linear[selected]
        live = _statistics(foreground)
        live = AppearanceStatistics(
            median_rgb=live.median_rgb,
            median_luminance=live.median_luminance,
            contrast_range=live.contrast_range,
            saturation=live.saturation,
            detail=_masked_detail(linear, selected),
        )
        room = self._room
        if room is None:
            return False
        target = room.statistics
        exposure = math.log2(
            max(target.median_luminance, MINIMUM_LINEAR_VALUE)
            / max(live.median_luminance, MINIMUM_LINEAR_VALUE),
        )
        contrast = target.contrast_range / max(live.contrast_range, MINIMUM_LINEAR_VALUE)
        live_chroma = live.median_rgb / max(float(live.median_rgb @ REC709), MINIMUM_LINEAR_VALUE)
        target_chroma = target.median_rgb / max(
            float(target.median_rgb @ REC709),
            MINIMUM_LINEAR_VALUE,
        )
        gains = np.clip(
            target_chroma / np.maximum(live_chroma, MINIMUM_LINEAR_VALUE),
            MIN_CHANNEL_GAIN,
            MAX_CHANNEL_GAIN,
        ).astype(np.float32)
        saturation = target.saturation / max(live.saturation, MINIMUM_LINEAR_VALUE)
        detail_ratio = target.detail / max(live.detail, MINIMUM_LINEAR_VALUE)
        detail_adjustment = float(
            np.clip(detail_ratio - 1.0, -MAX_DETAIL_ADJUSTMENT, MAX_DETAIL_ADJUSTMENT)
        )
        blur_sigma = (
            min(MAX_BLUR_SIGMA, (1.0 - detail_ratio) * MAX_BLUR_SIGMA)
            if detail_ratio < 1.0
            else 0.0
        )
        grain = min(MAX_GRAIN, max(0.0, target.detail - live.detail) * 0.05)
        self._target = AppearanceParameters(
            exposure_stops=float(np.clip(exposure, -MAX_EXPOSURE_STOPS, MAX_EXPOSURE_STOPS)),
            contrast=float(np.clip(contrast, MIN_CONTRAST, MAX_CONTRAST)),
            channel_gains=cast("NDArray[np.float32]", gains),
            saturation=float(np.clip(saturation, MIN_SATURATION, MAX_SATURATION)),
            detail_adjustment=detail_adjustment,
            blur_sigma=blur_sigma,
            grain=grain,
        )
        self._last_statistics_at = captured_at
        return True

    def _smooth(self, captured_at: float) -> None:
        elapsed = (
            1.0 / 30.0
            if self._last_frame_at is None
            else max(0.0, captured_at - self._last_frame_at)
        )
        amount = 1.0 - math.exp(-elapsed / self.smoothing_time)
        self._parameters = _interpolate(self._parameters, self._target, amount)
        self._last_frame_at = captured_at

    def _observe_budget(self, processing_ms: float) -> None:
        self._budget.observe(processing_ms)

    def _apply_global(
        self,
        foreground: NDArray[np.float32],
        room: RoomAppearance,
        *,
        minimal: bool,
    ) -> NDArray[np.float32]:
        parameters = self._parameters
        adjusted = foreground * (2.0**parameters.exposure_stops)
        pivot = room.statistics.median_luminance
        contrast = 1.0 if minimal else parameters.contrast
        saturation = 1.0 if minimal else parameters.saturation
        adjusted = (adjusted - pivot) * contrast + pivot
        adjusted *= parameters.channel_gains
        luminance = (adjusted @ REC709)[..., None]
        return cast(
            "NDArray[np.float32]",
            luminance + (adjusted - luminance) * saturation,
        )

    @staticmethod
    def _directional_field(
        room: RoomAppearance,
        shape: tuple[int, int],
    ) -> NDArray[np.float32]:
        height, width = shape
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        horizontal = room.left_rgb * (1.0 - x) + room.right_rgb * x
        vertical = room.top_rgb * (1.0 - y) + room.bottom_rgb * y
        field = (horizontal + vertical) * 0.5
        ratio = field / np.maximum(room.ambient_rgb, MINIMUM_LINEAR_VALUE)
        field_luminance = ratio @ REC709
        bounded_luminance = np.clip(
            field_luminance,
            MIN_DIRECTIONAL_LUMINANCE,
            MAX_DIRECTIONAL_LUMINANCE,
        )
        chroma = ratio / np.maximum(field_luminance[..., None], MINIMUM_LINEAR_VALUE)
        chroma = np.clip(chroma, MIN_DIRECTIONAL_CHANNEL, MAX_DIRECTIONAL_CHANNEL)
        return cast("NDArray[np.float32]", chroma * bounded_luminance[..., None])

    @staticmethod
    def _blurred_background(
        room: RoomAppearance,
        shape: tuple[int, int],
    ) -> NDArray[np.uint8]:
        if room.blurred_image.shape[:2] == shape:
            return room.blurred_image
        return cast(
            "NDArray[np.uint8]",
            cv2.resize(room.blurred_image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR),
        )

    def _apply_detail(
        self,
        foreground: NDArray[np.float32],
        matte: NDArray[np.float32],
        *,
        grain_enabled: bool,
    ) -> NDArray[np.float32]:
        parameters = self._parameters
        result = foreground
        if parameters.blur_sigma > DETAIL_EFFECT_THRESHOLD:
            result = cast(
                "NDArray[np.float32]",
                cv2.GaussianBlur(result, (0, 0), sigmaX=parameters.blur_sigma),
            )
        elif parameters.detail_adjustment > DETAIL_EFFECT_THRESHOLD:
            low = cv2.GaussianBlur(result, (0, 0), sigmaX=0.7)
            result = result + (result - low) * parameters.detail_adjustment
        if grain_enabled and parameters.grain > 0.0:
            shape = foreground.shape[:2]
            grain = self._grain_cache.get(shape)
            if grain is None:
                generator = np.random.default_rng(0)
                grain = cast(
                    "NDArray[np.float32]",
                    generator.standard_normal((*shape, 1), dtype=np.float32),
                )
                self._grain_cache[shape] = grain
            result = result + grain * parameters.grain * matte
        return cast("NDArray[np.float32]", result)


class AcceleratedAppearanceHarmonizer(AppearanceHarmonizer):
    """Run the full-resolution pass on CUDA or Metal with cached room tensors."""

    def __init__(
        self,
        preferred_device: str | None = None,
        *,
        enforce_accelerated_budget: bool = True,
    ) -> None:
        """Select an accelerator lazily when appearance matching is first enabled."""
        super().__init__(budget=HarmonizationBudget(overrun_limit=30))
        self._preferred_device = preferred_device
        self._enforce_accelerated_budget = enforce_accelerated_budget
        self._torch: ModuleType | None = None
        self._device: torch.device | None = None
        self._cache_key: tuple[int, int, int] | None = None
        self._background_linear: torch.Tensor | None = None
        self._blurred_linear: torch.Tensor | None = None
        self._directional: torch.Tensor | None = None
        self._grain: torch.Tensor | None = None
        self._rec709: torch.Tensor | None = None

    @property
    def backend_name(self) -> str:
        """Describe the selected full-resolution execution path."""
        if self._device is None:
            return "portable"
        return str(self._device).upper()

    def set_room(self, background: NDArray[np.uint8], *, revision: int) -> None:
        """Reset accelerator caches beside changed room evidence."""
        super().set_room(background, revision=revision)
        self._clear_accelerator_cache()

    def clear_room(self) -> None:
        """Release cached accelerator tensors with the room."""
        super().clear_room()
        self._clear_accelerator_cache()

    def _render(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        room: RoomAppearance,
        stage: int,
    ) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8]]:
        torch = self._load_torch()
        if torch is None or self._device is None:
            return super()._render(source, alpha, background, room, stage)
        try:
            return self._render_accelerated(source, alpha, background, stage)
        except RuntimeError:
            self._device = None
            self._budget.enforced = True
            self._clear_accelerator_cache()
            return super()._render(source, alpha, background, room, stage)

    def _load_torch(self) -> ModuleType | None:
        if self._torch is not None:
            return self._torch
        import torch  # noqa: PLC0415

        preferred = self._preferred_device
        if preferred is not None:
            device_type = preferred
        elif torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            return None
        self._torch = torch
        self._device = torch.device(device_type)
        self._budget.enforced = self._enforce_accelerated_budget
        self._rec709 = torch.tensor(REC709, device=self._device)
        return torch

    def _render_accelerated(
        self,
        source: NDArray[np.uint8],
        alpha: NDArray[np.uint8],
        background: NDArray[np.uint8],
        stage: int,
    ) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8]]:
        torch = self._torch
        room = self._room
        if torch is None or room is None:
            msg = "accelerated renderer requires an initialized room and backend"
            raise RuntimeError(msg)
        self._cache_room_tensors(torch, background, room)
        background_linear = self._background_linear
        blurred_linear = self._blurred_linear
        directional = self._directional
        rec709 = self._rec709
        if (
            background_linear is None
            or blurred_linear is None
            or directional is None
            or rec709 is None
        ):
            msg = "accelerated room tensors are incomplete"
            raise RuntimeError(msg)
        source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to(
            device=self._device,
            dtype=torch.float32,
        )
        source_tensor.div_(255.0)
        matte = torch.from_numpy(np.ascontiguousarray(alpha)).to(
            device=self._device,
            dtype=torch.float32,
        )
        matte = matte.div_(255.0).unsqueeze_(2)
        source_linear = self._torch_linearize(source_tensor)
        processed = source_linear.clone()
        parameters = self.parameters
        if self.settings.global_appearance:
            processed.mul_(2.0**parameters.exposure_stops)
            minimal = stage >= GLOBAL_ONLY_DEGRADATION_STAGE
            contrast = 1.0 if minimal else parameters.contrast
            saturation = 1.0 if minimal else parameters.saturation
            pivot = room.statistics.median_luminance
            processed.sub_(pivot).mul_(contrast).add_(pivot)
            gains = torch.as_tensor(parameters.channel_gains, device=self._device)
            processed.mul_(gains)
            luminance = torch.sum(processed * rec709, dim=2, keepdim=True)
            processed = luminance + (processed - luminance) * saturation
        if self.settings.directional_shading and stage < DIRECTIONAL_DEGRADATION_STAGE:
            processed.mul_(directional)
        uncertain = matte * (1.0 - matte) * 4.0
        if self.settings.edge_decontamination and stage < GLOBAL_ONLY_DEGRADATION_STAGE:
            processed.add_(
                (processed - background_linear) * uncertain * EDGE_DECONTAMINATION_STRENGTH,
            )
        if self.settings.light_wrap and stage < LIGHT_WRAP_DEGRADATION_STAGE:
            processed.add_(
                (blurred_linear - processed) * uncertain * LIGHT_WRAP_STRENGTH,
            )
        if self.settings.detail_match and stage < GLOBAL_ONLY_DEGRADATION_STAGE:
            processed = self._torch_detail(processed, matte, stage)
        processed.clamp_(0.0, 1.0)
        processed = source_linear + matte * (processed - source_linear)
        composite = processed * matte + background_linear * (1.0 - matte)
        pixels = self._torch_encode(composite.clamp_(0.0, 1.0))
        pixels.mul_(255.0).round_()
        pixels = pixels.to(torch.uint8)
        host = pixels.cpu().numpy()
        return None, cast("NDArray[np.uint8]", host)

    def _cache_room_tensors(
        self,
        torch: ModuleType,
        background: NDArray[np.uint8],
        room: RoomAppearance,
    ) -> None:
        height, width = background.shape[:2]
        key = (room.revision, height, width)
        if self._cache_key == key:
            return
        encoded = torch.from_numpy(np.ascontiguousarray(background)).to(
            device=self._device,
            dtype=torch.float32,
        )
        encoded.div_(255.0)
        blurred = self._blurred_background(room, (height, width))
        blurred_tensor = torch.from_numpy(np.array(blurred, copy=True, order="C")).to(
            device=self._device,
            dtype=torch.float32,
        )
        blurred_tensor.div_(255.0)
        directional = self._directional_field(room, (height, width))
        self._background_linear = self._torch_linearize(encoded)
        self._blurred_linear = self._torch_linearize(blurred_tensor)
        self._directional = torch.from_numpy(np.ascontiguousarray(directional)).to(
            device=self._device,
        )
        generator = np.random.default_rng(0)
        grain = generator.standard_normal((height, width, 1), dtype=np.float32)
        self._grain = torch.from_numpy(grain).to(device=self._device)
        self._cache_key = key

    def _torch_detail(
        self,
        foreground: torch.Tensor,
        matte: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        from torch.nn import functional  # noqa: PLC0415

        parameters = self.parameters
        result = foreground
        if (
            parameters.blur_sigma > DETAIL_EFFECT_THRESHOLD
            or parameters.detail_adjustment > DETAIL_EFFECT_THRESHOLD
        ):
            channels_first = result.permute(2, 0, 1).unsqueeze_(0)
            low = functional.avg_pool2d(channels_first, 3, stride=1, padding=1)
            low = low.squeeze_(0).permute(1, 2, 0)
            if parameters.blur_sigma > DETAIL_EFFECT_THRESHOLD:
                amount = min(1.0, parameters.blur_sigma / MAX_BLUR_SIGMA)
                result = result + (low - result) * amount
            else:
                result = result + (result - low) * parameters.detail_adjustment
        if stage < GRAIN_DEGRADATION_STAGE and parameters.grain > 0.0:
            grain = self._grain
            if grain is None:
                msg = "accelerated grain tensor is unavailable"
                raise RuntimeError(msg)
            result = result + grain * parameters.grain * matte
        return result

    def _torch_linearize(self, encoded: torch.Tensor) -> torch.Tensor:
        torch = self._torch
        if torch is None:
            msg = "accelerated backend is not initialized"
            raise RuntimeError(msg)
        return torch.where(
            encoded <= SRGB_DECODE_THRESHOLD,
            encoded / 12.92,
            ((encoded + 0.055) / 1.055).pow(2.4),
        )

    def _torch_encode(self, linear: torch.Tensor) -> torch.Tensor:
        torch = self._torch
        if torch is None:
            msg = "accelerated backend is not initialized"
            raise RuntimeError(msg)
        return torch.where(
            linear <= SRGB_ENCODE_THRESHOLD,
            linear * 12.92,
            1.055 * linear.pow(1.0 / 2.4) - 0.055,
        )

    def _clear_accelerator_cache(self) -> None:
        self._cache_key = None
        self._background_linear = None
        self._blurred_linear = None
        self._directional = None
        self._grain = None
