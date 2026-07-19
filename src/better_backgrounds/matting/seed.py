"""One-shot MediaPipe person seed generation for MatAnyone 2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from better_backgrounds.matting import packaged_seed_model_path

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenter
    from numpy.typing import NDArray

RGB_DIMENSIONS = 3
RGB_CHANNELS = 3
MASK_DIMENSIONS = 2
MINIMUM_SEED_OCCUPANCY = 0.01
MAXIMUM_SEED_OCCUPANCY = 0.85


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    """Describe one independently selectable person component."""

    candidate_id: int
    mask: NDArray[np.uint8]
    bounds: tuple[int, int, int, int]
    occupancy: float


class StableFrameSelector:
    """Return a copy after consecutive frames remain below a motion threshold."""

    def __init__(
        self,
        *,
        required_stable_frames: int = 3,
        motion_threshold: float = 12.0,
    ) -> None:
        """Configure a small bounded pre-seed stability window."""
        if required_stable_frames < 1 or motion_threshold <= 0:
            msg = "stable frame settings must be positive"
            raise ValueError(msg)
        self.required_stable_frames = required_stable_frames
        self.motion_threshold = motion_threshold
        self._previous: NDArray[np.uint8] | None = None
        self._stable_count = 0

    def offer(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        """Observe one RGB frame and return it only after stable transitions."""
        if (
            frame.dtype != np.uint8
            or frame.ndim != RGB_DIMENSIONS
            or frame.shape[2] != RGB_CHANNELS
        ):
            msg = "seed candidate must be uint8 RGB"
            raise ValueError(msg)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        reduced = cast(
            "NDArray[np.uint8]",
            cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA),
        )
        if self._previous is None or self._previous.shape != reduced.shape:
            self._stable_count = 0
        else:
            motion = float(cv2.absdiff(reduced, self._previous).mean())
            self._stable_count = self._stable_count + 1 if motion <= self.motion_threshold else 0
        self._previous = reduced
        if self._stable_count >= self.required_stable_frames:
            return frame.copy()
        return None

    def reset(self) -> None:
        """Discard observations after retry, camera change, or reseed."""
        self._previous = None
        self._stable_count = 0


class MediaPipeSeedProvider:
    """Load the bundled MediaPipe model for one static person segmentation."""

    def __init__(self) -> None:
        """Create an image-mode segmenter from the verified bundled model."""
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions  # noqa: PLC0415
        from mediapipe.tasks.python.vision import (  # noqa: PLC0415
            ImageSegmenter,
            ImageSegmenterOptions,
            RunningMode,
        )

        model = packaged_seed_model_path()
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._mp = mp
        self._segmenter: ImageSegmenter | None = ImageSegmenter.create_from_options(options)

    def generate(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return one binary largest-person mask suitable for target initialization."""
        candidates = self.generate_candidates(frame)
        if not candidates:
            msg = "No person was found; stay in frame and retry"
            raise ValueError(msg)
        return candidates[0].mask

    def generate_candidates(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[PersonCandidate, ...]:
        """Return plausible person components ordered from largest to smallest."""
        segmenter = self._segmenter
        if segmenter is None:
            msg = "MediaPipe seed provider is closed"
            raise RuntimeError(msg)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame),
        )
        result = segmenter.segment(image)
        masks = result.confidence_masks or []
        if not masks:
            msg = "MediaPipe returned no person confidence mask"
            raise RuntimeError(msg)
        labels = [label.lower() for label in segmenter.labels]
        person_index = labels.index("person") if "person" in labels else len(masks) - 1
        confidence = np.squeeze(np.array(masks[person_index].numpy_view(), copy=True))
        return person_candidates(confidence, threshold=0.5)

    def close(self) -> None:
        """Unload the bootstrap model before MatAnyone 2 starts."""
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None


def largest_person_component(
    confidence: NDArray[np.floating],
    *,
    threshold: float,
) -> NDArray[np.uint8]:
    """Keep the largest connected foreground region from a confidence mask."""
    if confidence.ndim != MASK_DIMENSIONS:
        msg = "person confidence mask must be two-dimensional"
        raise ValueError(msg)
    candidates = person_candidates(confidence, threshold=threshold)
    if not candidates:
        msg = "No person was found; stay in frame and retry"
        raise ValueError(msg)
    return candidates[0].mask


def person_candidates(
    confidence: NDArray[np.floating],
    *,
    threshold: float,
) -> tuple[PersonCandidate, ...]:
    """Return every plausible connected person region in descending area order."""
    if confidence.ndim != MASK_DIMENSIONS:
        msg = "person confidence mask must be two-dimensional"
        raise ValueError(msg)
    binary = (confidence >= threshold).astype(np.uint8)
    component_count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    components = sorted(
        range(1, component_count),
        key=lambda label: int(statistics[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    candidates = []
    for label in components:
        area = int(statistics[label, cv2.CC_STAT_AREA])
        occupancy = area / confidence.size
        if not MINIMUM_SEED_OCCUPANCY <= occupancy <= MAXIMUM_SEED_OCCUPANCY:
            continue
        bounds = (
            int(statistics[label, cv2.CC_STAT_LEFT]),
            int(statistics[label, cv2.CC_STAT_TOP]),
            int(statistics[label, cv2.CC_STAT_WIDTH]),
            int(statistics[label, cv2.CC_STAT_HEIGHT]),
        )
        candidates.append(
            PersonCandidate(
                candidate_id=len(candidates) + 1,
                mask=np.where(labels == label, 255, 0).astype(np.uint8),
                bounds=bounds,
                occupancy=occupancy,
            ),
        )
    return tuple(candidates)
