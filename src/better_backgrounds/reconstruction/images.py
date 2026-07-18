"""Room-image selection, inspection, and hashing."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, SupportsFloat, SupportsIndex, cast

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from pathlib import Path

SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 100_000_000
MIN_IMAGE_DIMENSION = 256
FILE_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_FOCAL_LENGTH_MM = 30.0
MOBILE_FOCAL_THRESHOLD_MM = 10.0
LARGE_IMAGE_WARNING_DIMENSION = 8192
MAX_EXIF_ORIENTATION = 8


@dataclass(frozen=True, slots=True)
class SceneImageSelection:
    """Identify one trusted local image selected for a room build."""

    display_name: str
    source_path: Path | None
    source_kind: Literal["upload", "camera"] = "upload"


@dataclass(frozen=True, slots=True)
class SceneImageDiagnostics:
    """Describe the oriented image and warnings shown before building."""

    width: int
    height: int
    image_format: Literal["JPEG", "PNG", "WEBP"]
    focal_length_px: float
    orientation_applied: bool
    has_alpha: bool
    has_focal_metadata: bool
    warnings: tuple[str, ...]


def _exif_values(image: Image.Image) -> dict[str, object]:
    values: dict[str, object] = {}
    exif = image.getexif()
    for key, value in exif.items():
        name = ExifTags.TAGS.get(key)
        if name is not None:
            values[name] = value
    try:
        detail = exif.get_ifd(ExifTags.IFD.Exif)
    except KeyError, TypeError, ValueError:
        detail = {}
    for key, value in detail.items():
        name = ExifTags.TAGS.get(key)
        if name is not None:
            values[name] = value
    return values


def _focal_length_35mm(exif: dict[str, object]) -> tuple[float, bool]:
    raw = exif.get("FocalLengthIn35mmFilm", exif.get("FocalLenIn35mmFilm"))
    if raw is None:
        raw = exif.get("FocalLength")
    try:
        focal = (
            float(cast("str | SupportsFloat | SupportsIndex", raw))
            if raw is not None
            else DEFAULT_FOCAL_LENGTH_MM
        )
    except TypeError, ValueError, ZeroDivisionError:
        focal = DEFAULT_FOCAL_LENGTH_MM
        raw = None
    if not math.isfinite(focal) or focal <= 0:
        focal = DEFAULT_FOCAL_LENGTH_MM
        raw = None
    if focal < MOBILE_FOCAL_THRESHOLD_MM:
        focal *= 8.4
    return focal, raw is not None


def inspect_scene_image(path: Path) -> SceneImageDiagnostics:
    """Decode, orient, and validate one JPEG, PNG, or WebP without retaining pixels."""
    if not path.is_file():
        msg = "The selected room image does not exist"
        raise ValueError(msg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                opened.load()
                image_format = opened.format
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    msg = "Choose a JPEG, PNG, or WebP room image"
                    raise ValueError(msg)
                exif = _exif_values(opened)
                raw_orientation = exif.get("Orientation", 1)
                orientation = (
                    raw_orientation
                    if isinstance(raw_orientation, int)
                    and 1 <= raw_orientation <= MAX_EXIF_ORIENTATION
                    else 1
                )
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                has_alpha = "A" in oriented.getbands() or "transparency" in opened.info
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        msg = "The selected room image is corrupt, truncated, or too large"
        raise ValueError(msg) from error
    if min(width, height) < MIN_IMAGE_DIMENSION:
        msg = f"Room images must be at least {MIN_IMAGE_DIMENSION} pixels on each edge"
        raise ValueError(msg)
    if max(width, height) > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
        msg = "The selected room image exceeds the safe dimension limit"
        raise ValueError(msg)
    focal_35mm, has_focal = _focal_length_35mm(exif)
    focal_px = focal_35mm * math.hypot(width, height) / math.hypot(36.0, 24.0)
    concerns: list[str] = []
    if not has_focal:
        concerns.append("No focal metadata was found; SHARP will use its 30 mm default.")
    if has_alpha:
        concerns.append("Transparent pixels will be flattened onto white before inference.")
    if max(width, height) > LARGE_IMAGE_WARNING_DIMENSION:
        concerns.append("The large source image may take longer to normalize.")
    return SceneImageDiagnostics(
        width=width,
        height=height,
        image_format=cast("Literal['JPEG', 'PNG', 'WEBP']", image_format),
        focal_length_px=focal_px,
        orientation_applied=orientation != 1,
        has_alpha=has_alpha,
        has_focal_metadata=has_focal,
        warnings=tuple(concerns),
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a managed trust boundary."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(FILE_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
