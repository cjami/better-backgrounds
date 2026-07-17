"""Upload-first Apple SHARP scene creation and managed checkpoint preparation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import warnings
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, SupportsFloat, SupportsIndex, cast
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from plyfile import PlyData
from pydantic import Field, HttpUrl

from better_backgrounds.protocol import JobEvent, ProgressEvent, WarningEvent
from better_backgrounds.scene import (
    AssetInstaller,
    AssetResource,
    CameraBounds,
    SceneProvenance,
    SceneReference,
    StrictModel,
    Vector3,
    Viewpoint,
    sharp_scene_transform,
)
from better_backgrounds.sharp_runtime import (
    SharpDevice,
    SharpDeviceRequest,
    ensure_vendored_sharp,
    resolve_sharp_device,
    run_sharp_inference,
)

SHARP_BUILDER_REVISION = "1eaa046834b81852261262b41b0919f5c1efdd2e"
SHARP_MODEL_LICENSE = "Apple Machine Learning Research Model License Agreement"
SHARP_DEPENDENCY_VERSIONS = {
    "plyfile": "1.1.2",
    "scipy": "1.16.2",
    "timm": "1.0.20",
}
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 100_000_000
MIN_IMAGE_DIMENSION = 256
MAX_GAUSSIANS = 5_000_000
FILE_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_FOCAL_LENGTH_MM = 30.0
MOBILE_FOCAL_THRESHOLD_MM = 10.0
LARGE_IMAGE_WARNING_DIMENSION = 8192
PLY_HEADER_PARTS = 3
PLY_INTRINSIC_VALUE_COUNT = 9
PLY_IMAGE_SIZE_VALUE_COUNT = 2
MIN_ROTATION_NORM = 1e-6
MAX_EXIF_ORIENTATION = 8

EmitEvent = Callable[[JobEvent], None]
CancellationCheck = Callable[[], bool]
CheckpointProgress = Callable[[int, int], None]
ResourceOpener = Callable[[str], BinaryIO]


class SharpCancelledError(RuntimeError):
    """Raised when a SHARP preparation or build cooperatively stops."""


@dataclass(frozen=True, slots=True)
class SceneImageSelection:
    """Identify one trusted local image selected for a room build."""

    display_name: str
    source_path: Path | None
    source_kind: Literal["upload", "camera"] = "upload"


@dataclass(frozen=True, slots=True)
class SharpBuildConfig:
    """Configure one isolated SHARP build."""

    device: SharpDeviceRequest
    checkpoint_path: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class SharpCapabilities:
    """Describe the selected official prediction device."""

    device_type: SharpDevice
    accelerated: bool


@dataclass(frozen=True, slots=True)
class SceneBuildRequest:
    """Bind one image selection to a stable worker job."""

    job_id: str
    selection: SceneImageSelection
    config: SharpBuildConfig


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


@dataclass(frozen=True, slots=True)
class SharpPlyMetadata:
    """Return validated SHARP camera metadata needed by the renderer."""

    gaussian_count: int
    image_size: tuple[int, int]
    focal_length_px: float
    field_of_view: float


class SharpCheckpointManifest(StrictModel):
    """Pin the official research checkpoint and its accepted license identity."""

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1, max_length=80)
    filename: str = Field(pattern=r"^[a-zA-Z0-9_.-]+\.pt$")
    url: HttpUrl
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    builder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_name: str = Field(min_length=1, max_length=100)
    license_url: HttpUrl


def load_sharp_checkpoint_manifest() -> SharpCheckpointManifest:
    """Load the exact checked-in SHARP checkpoint identity."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/sharp/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = SharpCheckpointManifest.model_validate_json(content)
    if manifest.builder_revision != SHARP_BUILDER_REVISION:
        msg = "The SHARP checkpoint manifest does not match the vendored source revision"
        raise ValueError(msg)
    return manifest


def probe_sharp_capabilities(
    requested: SharpDeviceRequest = "auto",
) -> SharpCapabilities:
    """Verify the pinned inference imports and resolve one usable device."""
    for package, expected in SHARP_DEPENDENCY_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            msg = f"The pinned SHARP dependency {package} is unavailable"
            raise RuntimeError(msg) from error
        if actual != expected:
            msg = f"SHARP requires {package}=={expected}, found {actual}"
            raise RuntimeError(msg)
    ensure_vendored_sharp()
    device = resolve_sharp_device(requested)
    return SharpCapabilities(device_type=device, accelerated=device != "cpu")


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


def _urlopen(url: str) -> BinaryIO:
    return urlopen(url, timeout=30)  # noqa: S310


class SharpCheckpointInstaller:
    """Download and atomically publish the license-gated official checkpoint."""

    def __init__(
        self,
        root: Path,
        *,
        manifest: SharpCheckpointManifest | None = None,
        opener: ResourceOpener = _urlopen,
    ) -> None:
        """Use a dedicated model cache and injectable download boundary."""
        self.root = root
        self.manifest = manifest or load_sharp_checkpoint_manifest()
        self._opener = opener

    @property
    def checkpoint_path(self) -> Path:
        """Return the pinned managed checkpoint path."""
        return self.root / self.manifest.filename

    @property
    def marker_path(self) -> Path:
        """Return the integrity marker beside the managed checkpoint."""
        return self.root / ".complete.json"

    def is_ready(self) -> bool:
        """Check cached size, identity, and modification evidence without rehashing 2.8 GB."""
        try:
            stat = self.checkpoint_path.stat()
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                return False
            return bool(
                stat.st_size == self.manifest.size
                and marker.get("sha256") == self.manifest.sha256
                and marker.get("size") == self.manifest.size
                and marker.get("mtime_ns") == stat.st_mtime_ns
            )
        except OSError, TypeError, ValueError:
            return False

    def prepare(
        self,
        *,
        license_accepted: bool,
        progress: CheckpointProgress | None = None,
        is_cancelled: CancellationCheck = lambda: False,
    ) -> Path:
        """Download, checksum, and publish only after explicit license acceptance."""
        if not license_accepted:
            msg = "The SHARP research-model license must be accepted before download"
            raise PermissionError(msg)
        if self.is_ready():
            return self.checkpoint_path
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.manifest.size + 512 * 1024 * 1024:
            msg = "There is not enough free disk space for the SHARP checkpoint"
            raise OSError(msg)
        temporary = self.root / f".{self.manifest.filename}.{uuid4().hex}.part"
        digest = hashlib.sha256()
        completed = 0
        try:
            with (
                closing(self._opener(str(self.manifest.url))) as response,
                temporary.open("wb") as destination,
            ):
                while chunk := response.read(FILE_CHUNK_SIZE):
                    if is_cancelled():
                        msg = "SHARP checkpoint preparation was cancelled"
                        raise SharpCancelledError(msg)
                    destination.write(chunk)
                    digest.update(chunk)
                    completed += len(chunk)
                    if completed > self.manifest.size:
                        msg = "The SHARP checkpoint exceeded its pinned size"
                        raise ValueError(msg)
                    if progress is not None:
                        progress(completed, self.manifest.size)
                destination.flush()
                os.fsync(destination.fileno())
            if completed != self.manifest.size or digest.hexdigest() != self.manifest.sha256:
                msg = "The SHARP checkpoint failed SHA-256 validation"
                raise ValueError(msg)
            temporary.replace(self.checkpoint_path)
            self._write_marker()
        finally:
            temporary.unlink(missing_ok=True)
        return self.checkpoint_path

    def validate(self, path: Path) -> None:
        """Require the official checkpoint identity before loading pickle-backed weights."""
        if path.resolve() == self.checkpoint_path.resolve() and self.is_ready():
            return
        if not path.is_file() or path.stat().st_size != self.manifest.size:
            msg = "The pinned SHARP checkpoint is missing or has the wrong size"
            raise ValueError(msg)
        if sha256_file(path) != self.manifest.sha256:
            msg = "The SHARP checkpoint failed SHA-256 validation"
            raise ValueError(msg)

    def _write_marker(self) -> None:
        stat = self.checkpoint_path.stat()
        payload = {
            "schema_version": 1,
            "sha256": self.manifest.sha256,
            "size": self.manifest.size,
            "mtime_ns": stat.st_mtime_ns,
        }
        temporary = self.marker_path.with_name(f".{self.marker_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.marker_path)
        finally:
            temporary.unlink(missing_ok=True)


def _parse_binary_ply_bounds(
    path: Path,
) -> tuple[int, dict[str, int], dict[str, dict[str, str]]]:
    type_sizes = {
        "char": 1,
        "uchar": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "ushort": 2,
        "int16": 2,
        "uint16": 2,
        "int": 4,
        "uint": 4,
        "int32": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    with path.open("rb") as source:
        header = source.read(64 * 1024)
    marker = b"end_header\n"
    end = header.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = header.find(marker)
    if not header.startswith(b"ply\n") and not header.startswith(b"ply\r\n"):
        msg = "SHARP produced an invalid PLY signature"
        raise ValueError(msg)
    if end < 0:
        msg = "SHARP produced an incomplete PLY header"
        raise ValueError(msg)
    header_size = end + len(marker)
    try:
        lines = header[:header_size].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        msg = "SHARP produced a non-ASCII PLY header"
        raise ValueError(msg) from error
    if "format binary_little_endian 1.0" not in lines:
        msg = "SHARP PLY must use binary little-endian encoding"
        raise ValueError(msg)
    counts: dict[str, int] = {}
    properties: dict[str, dict[str, str]] = {}
    record_sizes: dict[str, int] = {}
    current: str | None = None
    for line in lines:
        parts = line.split()
        if len(parts) == PLY_HEADER_PARTS and parts[0] == "element":
            current = parts[1]
            if current in counts:
                msg = "SHARP PLY contains a duplicate element declaration"
                raise ValueError(msg)
            try:
                counts[current] = int(parts[2])
            except ValueError as error:
                msg = "SHARP PLY contains an invalid element count"
                raise ValueError(msg) from error
            if counts[current] < 0:
                msg = "SHARP PLY contains a negative element count"
                raise ValueError(msg)
            properties[current] = {}
            record_sizes[current] = 0
        elif parts[:1] == ["property"] and current is not None:
            if len(parts) != PLY_HEADER_PARTS or parts[1] not in type_sizes:
                msg = "SHARP PLY contains an unsupported list or property type"
                raise ValueError(msg)
            property_name = parts[2]
            if property_name in properties[current]:
                msg = "SHARP PLY contains a duplicate property declaration"
                raise ValueError(msg)
            properties[current][property_name] = parts[1]
            record_sizes[current] += type_sizes[parts[1]]
    expected_size = header_size + sum(counts[name] * record_sizes.get(name, 0) for name in counts)
    if expected_size != path.stat().st_size:
        msg = "SHARP PLY file bounds do not match its header"
        raise ValueError(msg)
    return header_size, counts, properties


def validate_sharp_ply(
    path: Path,
    *,
    expected_image_size: tuple[int, int] | None = None,
) -> SharpPlyMetadata:
    """Validate binary bounds, Gaussian values, image size, and camera intrinsics."""
    if not path.is_file():
        msg = "SHARP did not produce a PLY file"
        raise ValueError(msg)
    _header_size, counts, properties = _parse_binary_ply_bounds(path)
    gaussian_count = counts.get("vertex", 0)
    if gaussian_count <= 0 or gaussian_count > MAX_GAUSSIANS:
        msg = "SHARP PLY contains an invalid Gaussian count"
        raise ValueError(msg)
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    vertex_properties = properties.get("vertex", {})
    if not required.issubset(vertex_properties):
        msg = "SHARP PLY is missing required 3D Gaussian properties"
        raise ValueError(msg)
    if any(vertex_properties[name] not in {"float", "float32"} for name in required):
        msg = "SHARP PLY Gaussian properties must use 32-bit floats"
        raise ValueError(msg)
    if (
        counts.get("intrinsic") != PLY_INTRINSIC_VALUE_COUNT
        or counts.get("image_size") != PLY_IMAGE_SIZE_VALUE_COUNT
    ):
        msg = "SHARP PLY is missing embedded image size or intrinsics"
        raise ValueError(msg)
    if properties.get("intrinsic") != {"intrinsic": "float"}:
        msg = "SHARP PLY intrinsics must use the official 32-bit float layout"
        raise ValueError(msg)
    if properties.get("image_size") != {"image_size": "uint"}:
        msg = "SHARP PLY image size must use the official unsigned 32-bit layout"
        raise ValueError(msg)
    try:
        ply = PlyData.read(path, mmap=False)
        vertices = ply["vertex"].data
        for name in required:
            if not np.isfinite(np.asarray(vertices[name])).all():
                msg = f"SHARP PLY contains non-finite {name} values"
                raise ValueError(msg)
        scales = np.column_stack([np.asarray(vertices[f"scale_{index}"]) for index in range(3)])
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.isfinite(np.exp(scales)).all():
                msg = "SHARP PLY contains unusable Gaussian scales"
                raise ValueError(msg)
        rotations = np.column_stack([np.asarray(vertices[f"rot_{index}"]) for index in range(4)])
        if np.any(np.linalg.norm(rotations, axis=1) <= MIN_ROTATION_NORM):
            msg = "SHARP PLY contains unusable Gaussian rotations"
            raise ValueError(msg)
        image_values = np.asarray(ply["image_size"].data["image_size"]).reshape(-1)
        intrinsic_values = np.asarray(ply["intrinsic"].data["intrinsic"]).reshape(-1)
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("SHARP PLY"):
            raise
        msg = "SHARP PLY binary data is invalid or truncated"
        raise ValueError(msg) from error
    width, height = (int(image_values[0]), int(image_values[1]))
    if width <= 0 or height <= 0 or max(width, height) > MAX_IMAGE_DIMENSION:
        msg = "SHARP PLY contains an invalid embedded image size"
        raise ValueError(msg)
    if expected_image_size is not None and (width, height) != expected_image_size:
        msg = "SHARP PLY image size does not match the selected source"
        raise ValueError(msg)
    if (
        len(intrinsic_values) != PLY_INTRINSIC_VALUE_COUNT
        or not np.isfinite(intrinsic_values).all()
    ):
        msg = "SHARP PLY contains invalid camera intrinsics"
        raise ValueError(msg)
    intrinsics = intrinsic_values.reshape(3, 3)
    focal_x, focal_y = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    principal_x, principal_y = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if (
        focal_x <= 0
        or focal_y <= 0
        or not 0 <= principal_x <= width
        or not 0 <= principal_y <= height
        or not math.isclose(focal_x, focal_y, rel_tol=1e-3)
        or not math.isclose(float(intrinsics[0, 1]), 0.0, abs_tol=1e-5)
        or not math.isclose(float(intrinsics[1, 0]), 0.0, abs_tol=1e-5)
        or not math.isclose(float(intrinsics[2, 0]), 0.0, abs_tol=1e-5)
        or not math.isclose(float(intrinsics[2, 1]), 0.0, abs_tol=1e-5)
        or not math.isclose(float(intrinsics[2, 2]), 1.0, abs_tol=1e-5)
    ):
        msg = "SHARP PLY contains unusable camera intrinsics"
        raise ValueError(msg)
    field_of_view = math.degrees(2.0 * math.atan(height / (2.0 * focal_y)))
    if not math.isfinite(field_of_view) or field_of_view <= 0:
        msg = "SHARP PLY contains an unusable camera field of view"
        raise ValueError(msg)
    return SharpPlyMetadata(
        gaussian_count=gaussian_count,
        image_size=(width, height),
        focal_length_px=(focal_x + focal_y) / 2.0,
        field_of_view=field_of_view,
    )


class SharpPredictor(Protocol):
    """Allow official inference to be replaced only at test boundaries."""

    def __call__(
        self,
        image_path: Path,
        focal_length_px: float,
        checkpoint_path: Path,
        device: SharpDevice,
        output_path: Path,
        model_loaded: Callable[[], None],
    ) -> float:
        """Write one SHARP PLY and return accelerator-synchronized inference time."""
        ...


def _sharp_viewpoint(metadata: SharpPlyMetadata) -> Viewpoint:
    width, height = metadata.image_size
    return Viewpoint(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orbit_target=Vector3(x=0.0, y=0.0, z=-2.0),
        field_of_view=min(max(metadata.field_of_view, 24.0), 90.0),
        aspect_ratio=width / height,
        horizon=0.0,
        near_clip=0.03,
        far_clip=30.0,
        scene_transform=sharp_scene_transform(),
        safe_camera_region=CameraBounds(
            minimum=Vector3(x=-0.35, y=-0.25, z=-0.35),
            maximum=Vector3(x=0.35, y=0.25, z=0.35),
        ),
    )


def _normalized_input(source: Path, destination: Path) -> None:
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        if "A" in oriented.getbands() or "transparency" in opened.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            normalized = background.convert("RGB")
        else:
            normalized = oriented.convert("RGB")
        normalized.save(destination, format="PNG")


def _preview_image(normalized: Path, destination: Path) -> None:
    with Image.open(normalized) as opened:
        preview = opened.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    preview.save(destination, format="WEBP", quality=88, method=6)


def _scene_id(selection: SceneImageSelection, source_sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(selection.display_name).stem.lower()).strip("-")
    return f"{(stem or 'room')[:48]}-{source_sha256[:10]}"


class SharpSceneBuilder:
    """Build, validate, and atomically adopt one upload-first SHARP scene."""

    def __init__(
        self,
        *,
        predictor: SharpPredictor = run_sharp_inference,
        manifest: SharpCheckpointManifest | None = None,
    ) -> None:
        """Inject only the heavyweight inference edge and pinned manifest."""
        self._predictor = predictor
        self._manifest = manifest or load_sharp_checkpoint_manifest()

    def build(
        self,
        request: SceneBuildRequest,
        emit_event: EmitEvent,
        is_cancelled: CancellationCheck,
    ) -> SceneReference:
        """Run every gate and return only a verified managed SceneReference."""
        selection = request.selection
        source = selection.source_path
        if source is None:
            msg = "A SHARP build requires a selected room image"
            raise ValueError(msg)
        staging = request.config.output_root / f".sharp-{request.job_id}-{uuid4().hex}.part"
        try:
            self._progress(emit_event, request.job_id, "validation", 0.02, "Validating image")
            diagnostics = inspect_scene_image(source)
            source_sha256 = sha256_file(source)
            staging.mkdir(parents=True)
            normalized = staging / "input.png"
            preview = staging / "preview.webp"
            scene_path = staging / "scene.ply"
            _normalized_input(source, normalized)
            _preview_image(normalized, preview)
            for index, concern in enumerate(diagnostics.warnings):
                emit_event(
                    WarningEvent(
                        job_id=request.job_id,
                        code=f"image_warning_{index + 1}",
                        message=concern,
                    )
                )
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "model_preparation",
                0.16,
                "Verifying the pinned SHARP checkpoint",
            )
            checkpoint = SharpCheckpointInstaller(
                request.config.checkpoint_path.parent,
                manifest=self._manifest,
            )
            checkpoint.validate(request.config.checkpoint_path)
            capabilities = probe_sharp_capabilities(request.config.device)
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "model_loading",
                0.26,
                f"Loading SHARP on {capabilities.device_type.upper()}",
            )

            def model_loaded() -> None:
                self._progress(
                    emit_event,
                    request.job_id,
                    "inference",
                    0.38,
                    "Predicting metric 3D Gaussians",
                )

            inference_ms = self._predictor(
                normalized,
                diagnostics.focal_length_px,
                request.config.checkpoint_path,
                capabilities.device_type,
                scene_path,
                model_loaded,
            )
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "ply_validation",
                0.75,
                "Validating Gaussian scene and camera metadata",
            )
            metadata = validate_sharp_ply(
                scene_path,
                expected_image_size=(diagnostics.width, diagnostics.height),
            )
            self._cancel_if_requested(is_cancelled)
            normalized.unlink(missing_ok=True)
            reference = self._reference(
                selection,
                staging,
                source_sha256,
                diagnostics,
                metadata,
                capabilities,
                inference_ms,
            )
            self._progress(
                emit_event,
                request.job_id,
                "publication",
                0.9,
                "Publishing verified scene resources",
            )
            AssetInstaller(request.config.output_root).adopt(reference, staging)
            self._progress(
                emit_event,
                request.job_id,
                "preview_generation",
                0.98,
                "Preview ready",
            )
            return reference
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _reference(
        self,
        selection: SceneImageSelection,
        staging: Path,
        source_sha256: str,
        diagnostics: SceneImageDiagnostics,
        metadata: SharpPlyMetadata,
        capabilities: SharpCapabilities,
        inference_ms: float,
    ) -> SceneReference:
        resources = tuple(
            AssetResource(
                path=path.relative_to(staging).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in (staging / "scene.ply", staging / "preview.webp")
        )
        display_name = Path(selection.display_name).stem.replace("_", " ").strip().title()
        return SceneReference(
            asset_id=_scene_id(selection, source_sha256),
            display_name=display_name or "Room",
            format="ply",
            entrypoint="scene.ply",
            resources=resources,
            license_name="User source; SHARP research-model output",
            license_url=self._manifest.license_url,
            attribution=f"Generated locally with Apple SHARP from {selection.display_name}",
            attribution_url="https://github.com/apple/ml-sharp",
            preview="preview.webp",
            default_viewpoint=_sharp_viewpoint(metadata),
            provenance=SceneProvenance(
                source_kind=selection.source_kind,
                source_sha256=source_sha256,
                source_size=(diagnostics.width, diagnostics.height),
                builder_revision=self._manifest.builder_revision,
                checkpoint_sha256=self._manifest.sha256,
                device=capabilities.device_type,
                inference_ms=inference_ms,
                license_name=self._manifest.license_name,
                license_url=self._manifest.license_url,
            ),
        )

    @staticmethod
    def _progress(
        emit: EmitEvent,
        job_id: str,
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        emit(ProgressEvent(job_id=job_id, stage=stage, progress=progress, message=message))

    @staticmethod
    def _cancel_if_requested(is_cancelled: CancellationCheck) -> None:
        if is_cancelled():
            msg = "SHARP room build was cancelled"
            raise SharpCancelledError(msg)
