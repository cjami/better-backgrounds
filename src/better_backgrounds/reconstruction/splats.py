"""Validation and atomic publication for user-provided Gaussian scenes."""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np

from better_backgrounds.jobs.events import JobEvent, ProgressEvent
from better_backgrounds.reconstruction.images import sha256_file
from better_backgrounds.reconstruction.sharp.ply import MAX_GAUSSIANS, validate_sharp_ply
from better_backgrounds.reconstruction.ssog import (
    StreamedSogInspection,
    extract_streamed_sog,
    inspect_streamed_sog,
)
from better_backgrounds.scene import (
    AssetInstaller,
    AssetResource,
    CameraBounds,
    Quaternion,
    SceneReference,
    SceneTransform,
    Vector3,
    Viewpoint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

PLY_HEADER_LIMIT = 64 * 1024
PLY_SAMPLE_LIMIT = 200_000
CANONICAL_HALF_EXTENT = 2.5
DEFAULT_FIELD_OF_VIEW = 42.0
DEFAULT_ASPECT_RATIO = 16 / 9
STREAMED_FIELD_OF_VIEW = 80.0
STREAMED_ENTRY_SIDE = 0.65
STREAMED_ENTRY_DEPTH = 1.06
STREAMED_LOOK_DISTANCE = 0.9
STREAMED_NAVIGATION_PADDING = 0.25
PLY_DECLARATION_PARTS = 3
COMPRESSED_BASE_ELEMENT_COUNT = 2
MIN_ROTATION_NORM = 1e-6
MIN_SCENE_EXTENT = 1e-6

_PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("i1", 1),
    "uchar": ("u1", 1),
    "short": ("<i2", 2),
    "ushort": ("<u2", 2),
    "int": ("<i4", 4),
    "uint": ("<u4", 4),
    "float": ("<f4", 4),
    "double": ("<f8", 8),
}
_STANDARD_PROPERTIES = frozenset(
    {
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
)
_COMPRESSED_CHUNK_PROPERTIES = (
    "min_x",
    "min_y",
    "min_z",
    "max_x",
    "max_y",
    "max_z",
    "min_scale_x",
    "min_scale_y",
    "min_scale_z",
    "max_scale_x",
    "max_scale_y",
    "max_scale_z",
)
_COMPRESSED_COLOUR_PROPERTIES = (
    "min_r",
    "min_g",
    "min_b",
    "max_r",
    "max_g",
    "max_b",
)
_COMPRESSED_VERTEX_PROPERTIES = (
    "packed_position",
    "packed_rotation",
    "packed_scale",
    "packed_color",
)


@dataclass(frozen=True, slots=True)
class SplatSelection:
    """Identify one local Gaussian scene selected for managed import."""

    display_name: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class SplatDiagnostics:
    """Describe a renderer-compatible Gaussian scene and its initial framing."""

    gaussian_count: int
    file_size: int
    layout: Literal["standard", "compressed", "sharp", "streamed-sog"]
    framing: Literal[
        "Embedded SHARP camera",
        "Automatic COLMAP framing",
        "Automatic streamed bounds",
    ]
    bounds_minimum: tuple[float, float, float]
    bounds_maximum: tuple[float, float, float]
    lod_levels: int = 1
    resource_count: int = 1
    total_gaussian_count: int | None = None
    navigation_bounds_minimum: tuple[float, float, float] | None = None
    navigation_bounds_maximum: tuple[float, float, float] | None = None

    @property
    def encoding(self) -> str:
        """Return a concise user-facing encoding name."""
        return "Streamed SOG" if self.layout == "streamed-sog" else "Binary little-endian"


@dataclass(frozen=True, slots=True)
class SplatImportConfig:
    """Configure one isolated direct-import job."""

    output_root: Path


@dataclass(frozen=True, slots=True)
class SplatImportRequest:
    """Bind one selected PLY to a stable worker job."""

    job_id: str
    selection: SplatSelection
    config: SplatImportConfig


class SplatImportCancelledError(RuntimeError):
    """Raised when a direct import is cancelled at a safe boundary."""


@dataclass(frozen=True, slots=True)
class _PlyProperty:
    name: str
    type_name: str


@dataclass(frozen=True, slots=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]

    @property
    def record_size(self) -> int:
        return sum(_PLY_TYPES[item.type_name][1] for item in self.properties)


@dataclass(frozen=True, slots=True)
class _PlyHeader:
    size: int
    elements: tuple[_PlyElement, ...]

    def element(self, name: str) -> _PlyElement | None:
        return next((item for item in self.elements if item.name == name), None)

    def offset(self, name: str) -> int:
        offset = self.size
        for element in self.elements:
            if element.name == name:
                return offset
            offset += element.count * element.record_size
        raise KeyError(name)


def _read_header(path: Path) -> _PlyHeader:  # noqa: C901, PLR0912, PLR0915
    if not path.is_file():
        msg = "The selected Gaussian PLY does not exist"
        raise ValueError(msg)
    with path.open("rb") as source:
        payload = source.read(PLY_HEADER_LIMIT)
    marker = b"end_header\n"
    end = payload.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = payload.find(marker)
    if not payload.startswith((b"ply\n", b"ply\r\n")):
        msg = "The selected file does not have a valid PLY signature"
        raise ValueError(msg)
    if end < 0:
        msg = "The Gaussian PLY header is incomplete or too large"
        raise ValueError(msg)
    header_size = end + len(marker)
    try:
        lines = payload[:header_size].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        msg = "The Gaussian PLY header must contain ASCII metadata"
        raise ValueError(msg) from error
    format_lines = [line for line in lines if line.startswith("format ")]
    if format_lines != ["format binary_little_endian 1.0"]:
        msg = "Gaussian PLY imports must use binary little-endian encoding"
        raise ValueError(msg)

    elements: list[_PlyElement] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[_PlyProperty] = []

    def finish_element() -> None:
        if current_name is not None:
            elements.append(
                _PlyElement(current_name, current_count, tuple(current_properties)),
            )

    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            if len(parts) != PLY_DECLARATION_PARTS:
                msg = "Gaussian PLY contains an invalid element declaration"
                raise ValueError(msg)
            finish_element()
            current_name = parts[1]
            if any(element.name == current_name for element in elements):
                msg = "Gaussian PLY contains a duplicate element declaration"
                raise ValueError(msg)
            try:
                current_count = int(parts[2])
            except ValueError as error:
                msg = "Gaussian PLY contains an invalid element count"
                raise ValueError(msg) from error
            if current_count < 0:
                msg = "Gaussian PLY contains a negative element count"
                raise ValueError(msg)
            current_properties = []
        elif parts[:1] == ["property"]:
            if (
                current_name is None
                or len(parts) != PLY_DECLARATION_PARTS
                or parts[1] not in _PLY_TYPES
            ):
                msg = "Gaussian PLY contains an unsupported list or property type"
                raise ValueError(msg)
            if any(item.name == parts[2] for item in current_properties):
                msg = "Gaussian PLY contains a duplicate property declaration"
                raise ValueError(msg)
            current_properties.append(_PlyProperty(parts[2], parts[1]))
        elif parts[:1] not in (["ply"], ["format"], ["comment"], ["end_header"]):
            msg = "Gaussian PLY contains unsupported header metadata"
            raise ValueError(msg)
    finish_element()
    expected_size = header_size + sum(element.count * element.record_size for element in elements)
    if expected_size != path.stat().st_size:
        msg = "Gaussian PLY file bounds do not match its header"
        raise ValueError(msg)
    return _PlyHeader(header_size, tuple(elements))


def _dtype(element: _PlyElement) -> np.dtype[np.void]:
    return np.dtype(
        [(item.name, _PLY_TYPES[item.type_name][0]) for item in element.properties],
        align=False,
    )


def _vector_tuple(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _property_map(element: _PlyElement) -> dict[str, str]:
    return {item.name: item.type_name for item in element.properties}


def _is_float32(type_name: str) -> bool:
    return type_name == "float"


def _validate_standard(
    path: Path,
    header: _PlyHeader,
    vertex: _PlyElement,
    *,
    validate_values: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    properties = _property_map(vertex)
    if not _STANDARD_PROPERTIES.issubset(properties):
        msg = "Gaussian PLY is missing required 3D Gaussian properties"
        raise ValueError(msg)
    if any(not _is_float32(properties[name]) for name in _STANDARD_PROPERTIES):
        msg = "Gaussian PLY properties must use 32-bit floats"
        raise ValueError(msg)
    data = np.memmap(
        path,
        dtype=_dtype(vertex),
        mode="r",
        offset=header.offset("vertex"),
        shape=(vertex.count,),
    )
    if validate_values:
        for name in _STANDARD_PROPERTIES:
            if not np.isfinite(data[name]).all():
                msg = f"Gaussian PLY contains non-finite {name} values"
                raise ValueError(msg)
        scales = np.column_stack([data[f"scale_{index}"] for index in range(3)])
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.isfinite(np.exp(scales)).all():
                msg = "Gaussian PLY contains unusable Gaussian scales"
                raise ValueError(msg)
        rotations = np.column_stack([data[f"rot_{index}"] for index in range(4)])
        if np.any(np.linalg.norm(rotations, axis=1) <= MIN_ROTATION_NORM):
            msg = "Gaussian PLY contains unusable Gaussian rotations"
            raise ValueError(msg)
    sample_count = min(vertex.count, PLY_SAMPLE_LIMIT)
    indices = np.linspace(0, vertex.count - 1, sample_count, dtype=np.int64)
    positions = np.column_stack([data[name][indices] for name in ("x", "y", "z")])
    if not np.isfinite(positions).all():
        msg = "Gaussian PLY contains non-finite positions"
        raise ValueError(msg)
    minimum = _vector_tuple(np.quantile(positions, 0.01, axis=0))
    maximum = _vector_tuple(np.quantile(positions, 0.99, axis=0))
    return minimum, maximum


def _validate_compressed(
    path: Path,
    header: _PlyHeader,
    vertex: _PlyElement,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    chunk = header.element("chunk")
    if chunk is None or header.elements[0].name != "chunk":
        return None
    expected_elements = (
        ("chunk", "vertex")
        if len(header.elements) == COMPRESSED_BASE_ELEMENT_COUNT
        else ("chunk", "vertex", "sh")
    )
    if tuple(element.name for element in header.elements) != expected_elements:
        msg = "Compressed Gaussian PLY has unsupported elements"
        raise ValueError(msg)
    chunk_names = tuple(item.name for item in chunk.properties)
    expected_chunk_names = (
        _COMPRESSED_CHUNK_PROPERTIES,
        _COMPRESSED_CHUNK_PROPERTIES + _COMPRESSED_COLOUR_PROPERTIES,
    )
    if chunk_names not in expected_chunk_names or any(
        not _is_float32(item.type_name) for item in chunk.properties
    ):
        msg = "Compressed Gaussian PLY has an unsupported chunk layout"
        raise ValueError(msg)
    if tuple(item.name for item in vertex.properties) != _COMPRESSED_VERTEX_PROPERTIES or any(
        item.type_name != "uint" for item in vertex.properties
    ):
        msg = "Compressed Gaussian PLY has an unsupported packed vertex layout"
        raise ValueError(msg)
    expected_chunks = math.ceil(vertex.count / 256)
    if chunk.count != expected_chunks:
        msg = "Compressed Gaussian PLY has an invalid chunk count"
        raise ValueError(msg)
    sh = header.element("sh")
    if sh is not None and (
        sh.count != vertex.count
        or len(sh.properties) not in {9, 24, 45}
        or any(
            item.name != f"f_rest_{index}" or item.type_name != "uchar"
            for index, item in enumerate(sh.properties)
        )
    ):
        msg = "Compressed Gaussian PLY has an unsupported spherical-harmonic layout"
        raise ValueError(msg)
    chunks = np.memmap(
        path,
        dtype=_dtype(chunk),
        mode="r",
        offset=header.offset("chunk"),
        shape=(chunk.count,),
    )
    values = np.column_stack([chunks[name] for name in _COMPRESSED_CHUNK_PROPERTIES])
    if not np.isfinite(values).all():
        msg = "Compressed Gaussian PLY contains non-finite chunk bounds"
        raise ValueError(msg)
    minimums = values[:, :3]
    maximums = values[:, 3:6]
    if np.any(minimums > maximums):
        msg = "Compressed Gaussian PLY contains inverted position bounds"
        raise ValueError(msg)
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(np.exp(values[:, 6:12])).all():
            msg = "Compressed Gaussian PLY contains unusable Gaussian scales"
            raise ValueError(msg)
    minimum = _vector_tuple(np.quantile(minimums, 0.01, axis=0))
    maximum = _vector_tuple(np.quantile(maximums, 0.99, axis=0))
    return minimum, maximum


def inspect_gaussian_ply(path: Path, *, validate_values: bool = True) -> SplatDiagnostics:
    """Validate one renderer-compatible Gaussian PLY and return framing evidence."""
    if path.suffix.lower() != ".ply":
        msg = "Choose a Gaussian PLY file"
        raise ValueError(msg)
    header = _read_header(path)
    vertex = header.element("vertex")
    if vertex is None or vertex.count <= 0 or vertex.count > MAX_GAUSSIANS:
        msg = "Gaussian PLY contains an invalid Gaussian count"
        raise ValueError(msg)
    compressed_bounds = _validate_compressed(path, header, vertex)
    if compressed_bounds is None:
        bounds = _validate_standard(
            path,
            header,
            vertex,
            validate_values=validate_values,
        )
        layout: Literal["standard", "compressed", "sharp"] = "standard"
    else:
        bounds = compressed_bounds
        layout = "compressed"
    framing: Literal["Embedded SHARP camera", "Automatic COLMAP framing"]
    try:
        validate_sharp_ply(path)
    except ValueError:
        framing = "Automatic COLMAP framing"
    else:
        layout = "sharp"
        framing = "Embedded SHARP camera"
    return SplatDiagnostics(
        gaussian_count=vertex.count,
        file_size=path.stat().st_size,
        layout=layout,
        framing=framing,
        bounds_minimum=bounds[0],
        bounds_maximum=bounds[1],
    )


def _streamed_diagnostics(inspection: StreamedSogInspection) -> SplatDiagnostics:
    return SplatDiagnostics(
        gaussian_count=inspection.gaussian_count,
        file_size=inspection.file_size,
        layout="streamed-sog",
        framing="Automatic streamed bounds",
        bounds_minimum=inspection.bounds_minimum,
        bounds_maximum=inspection.bounds_maximum,
        lod_levels=inspection.lod_levels,
        resource_count=len(inspection.resources),
        total_gaussian_count=inspection.total_gaussian_count,
        navigation_bounds_minimum=inspection.navigation_bounds_minimum,
        navigation_bounds_maximum=inspection.navigation_bounds_maximum,
    )


def inspect_gaussian_scene(path: Path, *, validate_values: bool = True) -> SplatDiagnostics:
    """Validate a supported PLY or packaged Streamed SOG scene."""
    if path.suffix.lower() == ".ply":
        return inspect_gaussian_ply(path, validate_values=validate_values)
    return _streamed_diagnostics(
        inspect_streamed_sog(path, validate_payloads=validate_values),
    )


def _generic_viewpoint(
    diagnostics: SplatDiagnostics,
    *,
    colmap_coordinates: bool = True,
) -> Viewpoint:
    source_minimum = diagnostics.bounds_minimum
    source_maximum = diagnostics.bounds_maximum
    source_center = tuple(
        (minimum + maximum) / 2
        for minimum, maximum in zip(source_minimum, source_maximum, strict=True)
    )
    half_extents = tuple(
        (maximum - minimum) / 2
        for minimum, maximum in zip(source_minimum, source_maximum, strict=True)
    )
    maximum_extent = max(half_extents)
    if not math.isfinite(maximum_extent) or maximum_extent <= MIN_SCENE_EXTENT:
        msg = "Gaussian scene does not contain a usable spatial extent"
        raise ValueError(msg)
    scale = min(100.0, max(0.01, CANONICAL_HALF_EXTENT / maximum_extent))
    transformed_center = tuple(item * scale for item in source_center)
    orientation = Quaternion()
    camera_direction = 1.0
    if colmap_coordinates:
        transformed_center = (
            -source_center[0] * scale,
            -source_center[1] * scale,
            source_center[2] * scale,
        )
        orientation = Quaternion(z=1.0, w=0.0)
        camera_direction = -1.0
    target = Vector3(x=0.0, y=1.1, z=0.0)
    transform = SceneTransform(
        translation=Vector3(
            x=target.x - transformed_center[0],
            y=target.y - transformed_center[1],
            z=target.z - transformed_center[2],
        ),
        orientation=orientation,
        scale=scale,
    )
    fitted_x, fitted_y, fitted_z = (extent * scale for extent in half_extents)
    vertical_distance = fitted_y / math.tan(math.radians(DEFAULT_FIELD_OF_VIEW / 2))
    horizontal_fov = 2 * math.atan(
        math.tan(math.radians(DEFAULT_FIELD_OF_VIEW / 2)) * DEFAULT_ASPECT_RATIO,
    )
    horizontal_distance = fitted_x / math.tan(horizontal_fov / 2)
    distance = max(2.0, vertical_distance, horizontal_distance) + fitted_z
    far_clip = min(1_000.0, max(40.0, distance + fitted_z * 4 + 10.0))
    position = Vector3(x=target.x, y=target.y, z=target.z + camera_direction * distance)
    padding = max(0.5, min(4.0, maximum_extent * scale * 0.35))
    return Viewpoint(
        position=position,
        orbit_target=target,
        field_of_view=DEFAULT_FIELD_OF_VIEW,
        horizon=0.0,
        near_clip=0.03,
        far_clip=far_clip,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        scene_transform=transform,
        safe_camera_region=CameraBounds(
            minimum=Vector3(
                x=position.x - padding,
                y=position.y - padding,
                z=position.z - padding,
            ),
            maximum=Vector3(
                x=position.x + padding,
                y=position.y + padding,
                z=position.z + padding,
            ),
        ),
    )


def _streamed_viewpoint(diagnostics: SplatDiagnostics) -> Viewpoint:
    source_minimum = diagnostics.bounds_minimum
    source_maximum = diagnostics.bounds_maximum
    minimum = (-source_maximum[0], -source_maximum[1], source_minimum[2])
    maximum = (-source_minimum[0], -source_minimum[1], source_maximum[2])
    navigation_source_minimum = diagnostics.navigation_bounds_minimum or source_minimum
    navigation_source_maximum = diagnostics.navigation_bounds_maximum or source_maximum
    navigation_minimum = (
        -navigation_source_maximum[0],
        -navigation_source_maximum[1],
        navigation_source_minimum[2],
    )
    navigation_maximum = (
        -navigation_source_minimum[0],
        -navigation_source_minimum[1],
        navigation_source_maximum[2],
    )
    spans = tuple(high - low for low, high in zip(minimum, maximum, strict=True))
    if any(not math.isfinite(span) or span <= MIN_SCENE_EXTENT for span in spans):
        msg = "Streamed SOG scene does not contain a usable spatial extent"
        raise ValueError(msg)
    center = tuple((low + high) / 2 for low, high in zip(minimum, maximum, strict=True))
    half_x, _half_y, half_z = (span / 2 for span in spans)
    horizontal_extent = max(spans[0], spans[2])
    ceiling_clearance = min(1.7, max(0.5, spans[1] * 0.065))
    position = Vector3(
        x=center[0] + half_x * STREAMED_ENTRY_SIDE,
        y=maximum[1] - ceiling_clearance,
        z=center[2] + half_z * STREAMED_ENTRY_DEPTH,
    )
    inward_x = center[0] - position.x
    inward_z = center[2] - position.z
    inward_length = math.hypot(inward_x, inward_z)
    if inward_length <= MIN_SCENE_EXTENT:
        inward_x, inward_z, inward_length = 0.0, -1.0, 1.0
    look_distance = min(
        1.5,
        max(STREAMED_LOOK_DISTANCE, horizontal_extent * 0.08),
    )
    target = Vector3(
        x=position.x + inward_x / inward_length * look_distance,
        y=position.y,
        z=position.z + inward_z / inward_length * look_distance,
    )
    navigation_spans = tuple(
        high - low for low, high in zip(navigation_minimum, navigation_maximum, strict=True)
    )
    padding = tuple(max(0.5, span * STREAMED_NAVIGATION_PADDING) for span in navigation_spans)
    diagonal = math.hypot(*navigation_spans)
    return Viewpoint(
        position=position,
        orbit_target=target,
        field_of_view=STREAMED_FIELD_OF_VIEW,
        horizon=0.0,
        near_clip=0.03,
        far_clip=min(1_000.0, max(40.0, diagonal * 8.0)),
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        scene_transform=SceneTransform(orientation=Quaternion(z=1.0, w=0.0)),
        safe_camera_region=CameraBounds(
            minimum=Vector3(
                x=navigation_minimum[0] - padding[0],
                y=navigation_minimum[1] - padding[1],
                z=navigation_minimum[2] - padding[2],
            ),
            maximum=Vector3(
                x=navigation_maximum[0] + padding[0],
                y=navigation_maximum[1] + padding[1],
                z=navigation_maximum[2] + padding[2],
            ),
        ),
    )


def _sharp_viewpoint(path: Path) -> Viewpoint:
    metadata = validate_sharp_ply(path)
    width, height = metadata.image_size
    return Viewpoint(
        position=Vector3(),
        orbit_target=Vector3(z=-2.0),
        field_of_view=min(max(metadata.field_of_view, 24.0), 90.0),
        aspect_ratio=width / height,
        horizon=0.0,
        near_clip=0.03,
        far_clip=30.0,
        scene_transform=SceneTransform(orientation=Quaternion(x=1.0, w=0.0)),
        safe_camera_region=CameraBounds(
            minimum=Vector3(x=-0.35, y=-0.25, z=-0.35),
            maximum=Vector3(x=0.35, y=0.25, z=0.35),
        ),
    )


def _scene_id(display_name: str, source_sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(display_name).stem.lower()).strip("-")
    return f"{(stem or 'room')[:48]}-{source_sha256[:10]}"


class SplatSceneImporter:
    """Validate and atomically adopt one user-provided Gaussian scene."""

    def import_scene(
        self,
        request: SplatImportRequest,
        emit_event: Callable[[JobEvent], None],
        is_cancelled: Callable[[], bool],
    ) -> SceneReference:
        """Copy or extract, validate, frame, and publish one managed scene."""
        source = request.selection.source_path
        if source.suffix.lower() not in {".ply", ".ssog", ".zip"}:
            msg = "Choose a Gaussian PLY or Streamed SOG archive"
            raise ValueError(msg)
        staging = request.config.output_root / f".splat-{request.job_id}-{uuid4().hex}.part"
        try:
            self._progress(emit_event, request.job_id, "validation", 0.05, "Reading Gaussian scene")
            staging.mkdir(parents=True)
            self._progress(
                emit_event,
                request.job_id,
                "ply_validation",
                0.4,
                "Validating Gaussian scene",
            )
            if source.suffix.lower() == ".ply":
                reference = self._import_ply(request.selection, source, staging, is_cancelled)
            else:
                reference = self._import_streamed_sog(
                    request.selection,
                    source,
                    staging,
                    is_cancelled,
                )
            self._cancel_if_requested(is_cancelled)
            self._progress(
                emit_event,
                request.job_id,
                "publication",
                0.85,
                "Publishing imported room",
            )
            AssetInstaller(request.config.output_root).adopt(reference, staging)
            self._progress(emit_event, request.job_id, "publication", 1.0, "Imported room ready")
            return reference
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _import_ply(
        self,
        selection: SplatSelection,
        source: Path,
        staging: Path,
        is_cancelled: Callable[[], bool],
    ) -> SceneReference:
        scene_path = staging / "scene.ply"
        self._copy(source, scene_path, is_cancelled)
        self._cancel_if_requested(is_cancelled)
        diagnostics = inspect_gaussian_ply(scene_path)
        source_sha256 = sha256_file(scene_path)
        viewpoint = (
            _sharp_viewpoint(scene_path)
            if diagnostics.layout == "sharp"
            else _generic_viewpoint(diagnostics)
        )
        return self._reference(
            selection,
            source_sha256,
            viewpoint,
            format_name="ply",
            entrypoint="scene.ply",
            resources=(
                AssetResource(
                    path="scene.ply",
                    size=scene_path.stat().st_size,
                    sha256=source_sha256,
                ),
            ),
        )

    def _import_streamed_sog(
        self,
        selection: SplatSelection,
        source: Path,
        staging: Path,
        is_cancelled: Callable[[], bool],
    ) -> SceneReference:
        self._cancel_if_requested(is_cancelled)
        inspection = inspect_streamed_sog(source)
        self._cancel_if_requested(is_cancelled)
        try:
            extracted = extract_streamed_sog(source, inspection, staging, is_cancelled)
        except InterruptedError as error:
            raise SplatImportCancelledError from error
        diagnostics = _streamed_diagnostics(inspection)
        return self._reference(
            selection,
            sha256_file(source),
            _streamed_viewpoint(diagnostics),
            format_name="ssog",
            entrypoint="lod-meta.json",
            resources=tuple(
                AssetResource(path=item.path, size=item.size, sha256=item.sha256)
                for item in extracted
            ),
        )

    @staticmethod
    def _copy(source: Path, destination: Path, is_cancelled: Callable[[], bool]) -> None:
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            while chunk := input_file.read(4 * 1024 * 1024):
                if is_cancelled():
                    raise SplatImportCancelledError
                output_file.write(chunk)

    @staticmethod
    def _reference(  # noqa: PLR0913
        selection: SplatSelection,
        source_sha256: str,
        viewpoint: Viewpoint,
        *,
        format_name: Literal["ply", "ssog"],
        entrypoint: str,
        resources: tuple[AssetResource, ...],
    ) -> SceneReference:
        display_name = Path(selection.display_name).stem.replace("_", " ").strip().title()
        return SceneReference(
            asset_id=_scene_id(selection.display_name, source_sha256),
            display_name=display_name or "Room",
            format=format_name,
            entrypoint=entrypoint,
            resources=resources,
            license_name="User-provided asset",
            attribution=f"Imported from {selection.display_name}",
            default_viewpoint=viewpoint,
        )

    @staticmethod
    def _progress(
        emit: Callable[[JobEvent], None],
        job_id: str,
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        emit(ProgressEvent(job_id=job_id, stage=stage, progress=progress, message=message))

    @staticmethod
    def _cancel_if_requested(is_cancelled: Callable[[], bool]) -> None:
        if is_cancelled():
            raise SplatImportCancelledError
