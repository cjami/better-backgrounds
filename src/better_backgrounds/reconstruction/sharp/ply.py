"""Strict validation for SHARP's binary Gaussian PLY output."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from plyfile import PlyData

from better_backgrounds.reconstruction.sharp.contracts import SharpPlyMetadata

if TYPE_CHECKING:
    from pathlib import Path

MAX_IMAGE_DIMENSION = 16_384
MAX_GAUSSIANS = 5_000_000
PLY_HEADER_PARTS = 3
PLY_INTRINSIC_VALUE_COUNT = 9
PLY_IMAGE_SIZE_VALUE_COUNT = 2
MIN_ROTATION_NORM = 1e-6


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
