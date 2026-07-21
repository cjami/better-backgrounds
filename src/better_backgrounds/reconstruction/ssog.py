"""Validation and extraction for packaged Streamed SOG scene datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_ARCHIVE_FILES = 16_384
MAX_ARCHIVE_SIZE = 4 * 1024**3
MAX_MEMBER_SIZE = 1024**3
MAX_JSON_SIZE = 16 * 1024**2
MAX_LOD_LEVELS = 32
MAX_TREE_DEPTH = 64
MAX_TREE_NODES = 1_000_000
COPY_CHUNK_SIZE = 1024 * 1024
MAX_PATH_LENGTH = 500
CONTROL_CHARACTER_LIMIT = 32
VECTOR_DIMENSIONS = 3
BINARY_TREE_CHILDREN = 2
CODEBOOK_SIZE = 256
SOG_VERSION = 2
MAX_LOG_POSITION = 80
MAX_SH_BANDS = 3
MAX_SH_PALETTE = 65_536
MAX_EXP_INPUT = 709.0
MAX_IMAGE_PIXELS = 16_000_000
MAX_STREAMED_GAUSSIANS = 100_000_000
MAX_POSITION_SAMPLES_PER_CHUNK = 4_096
ROBUST_BOUND_QUANTILE = 0.001
MIN_ROBUST_EXTENT = 1e-6
SOG_COMPONENT_FILES = {"means": 2, "scales": 1, "quats": 1, "sh0": 1}


@dataclass(frozen=True, slots=True)
class StreamedSogResource:
    """Map one managed relative path to its member in the source archive."""

    path: str
    archive_name: str


@dataclass(frozen=True, slots=True)
class StreamedSogInspection:
    """Describe a safe, renderer-compatible Streamed SOG archive."""

    gaussian_count: int
    total_gaussian_count: int
    lod_levels: int
    file_size: int
    bounds_minimum: tuple[float, float, float]
    bounds_maximum: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    navigation_bounds_minimum: tuple[float, float, float]
    navigation_bounds_maximum: tuple[float, float, float]
    resources: tuple[StreamedSogResource, ...]


@dataclass(frozen=True, slots=True)
class ExtractedSogResource:
    """Record the integrity metadata of one extracted managed resource."""

    path: str
    size: int
    sha256: str


@dataclass(slots=True)
class _TreeValidation:
    counts: list[int]
    file_ranges: dict[int, list[tuple[int, int]]]
    level_files: dict[int, set[int]]
    nodes: int = 0


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Streamed SOG metadata contains duplicate {key!r} fields"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    msg = f"Streamed SOG metadata contains non-finite {value}"
    raise ValueError(msg)


def _read_json(archive: ZipFile, info: ZipInfo, description: str) -> dict[str, Any]:
    if info.file_size > MAX_JSON_SIZE:
        msg = f"{description} is too large"
        raise ValueError(msg)
    try:
        payload = archive.read(info)
        value = json.loads(
            payload,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"{description} is not valid UTF-8 JSON"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"{description} must contain a JSON object"
        raise ValueError(msg)  # noqa: TRY004
    return value


def _safe_path(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PATH_LENGTH or "\\" in value:
        msg = f"{description} must be a safe relative path"
        raise ValueError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            ":" in part or any(ord(character) < CONTROL_CHARACTER_LIMIT for character in part)
            for part in path.parts
        )
    ):
        msg = f"{description} must be a safe relative path"
        raise ValueError(msg)
    return path.as_posix()


def _archive_members(archive: ZipFile) -> dict[str, ZipInfo]:
    files = [info for info in archive.infolist() if not info.is_dir()]
    if not files or len(files) > MAX_ARCHIVE_FILES:
        msg = "Streamed SOG archive contains an invalid number of files"
        raise ValueError(msg)
    total_size = 0
    members: dict[str, ZipInfo] = {}
    casefolded: set[str] = set()
    for info in files:
        name = _safe_path(info.filename, "Archive member")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or info.flag_bits & 1:
            msg = "Streamed SOG archives cannot contain links or encrypted files"
            raise ValueError(msg)
        if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
            msg = "Streamed SOG archive uses an unsupported compression method"
            raise ValueError(msg)
        if info.file_size > MAX_MEMBER_SIZE:
            msg = f"Streamed SOG member {name!r} is too large"
            raise ValueError(msg)
        total_size += info.file_size
        if total_size > MAX_ARCHIVE_SIZE:
            msg = "Streamed SOG archive expands beyond the managed import limit"
            raise ValueError(msg)
        folded = name.casefold()
        if name in members or folded in casefolded:
            msg = "Streamed SOG archive contains duplicate resource paths"
            raise ValueError(msg)
        members[name] = info
        casefolded.add(folded)
    return members


def _integer(value: object, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{description} must be an integer of at least {minimum}"
        raise ValueError(msg)
    return value


def _finite_vector(value: object, description: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != VECTOR_DIMENSIONS
        or any(isinstance(item, bool) or not isinstance(item, int | float) for item in value)
    ):
        msg = f"{description} must contain three finite numbers"
        raise ValueError(msg)
    items = cast("list[int | float]", value)
    try:
        result = tuple(float(items[index]) for index in range(VECTOR_DIMENSIONS))
    except OverflowError as error:
        msg = f"{description} must contain three finite numbers"
        raise ValueError(msg) from error
    if not all(math.isfinite(item) for item in result):
        msg = f"{description} must contain three finite numbers"
        raise ValueError(msg)
    return result[0], result[1], result[2]


def _bounds(
    value: object,
    description: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not isinstance(value, dict):
        msg = f"{description} must contain min and max bounds"
        raise ValueError(msg)  # noqa: TRY004
    minimum = _finite_vector(value.get("min"), f"{description} minimum")
    maximum = _finite_vector(value.get("max"), f"{description} maximum")
    if any(low > high for low, high in zip(minimum, maximum, strict=True)):
        msg = f"{description} contains inverted bounds"
        raise ValueError(msg)
    return minimum, maximum


def _validate_tree(  # noqa: C901
    node: object,
    *,
    depth: int,
    lod_levels: int,
    file_count: int,
    validation: _TreeValidation,
) -> None:
    if not isinstance(node, dict) or depth > MAX_TREE_DEPTH:
        msg = "Streamed SOG contains an invalid or excessively deep spatial tree"
        raise ValueError(msg)
    validation.nodes += 1
    if validation.nodes > MAX_TREE_NODES:
        msg = "Streamed SOG spatial tree exceeds the node limit"
        raise ValueError(msg)
    _bounds(node.get("bound"), "Spatial tree node")
    children = node.get("children")
    lods = node.get("lods")
    if children is not None:
        if (
            lods is not None
            or not isinstance(children, list)
            or len(children) != BINARY_TREE_CHILDREN
        ):
            msg = "Streamed SOG interior nodes must contain exactly two children"
            raise ValueError(msg)
        for child in children:
            _validate_tree(
                child,
                depth=depth + 1,
                lod_levels=lod_levels,
                file_count=file_count,
                validation=validation,
            )
        return
    if not isinstance(lods, dict) or not lods:
        msg = "Streamed SOG leaf nodes must contain LOD ranges"
        raise ValueError(msg)
    for raw_level, raw_range in lods.items():
        if not isinstance(raw_level, str) or not raw_level.isdecimal():
            msg = "Streamed SOG leaf contains an invalid LOD level"
            raise ValueError(msg)
        level = int(raw_level)
        if level >= lod_levels or str(level) != raw_level or not isinstance(raw_range, dict):
            msg = "Streamed SOG leaf contains an out-of-range LOD level"
            raise ValueError(msg)
        file_index = _integer(raw_range.get("file"), "LOD file index")
        offset = _integer(raw_range.get("offset", 0), "LOD range offset")
        count = _integer(raw_range.get("count"), "LOD range count", minimum=1)
        if file_index >= file_count:
            msg = "Streamed SOG leaf references an unknown chunk"
            raise ValueError(msg)
        validation.counts[level] += count
        validation.file_ranges.setdefault(file_index, []).append((offset, offset + count))
        validation.level_files.setdefault(level, set()).add(file_index)


def _component_files(meta: dict[str, Any], component: str, expected: int) -> list[str]:
    value = meta.get(component)
    if not isinstance(value, dict):
        msg = f"SOG metadata is missing its {component} component"
        raise ValueError(msg)  # noqa: TRY004
    files = value.get("files")
    if not isinstance(files, list) or len(files) != expected:
        msg = f"SOG {component} component has an invalid file list"
        raise ValueError(msg)
    return [_safe_path(item, f"SOG {component} resource") for item in files]


def _codebook(meta: dict[str, Any], component: str) -> None:
    value = meta[component].get("codebook")
    if not isinstance(value, list) or len(value) != CODEBOOK_SIZE:
        msg = f"SOG {component} component has an invalid codebook"
        raise ValueError(msg)
    try:
        numbers = [float(item) for item in value if not isinstance(item, bool)]
    except OverflowError as error:
        msg = f"SOG {component} component has an invalid codebook"
        raise ValueError(msg) from error
    if len(numbers) != CODEBOOK_SIZE or any(not math.isfinite(item) for item in numbers):
        msg = f"SOG {component} component has an invalid codebook"
        raise ValueError(msg)
    if component == "scales" and any(item > MAX_EXP_INPUT for item in numbers):
        msg = "SOG scales contain unusable values"
        raise ValueError(msg)


def _resolve_resource(meta_path: str, value: str) -> str:
    return (PurePosixPath(meta_path).parent / PurePosixPath(value)).as_posix()


def _verify_image(
    archive: ZipFile,
    info: ZipInfo,
    description: str,
    *,
    validate_payload: bool,
) -> tuple[int, int]:
    try:
        with archive.open(info) as source, Image.open(source) as image:
            size = image.size
            image_format = image.format
            if validate_payload:
                image.verify()
    except (OSError, UnidentifiedImageError) as error:
        msg = f"{description} is not a valid lossless splat image"
        raise ValueError(msg) from error
    if (
        image_format not in {"WEBP", "PNG"}
        or size[0] <= 0
        or size[1] <= 0
        or size[0] * size[1] > MAX_IMAGE_PIXELS
    ):
        msg = f"{description} uses an unsupported image encoding"
        raise ValueError(msg)
    return size


def _sample_positions(
    archive: ZipFile,
    members: dict[str, ZipInfo],
    meta_path: str,
    means: dict[str, Any],
    count: int,
) -> tuple[tuple[float, float, float], ...]:
    files = _component_files({"means": means}, "means", 2)
    images: list[Image.Image] = []
    try:
        for name in files:
            path = _resolve_resource(meta_path, name)
            with archive.open(members[path]) as source, Image.open(source) as image:
                images.append(image.convert("RGBA"))
        low, high = images
        width = low.width
        stride = max(1, math.ceil(count / MAX_POSITION_SAMPLES_PER_CHUNK))
        indices = list(range(0, count, stride))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        minimum = _finite_vector(means.get("mins"), "SOG position minimum")
        maximum = _finite_vector(means.get("maxs"), "SOG position maximum")

        def decode(
            axis: int,
            low_value: tuple[int, int, int, int],
            high_value: tuple[int, int, int, int],
        ) -> float:
            normalized = ((high_value[axis] << 8) + low_value[axis]) / 65_535
            logarithmic = minimum[axis] + (maximum[axis] - minimum[axis]) * normalized
            return math.copysign(math.expm1(abs(logarithmic)), logarithmic)

        samples: list[tuple[float, float, float]] = []
        for index in indices:
            coordinate = index % width, index // width
            low_value = cast("tuple[int, int, int, int]", low.getpixel(coordinate))
            high_value = cast("tuple[int, int, int, int]", high.getpixel(coordinate))
            samples.append(
                (
                    decode(0, low_value, high_value),
                    decode(1, low_value, high_value),
                    decode(2, low_value, high_value),
                ),
            )
        return tuple(samples)
    finally:
        for image in images:
            image.close()


def _robust_sample_bounds(
    samples: list[tuple[float, float, float]],
    *,
    quantile: float = ROBUST_BOUND_QUANTILE,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not samples:
        return None
    minimum: list[float] = []
    maximum: list[float] = []
    for axis in range(VECTOR_DIMENSIONS):
        values = sorted(sample[axis] for sample in samples)
        lower = math.floor((len(values) - 1) * quantile)
        upper = math.ceil((len(values) - 1) * (1 - quantile))
        minimum.append(values[lower])
        maximum.append(values[upper])
    if any(high - low <= MIN_ROBUST_EXTENT for low, high in zip(minimum, maximum, strict=True)):
        return None
    return (
        cast("tuple[float, float, float]", tuple(minimum)),
        cast("tuple[float, float, float]", tuple(maximum)),
    )


def _validate_sog(  # noqa: C901, PLR0912, PLR0915
    archive: ZipFile,
    members: dict[str, ZipInfo],
    meta_path: str,
    *,
    validate_payloads: bool,
    sample_positions: bool = False,
) -> tuple[
    int,
    tuple[str, ...],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[tuple[float, float, float], ...],
]:
    info = members.get(meta_path)
    if info is None:
        msg = f"Streamed SOG is missing chunk metadata {meta_path!r}"
        raise ValueError(msg)
    meta = _read_json(archive, info, f"SOG metadata {meta_path!r}")
    if meta.get("version") != SOG_VERSION:
        msg = "Streamed SOG chunks must use supported SOG version 2"
        raise ValueError(msg)
    count = _integer(meta.get("count"), "SOG Gaussian count", minimum=1)
    means = meta.get("means")
    if not isinstance(means, dict):
        msg = "SOG metadata is missing position bounds"
        raise ValueError(msg)  # noqa: TRY004
    logarithmic_minimum = _finite_vector(means.get("mins"), "SOG position minimum")
    logarithmic_maximum = _finite_vector(means.get("maxs"), "SOG position maximum")
    if any(
        low > high for low, high in zip(logarithmic_minimum, logarithmic_maximum, strict=True)
    ) or any(abs(item) > MAX_LOG_POSITION for item in logarithmic_minimum + logarithmic_maximum):
        msg = "SOG metadata contains unusable position bounds"
        raise ValueError(msg)
    minimum = cast(
        "tuple[float, float, float]",
        tuple(math.copysign(math.expm1(abs(value)), value) for value in logarithmic_minimum),
    )
    maximum = cast(
        "tuple[float, float, float]",
        tuple(math.copysign(math.expm1(abs(value)), value) for value in logarithmic_maximum),
    )

    resource_names: list[str] = []
    per_gaussian: list[str] = []
    for component, expected in SOG_COMPONENT_FILES.items():
        files = _component_files(meta, component, expected)
        resource_names.extend(files)
        per_gaussian.extend(files)
    _codebook(meta, "scales")
    _codebook(meta, "sh0")
    sh_n = meta.get("shN")
    centroid_name: str | None = None
    sh_bands: int | None = None
    sh_palette_count: int | None = None
    if sh_n is not None:
        files = _component_files(meta, "shN", 2)
        centroid_name = files[0]
        resource_names.extend(files)
        per_gaussian.append(files[1])
        bands = _integer(sh_n.get("bands"), "SOG spherical-harmonic bands", minimum=1)
        palette_count = _integer(
            sh_n.get("count"),
            "SOG spherical-harmonic palette count",
            minimum=1,
        )
        if bands > MAX_SH_BANDS or palette_count > MAX_SH_PALETTE:
            msg = "SOG spherical-harmonic metadata exceeds supported bounds"
            raise ValueError(msg)
        _codebook(meta, "shN")
        sh_bands = bands
        sh_palette_count = palette_count

    resolved = tuple(_resolve_resource(meta_path, name) for name in resource_names)
    if len(resolved) != len(set(resolved)):
        msg = "SOG metadata contains duplicate image resources"
        raise ValueError(msg)
    expected_size: tuple[int, int] | None = None
    per_gaussian_set = {_resolve_resource(meta_path, name) for name in per_gaussian}
    for path in resolved:
        image_info = members.get(path)
        if image_info is None:
            msg = f"Streamed SOG is missing image resource {path!r}"
            raise ValueError(msg)
        size = _verify_image(
            archive,
            image_info,
            f"SOG image {path!r}",
            validate_payload=validate_payloads,
        )
        if path in per_gaussian_set:
            expected_size = expected_size or size
            if size != expected_size or count > size[0] * size[1]:
                msg = "SOG per-Gaussian images have inconsistent dimensions"
                raise ValueError(msg)
        elif centroid_name is not None and sh_bands is not None and sh_palette_count is not None:
            coefficients = (3, 8, 15)[sh_bands - 1]
            if size[0] != 64 * coefficients or sh_palette_count > 64 * size[1]:
                msg = "SOG spherical-harmonic palette has invalid dimensions"
                raise ValueError(msg)
    samples = (
        _sample_positions(archive, members, meta_path, means, count) if sample_positions else ()
    )
    return count, (meta_path, *resolved), minimum, maximum, samples


def _root_relative(archive_name: str, root: PurePosixPath) -> str:
    return PurePosixPath(archive_name).relative_to(root).as_posix() if root.parts else archive_name


def inspect_streamed_sog(  # noqa: C901, PLR0912, PLR0915
    path: Path,
    *,
    validate_payloads: bool = True,
) -> StreamedSogInspection:
    """Validate a packaged Streamed SOG dataset without extracting it."""
    if path.suffix.lower() not in {".ssog", ".zip"}:
        msg = "Choose a Streamed SOG .ssog or .zip archive"
        raise ValueError(msg)
    if not path.is_file():
        msg = "The selected Streamed SOG archive does not exist"
        raise ValueError(msg)
    try:
        with ZipFile(path) as archive:
            members = _archive_members(archive)
            manifests = [name for name in members if PurePosixPath(name).name == "lod-meta.json"]
            if len(manifests) != 1:
                msg = "Streamed SOG archive must contain exactly one lod-meta.json"
                raise ValueError(msg)
            manifest_name = manifests[0]
            root = PurePosixPath(manifest_name).parent
            manifest = _read_json(archive, members[manifest_name], "Streamed SOG lod-meta.json")
            manifest_version = manifest.get("version")
            if manifest_version not in {None, 1}:
                msg = "Streamed SOG archive uses an unsupported format version"
                raise ValueError(msg)
            lod_levels = _integer(
                manifest.get("lodLevels"),
                "Streamed SOG LOD level count",
                minimum=1,
            )
            if lod_levels > MAX_LOD_LEVELS:
                msg = "Streamed SOG archive contains too many LOD levels"
                raise ValueError(msg)
            declared_counts: list[int] | None = None
            declared_total: int | None = None
            if manifest_version == 1:
                raw_counts = manifest.get("counts")
                if not isinstance(raw_counts, list) or len(raw_counts) != lod_levels:
                    msg = "Streamed SOG archive has an invalid per-LOD count list"
                    raise ValueError(msg)
                declared_counts = [
                    _integer(item, "Per-LOD Gaussian count", minimum=1) for item in raw_counts
                ]
                declared_total = _integer(
                    manifest.get("count"),
                    "Streamed SOG total Gaussian count",
                    minimum=1,
                )
                if declared_total != sum(declared_counts):
                    msg = "Streamed SOG total Gaussian count does not match its LOD counts"
                    raise ValueError(msg)
            filenames = manifest.get("filenames")
            if not isinstance(filenames, list) or not filenames:
                msg = "Streamed SOG archive does not list any chunk files"
                raise ValueError(msg)
            chunk_paths = [_safe_path(item, "Streamed SOG chunk path") for item in filenames]
            if len(chunk_paths) != len(set(chunk_paths)) or any(
                PurePosixPath(item).name != "meta.json" for item in chunk_paths
            ):
                msg = "Streamed SOG chunk paths must uniquely name meta.json files"
                raise ValueError(msg)
            chunk_archive_paths = [(root / PurePosixPath(item)).as_posix() for item in chunk_paths]
            validation = _TreeValidation(
                counts=[0] * lod_levels,
                file_ranges={},
                level_files={},
            )
            _validate_tree(
                manifest.get("tree"),
                depth=0,
                lod_levels=lod_levels,
                file_count=len(chunk_paths),
                validation=validation,
            )
            if set(validation.file_ranges) != set(range(len(chunk_paths))):
                msg = "Streamed SOG tree does not reference every declared chunk"
                raise ValueError(msg)
            if declared_counts is not None and validation.counts != declared_counts:
                msg = "Streamed SOG tree ranges do not match its declared LOD counts"
                raise ValueError(msg)
            counts = validation.counts
            if any(count <= 0 for count in counts):
                msg = "Streamed SOG tree does not contain every declared LOD level"
                raise ValueError(msg)
            if counts[0] > MAX_STREAMED_GAUSSIANS:
                msg = "Streamed SOG scene exceeds the hundred-million-Gaussian streaming limit"
                raise ValueError(msg)
            total_count = declared_total if declared_total is not None else sum(counts)

            resource_paths: list[str] = [manifest_name]
            visible_count = counts[0]
            full_detail_minimum = [math.inf] * VECTOR_DIMENSIONS
            full_detail_maximum = [-math.inf] * VECTOR_DIMENSIONS
            full_detail_samples: list[tuple[float, float, float]] = []
            full_detail_center_sum = [0.0] * VECTOR_DIMENSIONS
            full_detail_center_weight = 0
            for index, meta_path in enumerate(chunk_archive_paths):
                (
                    chunk_count,
                    chunk_resources,
                    chunk_minimum,
                    chunk_maximum,
                    chunk_samples,
                ) = _validate_sog(
                    archive,
                    members,
                    meta_path,
                    validate_payloads=validate_payloads,
                    sample_positions=index in validation.level_files[0],
                )
                ranges = sorted(validation.file_ranges[index])
                cursor = 0
                for start, end in ranges:
                    if start != cursor or end > chunk_count:
                        msg = "Streamed SOG chunk ranges overlap or leave gaps"
                        raise ValueError(msg)
                    cursor = end
                if cursor != chunk_count:
                    msg = "Streamed SOG chunk ranges do not cover their SOG data"
                    raise ValueError(msg)
                if index in validation.level_files[0]:
                    full_detail_samples.extend(chunk_samples)
                    full_detail_center_weight += chunk_count
                    for axis in range(VECTOR_DIMENSIONS):
                        full_detail_minimum[axis] = min(
                            full_detail_minimum[axis],
                            chunk_minimum[axis],
                        )
                        full_detail_maximum[axis] = max(
                            full_detail_maximum[axis],
                            chunk_maximum[axis],
                        )
                        full_detail_center_sum[axis] += (
                            sum(sample[axis] for sample in chunk_samples)
                            / len(chunk_samples)
                            * chunk_count
                        )
                resource_paths.extend(chunk_resources)

            environment = manifest.get("environment")
            if environment is not None:
                environment_path = _safe_path(environment, "Streamed SOG environment path")
                if PurePosixPath(environment_path).name != "meta.json":
                    msg = "Streamed SOG environment must name a SOG meta.json file"
                    raise ValueError(msg)
                (
                    environment_count,
                    environment_resources,
                    _minimum,
                    _maximum,
                    _samples,
                ) = _validate_sog(
                    archive,
                    members,
                    (root / PurePosixPath(environment_path)).as_posix(),
                    validate_payloads=validate_payloads,
                )
                visible_count += environment_count
                total_count += environment_count
                resource_paths.extend(environment_resources)
            if visible_count > MAX_STREAMED_GAUSSIANS:
                msg = "Streamed SOG scene exceeds the hundred-million-Gaussian streaming limit"
                raise ValueError(msg)
            unique_resources = tuple(dict.fromkeys(resource_paths))
            managed_resources = tuple(
                StreamedSogResource(
                    path=_root_relative(name, root),
                    archive_name=name,
                )
                for name in unique_resources
            )
            navigation_minimum = cast(
                "tuple[float, float, float]",
                tuple(full_detail_minimum),
            )
            navigation_maximum = cast(
                "tuple[float, float, float]",
                tuple(full_detail_maximum),
            )
            robust_bounds = _robust_sample_bounds(full_detail_samples)
            if robust_bounds is not None:
                minimum, maximum = robust_bounds
                navigation_minimum, navigation_maximum = robust_bounds
            else:
                minimum, maximum = navigation_minimum, navigation_maximum
            center_of_mass = cast(
                "tuple[float, float, float]",
                tuple(value / full_detail_center_weight for value in full_detail_center_sum),
            )
    except BadZipFile as error:
        msg = "The selected file is not a valid Streamed SOG ZIP archive"
        raise ValueError(msg) from error
    return StreamedSogInspection(
        gaussian_count=visible_count,
        total_gaussian_count=total_count,
        lod_levels=lod_levels,
        file_size=path.stat().st_size,
        bounds_minimum=minimum,
        bounds_maximum=maximum,
        center_of_mass=center_of_mass,
        navigation_bounds_minimum=navigation_minimum,
        navigation_bounds_maximum=navigation_maximum,
        resources=managed_resources,
    )


def extract_streamed_sog(
    path: Path,
    inspection: StreamedSogInspection,
    destination: Path,
    is_cancelled: Callable[[], bool],
) -> tuple[ExtractedSogResource, ...]:
    """Extract only inspected resources and calculate their managed hashes."""
    extracted: list[ExtractedSogResource] = []
    with ZipFile(path) as archive:
        for resource in inspection.resources:
            if is_cancelled():
                raise InterruptedError
            target = destination.joinpath(*PurePosixPath(resource.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(resource.archive_name) as source, target.open("xb") as output:
                while chunk := source.read(COPY_CHUNK_SIZE):
                    if is_cancelled():
                        raise InterruptedError
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            extracted.append(
                ExtractedSogResource(
                    path=resource.path,
                    size=size,
                    sha256=digest.hexdigest(),
                )
            )
    return tuple(extracted)


def remove_partial_streamed_sog(path: Path) -> None:
    """Remove an unpublished extraction tree."""
    shutil.rmtree(path, ignore_errors=True)
