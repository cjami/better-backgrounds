"""Direct Gaussian PLY validation and publication tests."""

from __future__ import annotations

import json
import math
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from better_backgrounds.jobs.events import JobEvent, ResultEvent
from better_backgrounds.reconstruction import (
    SplatImportCancelledError,
    SplatImportConfig,
    SplatImportRequest,
    SplatSceneImporter,
    SplatSelection,
    inspect_gaussian_ply,
    inspect_gaussian_scene,
)
from better_backgrounds.reconstruction.splat_worker import SplatImportWorker
from better_backgrounds.scene import SceneCatalogue

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

STANDARD_NAMES = (
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
)


def write_standard_ply(path: Path, *, invalid_property: str | None = None) -> None:
    """Write a small renderer-compatible standard 3DGS PLY."""
    vertices = np.zeros(4, dtype=[(name, "f4") for name in STANDARD_NAMES])
    vertices["x"] = [-1, 1, -1, 1]
    vertices["y"] = [-1, -1, 1, 1]
    vertices["z"] = [2, 2, 3, 3]
    vertices["rot_0"] = 1
    if invalid_property is not None:
        vertices[invalid_property][0] = np.nan
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(path)


def write_sharp_ply(path: Path) -> None:
    """Write a minimal SHARP-layout PLY with embedded camera metadata."""
    vertices = np.zeros(4, dtype=[(name, "f4") for name in STANDARD_NAMES])
    vertices["x"] = [-1, 1, -1, 1]
    vertices["y"] = [-1, -1, 1, 1]
    vertices["z"] = 2
    vertices["rot_0"] = 1
    intrinsics = np.array(
        [(300,), (0,), (160,), (0,), (300,), (128,), (0,), (0,), (1,)],
        dtype=[("intrinsic", "f4")],
    )
    image_size = np.array([(320,), (256,)], dtype=[("image_size", "u4")])
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(intrinsics, "intrinsic"),
            PlyElement.describe(image_size, "image_size"),
        ],
        byte_order="<",
    ).write(path)


def write_compressed_ply(path: Path) -> None:
    """Write the bounded PlayCanvas compressed PLY layout."""
    chunk_names = (
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
    chunks = np.zeros(1, dtype=[(name, "f4") for name in chunk_names])
    chunks["min_x"] = chunks["min_y"] = -1
    chunks["min_z"] = 2
    chunks["max_x"] = chunks["max_y"] = 1
    chunks["max_z"] = 3
    vertices = np.zeros(
        4,
        dtype=[
            ("packed_position", "u4"),
            ("packed_rotation", "u4"),
            ("packed_scale", "u4"),
            ("packed_color", "u4"),
        ],
    )
    PlyData(
        [
            PlyElement.describe(chunks, "chunk"),
            PlyElement.describe(vertices, "vertex"),
        ],
        byte_order="<",
    ).write(path)


def _webp(width: int = 2, height: int = 2) -> bytes:
    output = BytesIO()
    Image.new("RGBA", (width, height), (128, 128, 128, 255)).save(
        output,
        format="WEBP",
        lossless=True,
    )
    return output.getvalue()


def _webp_pixels(pixels: list[tuple[int, int, int, int]]) -> bytes:
    output = BytesIO()
    image = Image.new("RGBA", (2, 2))
    image.putdata(pixels)
    image.save(output, format="WEBP", lossless=True)
    return output.getvalue()


def _sog_metadata(count: int, *, position_limit: float = 1.0) -> dict[str, object]:
    codebook = [0.0] * 256
    return {
        "version": 2,
        "count": count,
        "means": {
            "mins": [-math.log1p(position_limit)] * 3,
            "maxs": [math.log1p(position_limit)] * 3,
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {"codebook": codebook, "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": codebook, "files": ["sh0.webp"]},
    }


def write_streamed_sog(  # noqa: C901, PLR0912, PLR0913
    path: Path,
    *,
    prefix: str = "",
    omit: str | None = None,
    traversal: bool = False,
    wrong_ranges: bool = False,
    include_environment: bool = False,
    oversized: bool = False,
    manifest_version: int | None = 1,
    loose_position_bounds: bool = False,
) -> None:
    """Write a small packaged two-level Streamed SOG fixture."""
    root = f"{prefix.rstrip('/')}/" if prefix else ""
    manifest = {
        "version": 1,
        "count": 6,
        "counts": [4, 2],
        "lodLevels": 2,
        "filenames": ["0_0/meta.json", "1_0/meta.json"],
        "tree": {
            "bound": {"min": [-4.0, -2.0, -6.0], "max": [4.0, 3.0, 2.0]},
            "lods": {
                "0": {"file": 0, "offset": 1 if wrong_ranges else 0, "count": 4},
                "1": {"file": 1, "offset": 0, "count": 2},
            },
        },
    }
    if oversized:
        manifest["count"] = 100_000_003
        manifest["counts"] = [100_000_001, 2]
        manifest["tree"]["lods"]["0"]["count"] = 100_000_001
    if manifest_version is None:
        manifest.pop("version")
        manifest.pop("count")
        manifest.pop("counts")
    else:
        manifest["version"] = manifest_version
    if include_environment:
        manifest["environment"] = "env/meta.json"
    members: dict[str, bytes] = {
        f"{root}lod-meta.json": json.dumps(manifest).encode(),
    }
    image = _webp()
    low_means = _webp_pixels([(0, 0, 0, 255)] * 4)
    high_means = _webp_pixels(
        [
            (120, 121, 122, 255),
            (124, 125, 126, 255),
            (132, 133, 134, 255),
            (136, 137, 138, 255),
        ],
    )
    for directory, count in (("0_0", 4), ("1_0", 2)):
        metadata = _sog_metadata(count, position_limit=75.0 if loose_position_bounds else 1.0)
        members[f"{root}{directory}/meta.json"] = json.dumps(metadata).encode()
        for name in ("means_l.webp", "means_u.webp", "scales.webp", "quats.webp", "sh0.webp"):
            payload = image
            if loose_position_bounds and name == "means_l.webp":
                payload = low_means
            elif loose_position_bounds and name == "means_u.webp":
                payload = high_means
            members[f"{root}{directory}/{name}"] = payload
    if include_environment:
        members[f"{root}env/meta.json"] = json.dumps(_sog_metadata(1)).encode()
        for name in ("means_l.webp", "means_u.webp", "scales.webp", "quats.webp", "sh0.webp"):
            members[f"{root}env/{name}"] = image
    if omit is not None:
        members.pop(f"{root}{omit}")
    if traversal:
        members["../escaped.webp"] = image
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize(
    ("writer", "layout", "framing"),
    [
        (write_standard_ply, "standard", "Automatic COLMAP framing"),
        (write_sharp_ply, "sharp", "Embedded SHARP camera"),
        (write_compressed_ply, "compressed", "Automatic COLMAP framing"),
    ],
)
def test_inspector_accepts_supported_gaussian_ply_layouts(
    tmp_path: Path,
    writer: Callable[[Path], None],
    layout: str,
    framing: str,
) -> None:
    """Accept every PLY layout the embedded PlayCanvas parser can render."""
    path = tmp_path / "scene.ply"
    writer(path)

    diagnostics = inspect_gaussian_ply(path)

    assert diagnostics.gaussian_count == 4
    assert diagnostics.layout == layout
    assert diagnostics.framing == framing


def test_inspector_rejects_unsafe_or_unsupported_ply_data(tmp_path: Path) -> None:
    """Reject text, poisoned, truncated, and over-limit inputs before adoption."""
    text_path = tmp_path / "text.ply"
    text_path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="ascii")
    with pytest.raises(ValueError, match="binary little-endian"):
        inspect_gaussian_ply(text_path)

    poisoned = tmp_path / "poisoned.ply"
    write_standard_ply(poisoned, invalid_property="x")
    with pytest.raises(ValueError, match="non-finite x"):
        inspect_gaussian_ply(poisoned)

    truncated = tmp_path / "truncated.ply"
    write_standard_ply(truncated)
    truncated.write_bytes(truncated.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bounds"):
        inspect_gaussian_ply(truncated)

    oversized = tmp_path / "oversized.ply"
    oversized.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 5000001\nend_header\n"
    )
    with pytest.raises(ValueError, match="Gaussian count"):
        inspect_gaussian_ply(oversized)


def test_inspector_accepts_packaged_streamed_sog_with_nested_root(tmp_path: Path) -> None:
    """Accept the official lod-meta plus unbundled SOG chunk layout."""
    path = tmp_path / "environment.ssog"
    write_streamed_sog(path, prefix="exported-scene", include_environment=True)

    diagnostics = inspect_gaussian_scene(path)

    assert diagnostics.layout == "streamed-sog"
    assert diagnostics.encoding == "Streamed SOG"
    assert diagnostics.gaussian_count == 5
    assert diagnostics.total_gaussian_count == 7
    assert diagnostics.lod_levels == 2
    assert diagnostics.resource_count == 19
    assert diagnostics.bounds_minimum == (-1.0, -1.0, -1.0)
    assert diagnostics.bounds_maximum == (1.0, 1.0, 1.0)


def test_inspector_accepts_pre_release_streamed_sog_metadata(tmp_path: Path) -> None:
    """Derive counts for compatible exports that predate the version and count fields."""
    path = tmp_path / "legacy-export.zip"
    write_streamed_sog(path, manifest_version=None)

    diagnostics = inspect_gaussian_scene(path)

    assert diagnostics.gaussian_count == 4
    assert diagnostics.total_gaussian_count == 6
    assert diagnostics.lod_levels == 2


def test_inspector_uses_sampled_positions_instead_of_loose_chunk_bounds(tmp_path: Path) -> None:
    """Keep framing stable when quantization metadata contains distant outliers."""
    path = tmp_path / "loose-bounds.ssog"
    write_streamed_sog(path, loose_position_bounds=True)

    diagnostics = inspect_gaussian_scene(path)

    assert max(abs(value) for value in diagnostics.bounds_minimum) < 10
    assert max(abs(value) for value in diagnostics.bounds_maximum) < 10
    assert diagnostics.navigation_bounds_minimum is not None
    assert diagnostics.navigation_bounds_maximum is not None
    assert max(abs(value) for value in diagnostics.navigation_bounds_minimum) < 10
    assert max(abs(value) for value in diagnostics.navigation_bounds_maximum) < 10


def test_inspector_rejects_unknown_explicit_streamed_sog_version(tmp_path: Path) -> None:
    """Do not silently interpret future explicit SSOG schema versions."""
    path = tmp_path / "future-export.zip"
    write_streamed_sog(path, manifest_version=99)

    with pytest.raises(ValueError, match="unsupported format version"):
        inspect_gaussian_scene(path)


@pytest.mark.parametrize(
    ("omit", "traversal", "wrong_ranges", "message"),
    [
        ("0_0/means_l.webp", False, False, "missing image resource"),
        (None, True, False, "safe relative path"),
        (None, False, True, "overlap or leave gaps"),
    ],
)
def test_inspector_rejects_unsafe_or_incomplete_streamed_sog(
    tmp_path: Path,
    omit: str | None,
    traversal: int,
    wrong_ranges: int,
    message: str,
) -> None:
    """Reject unsafe paths, missing payloads, and corrupt chunk ranges."""
    path = tmp_path / "broken.zip"
    write_streamed_sog(
        path,
        omit=omit,
        traversal=bool(traversal),
        wrong_ranges=bool(wrong_ranges),
    )

    with pytest.raises(ValueError, match=message):
        inspect_gaussian_scene(path)


def test_inspector_bounds_streamed_sog_full_detail_count(tmp_path: Path) -> None:
    """Bound even streamed datasets before reading their image payloads."""
    path = tmp_path / "oversized.ssog"
    write_streamed_sog(path, oversized=True)

    with pytest.raises(ValueError, match="hundred-million-Gaussian"):
        inspect_gaussian_scene(path)


def test_importer_adopts_a_deterministic_scene_without_touching_source(
    tmp_path: Path,
) -> None:
    """Copy one validated PLY into the managed cache with automatic framing."""
    source = tmp_path / "my_room.ply"
    write_standard_ply(source)
    original = source.read_bytes()
    scene_root = tmp_path / "scenes"
    request = SplatImportRequest(
        job_id="import-1",
        selection=SplatSelection(source.name, source),
        config=SplatImportConfig(scene_root),
    )

    importer = SplatSceneImporter()
    first = importer.import_scene(request, lambda _event: None, lambda: False)
    second = importer.import_scene(request, lambda _event: None, lambda: False)

    assert first.asset_id == second.asset_id
    assert first.display_name == "My Room"
    assert first.preview is None
    assert first.default_viewpoint.depth_of_field.blur_strength == 0
    assert first.default_viewpoint.scene_transform.orientation.z == 1
    assert (scene_root / first.asset_id / "scene.ply").read_bytes() == original
    assert source.read_bytes() == original
    assert not list(scene_root.glob(".splat-*.part"))


def test_importer_adopts_streamed_sog_resources_without_touching_source(
    tmp_path: Path,
) -> None:
    """Extract an SSOG archive into a self-contained managed streaming scene."""
    source = tmp_path / "whole_room.ssog"
    write_streamed_sog(source, prefix="download", include_environment=True)
    original = source.read_bytes()
    scene_root = tmp_path / "scenes"
    request = SplatImportRequest(
        job_id="ssog-import",
        selection=SplatSelection(source.name, source),
        config=SplatImportConfig(scene_root),
    )

    first = SplatSceneImporter().import_scene(request, lambda _event: None, lambda: False)
    second = SplatSceneImporter().import_scene(request, lambda _event: None, lambda: False)

    assert first.asset_id == second.asset_id
    assert first.format == "ssog"
    assert first.entrypoint == "lod-meta.json"
    assert first.preview is None
    assert first.default_viewpoint.depth_of_field.blur_strength == 0
    assert first.default_viewpoint.scene_transform.orientation.z == 1
    assert first.default_viewpoint.scene_transform.scale == 1
    assert first.default_viewpoint.position.x == pytest.approx(0.65)
    assert first.default_viewpoint.position.y == pytest.approx(0.5)
    assert first.default_viewpoint.position.z == pytest.approx(1.06)
    assert first.default_viewpoint.field_of_view == 80
    assert first.default_viewpoint.safe_camera_region.minimum.x == pytest.approx(-1.5)
    assert first.default_viewpoint.safe_camera_region.maximum.z == pytest.approx(1.5)
    assert len(first.resources) == 19
    assert (scene_root / first.asset_id / "0_0" / "meta.json").is_file()
    assert (scene_root / first.asset_id / "env" / "meta.json").is_file()
    assert source.read_bytes() == original
    assert not list(scene_root.glob(".splat-*.part"))


def test_cancelled_import_removes_staging_and_preserves_source(tmp_path: Path) -> None:
    """Leave no managed partial scene after cooperative cancellation."""
    source = tmp_path / "room.ply"
    write_standard_ply(source)
    scene_root = tmp_path / "scenes"
    request = SplatImportRequest(
        job_id="cancelled",
        selection=SplatSelection(source.name, source),
        config=SplatImportConfig(scene_root),
    )

    with pytest.raises(SplatImportCancelledError):
        SplatSceneImporter().import_scene(request, lambda _event: None, lambda: True)

    assert source.is_file()
    assert not list(scene_root.glob(".splat-*.part"))


def test_worker_catalogues_the_imported_scene(tmp_path: Path) -> None:
    """Publish the imported reference before reporting its terminal result."""
    source = tmp_path / "catalogued.ply"
    write_standard_ply(source)
    scene_root = tmp_path / "scenes"
    catalogue_path = tmp_path / "catalogue.json"
    events: list[JobEvent] = []

    result = SplatImportWorker(
        job_id="worker-import",
        source=source,
        scene_cache_root=scene_root,
        catalogue_path=catalogue_path,
        emit=events.append,
    ).run()

    assert result == 0
    reference = SceneCatalogue(catalogue_path).scenes()[0]
    assert reference.display_name == "Catalogued"
    assert isinstance(events[-1], ResultEvent)
    assert reference.asset_id == events[-1].scene_id
