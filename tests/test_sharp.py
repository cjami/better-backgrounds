"""Upload-first SHARP build, validation, and managed-checkpoint tests."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from better_backgrounds.protocol import CancelControl, ProgressEvent
from better_backgrounds.sharp import (
    SHARP_BUILDER_REVISION,
    SceneBuildRequest,
    SceneImageSelection,
    SharpBuildConfig,
    SharpCancelledError,
    SharpCheckpointInstaller,
    SharpCheckpointManifest,
    SharpSceneBuilder,
    inspect_scene_image,
    validate_sharp_ply,
)
from better_backgrounds.sharp_worker import watch_control

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from better_backgrounds.sharp_runtime import SharpDevice


def manifest(checkpoint: bytes) -> SharpCheckpointManifest:
    """Create a compact pinned checkpoint identity for boundary tests."""
    return SharpCheckpointManifest(
        model_id="test-sharp",
        filename="sharp.pt",
        url="https://example.com/sharp.pt",
        size=len(checkpoint),
        sha256=hashlib.sha256(checkpoint).hexdigest(),
        builder_revision=SHARP_BUILDER_REVISION,
        license_name="Research model license",
        license_url="https://example.com/model-license",
    )


def write_sharp_ply(
    path: Path,
    *,
    image_size: tuple[int, int] = (320, 256),
    invalid_property: str | None = None,
    focal_length_px: float = 300.0,
    text: bool = False,
) -> None:
    """Write a minimal official-shape binary SHARP PLY."""
    names = [
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
    ]
    vertices = np.zeros(2, dtype=[(name, "f4") for name in names])
    vertices["z"] = 2.0
    vertices["rot_0"] = 1.0
    if invalid_property is not None:
        vertices[invalid_property][0] = np.nan
    width, height = image_size
    image_values = np.array([(width,), (height,)], dtype=[("image_size", "u4")])
    intrinsics = np.array(
        [
            (focal_length_px,),
            (0,),
            (width / 2,),
            (0,),
            (focal_length_px,),
            (height / 2,),
            (0,),
            (0,),
            (1,),
        ],
        dtype=[("intrinsic", "f4")],
    )
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(intrinsics, "intrinsic"),
            PlyElement.describe(image_values, "image_size"),
        ],
        text=text,
        byte_order="<",
    ).write(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows named-pipe regression")
def test_sharp_control_listener_polls_fragmented_windows_pipe() -> None:
    """Avoid a blocking stdin read while the worker loads its large checkpoint."""
    read_fd, write_fd = os.pipe()
    cancelled = threading.Event()
    with (
        os.fdopen(read_fd, encoding="utf-8") as reader,
        os.fdopen(write_fd, "wb", buffering=0) as writer,
    ):
        listener = watch_control("pipe-job", cancelled.set, stdin=reader)
        payload = CancelControl(job_id="pipe-job").model_dump_json().encode()
        midpoint = len(payload) // 2
        writer.write(payload[:midpoint])
        assert not cancelled.wait(0.05)
        writer.write(payload[midpoint:] + b"\n")
        assert cancelled.wait(1.0)
        listener.join(1.0)
    assert not listener.is_alive()


def test_image_review_applies_exif_orientation_and_warns_for_missing_focal(
    tmp_path: Path,
) -> None:
    """Review the same oriented dimensions that official inference receives."""
    source = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (300, 400), "navy").save(source, exif=exif)

    diagnostics = inspect_scene_image(source)

    assert (diagnostics.width, diagnostics.height) == (400, 300)
    assert diagnostics.orientation_applied
    assert not diagnostics.has_focal_metadata
    assert "30 mm default" in diagnostics.warnings[0]


def test_image_review_accepts_png_alpha_with_explicit_warning(tmp_path: Path) -> None:
    """Keep PNG alpha deterministic instead of passing hidden RGB into SHARP."""
    source = tmp_path / "room.png"
    Image.new("RGBA", (320, 256), (20, 30, 40, 0)).save(source)

    diagnostics = inspect_scene_image(source)

    assert diagnostics.image_format == "PNG"
    assert diagnostics.has_alpha
    assert any("flattened onto white" in item for item in diagnostics.warnings)


def test_image_review_rejects_corrupt_and_extreme_inputs(tmp_path: Path) -> None:
    """Fail before model loading for untrusted or unsupported source files."""
    corrupt = tmp_path / "room.webp"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="corrupt"):
        inspect_scene_image(corrupt)

    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (255, 512)).save(tiny)
    with pytest.raises(ValueError, match="at least 256"):
        inspect_scene_image(tiny)


def test_checkpoint_preparation_requires_license_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    """Gate model use on acceptance, streaming size, and SHA-256."""
    checkpoint = b"pinned research weights"
    installer = SharpCheckpointInstaller(
        tmp_path / "models",
        manifest=manifest(checkpoint),
        opener=lambda _url: io.BytesIO(checkpoint),
    )
    with pytest.raises(PermissionError, match="license"):
        installer.prepare(license_accepted=False)

    progress: list[tuple[int, int]] = []
    result = installer.prepare(
        license_accepted=True,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert result.read_bytes() == checkpoint
    assert installer.is_ready()
    assert progress[-1] == (len(checkpoint), len(checkpoint))
    assert not list(installer.root.glob("*.part"))

    installer.marker_path.write_text("[]", encoding="utf-8")
    assert not installer.is_ready()


def test_checkpoint_checksum_failure_removes_staging(tmp_path: Path) -> None:
    """Never retain or publish interrupted checkpoint bytes."""
    checkpoint = b"expected"
    installer = SharpCheckpointInstaller(
        tmp_path / "models",
        manifest=manifest(checkpoint),
        opener=lambda _url: io.BytesIO(b"tampered"),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        installer.prepare(license_accepted=True)

    assert not installer.checkpoint_path.exists()
    assert not list(installer.root.glob("*.part"))


def test_sharp_ply_validation_returns_camera_metadata(tmp_path: Path) -> None:
    """Validate renderer-required Gaussians, file bounds, and embedded intrinsics."""
    path = tmp_path / "scene.ply"
    write_sharp_ply(path)

    metadata = validate_sharp_ply(path, expected_image_size=(320, 256))

    assert metadata.gaussian_count == 2
    assert metadata.image_size == (320, 256)
    assert metadata.focal_length_px == pytest.approx(300.0)
    assert metadata.field_of_view == pytest.approx(46.2, abs=0.1)


def test_sharp_ply_validation_rejects_nonfinite_and_truncated_data(tmp_path: Path) -> None:
    """Reject poisoned Gaussian values and declared data beyond file bounds."""
    invalid = tmp_path / "invalid.ply"
    write_sharp_ply(invalid, invalid_property="x")
    with pytest.raises(ValueError, match="non-finite x"):
        validate_sharp_ply(invalid)

    truncated = tmp_path / "truncated.ply"
    write_sharp_ply(truncated)
    truncated.write_bytes(truncated.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bounds"):
        validate_sharp_ply(truncated)


def test_sharp_ply_validation_rejects_wrong_encoding_and_camera(tmp_path: Path) -> None:
    """Require the official binary layout and usable embedded focal length."""
    ascii_path = tmp_path / "ascii.ply"
    write_sharp_ply(ascii_path, text=True)
    with pytest.raises(ValueError, match="binary little-endian"):
        validate_sharp_ply(ascii_path)

    invalid_camera = tmp_path / "invalid-camera.ply"
    write_sharp_ply(invalid_camera, focal_length_px=0.0)
    with pytest.raises(ValueError, match="camera intrinsics"):
        validate_sharp_ply(invalid_camera)


def test_scene_builder_publishes_ply_preview_and_provenance(tmp_path: Path) -> None:
    """Adopt only validated outputs and remove every job staging file."""
    source = tmp_path / "my_room.jpg"
    image = Image.new("RGB", (320, 256), "red")
    image.paste("blue", (160, 0, 320, 256))
    exif = Image.Exif()
    exif[274] = 2
    image.save(source, exif=exif)
    checkpoint = b"checkpoint"
    checkpoint_path = tmp_path / "models" / "sharp.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(checkpoint)

    def predict(  # noqa: PLR0913
        image_path: Path,
        focal_length_px: float,
        checkpoint_path: Path,
        device: SharpDevice,
        output_path: Path,
        model_loaded: Callable[[], None],
    ) -> float:
        del focal_length_px, checkpoint_path
        assert device == "cpu"
        model_loaded()
        with Image.open(image_path) as normalized:
            normalized_rgb = np.asarray(normalized.convert("RGB"))
            assert normalized_rgb[128, 40, 2] > normalized_rgb[128, 40, 0]
            assert normalized_rgb[128, 280, 0] > normalized_rgb[128, 280, 2]
        write_sharp_ply(output_path)
        return 42.5

    events: list[object] = []
    scene_root = tmp_path / "scenes"
    reference = SharpSceneBuilder(
        predictor=predict,
        manifest=manifest(checkpoint),
    ).build(
        SceneBuildRequest(
            job_id="job-1",
            selection=SceneImageSelection("my_room.jpg", source),
            config=SharpBuildConfig("cpu", checkpoint_path, scene_root),
        ),
        events.append,
        lambda: False,
    )

    assert reference.format == "ply"
    assert reference.entrypoint == "scene.ply"
    assert reference.preview == "preview.webp"
    assert reference.provenance is not None
    assert reference.provenance.source_size == (320, 256)
    assert reference.provenance.inference_ms == 42.5
    assert reference.default_viewpoint.scene_transform.orientation.x == 1.0
    assert (scene_root / reference.asset_id / "scene.ply").is_file()
    assert (scene_root / reference.asset_id / "preview.webp").is_file()
    assert source.is_file()
    assert not list(scene_root.glob(".sharp-*.part"))
    stages = [event.stage for event in events if isinstance(event, ProgressEvent)]
    assert stages == [
        "validation",
        "model_preparation",
        "model_loading",
        "inference",
        "ply_validation",
        "publication",
        "preview_generation",
    ]


def test_scene_builder_cancellation_removes_staging(tmp_path: Path) -> None:
    """Cooperative cancellation leaves neither uploaded copies nor partial scenes."""
    source = tmp_path / "room.png"
    Image.new("RGB", (320, 256)).save(source)
    checkpoint = b"checkpoint"
    checkpoint_path = tmp_path / "sharp.pt"
    checkpoint_path.write_bytes(checkpoint)
    scene_root = tmp_path / "scenes"

    with pytest.raises(SharpCancelledError):
        SharpSceneBuilder(manifest=manifest(checkpoint)).build(
            SceneBuildRequest(
                "job-cancel",
                SceneImageSelection("room.png", source),
                SharpBuildConfig("cpu", checkpoint_path, scene_root),
            ),
            lambda _event: None,
            lambda: True,
        )

    assert source.is_file()
    assert not list(scene_root.glob(".sharp-*.part"))
