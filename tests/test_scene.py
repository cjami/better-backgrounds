"""Tests for Phase 3 scene assets and persisted viewpoints."""

from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from PySide6.QtCore import QUrl

from better_backgrounds.scene import (
    AssetInstaller,
    AssetResource,
    CropRegion,
    ManagedSceneResolver,
    Quaternion,
    SceneReference,
    SubjectRegion,
    Vector3,
    Viewpoint,
    ViewpointStore,
    load_sample_manifest,
)

SAMPLE_DOWNLOAD_SIZE = 15_011_322
FIXTURES = Path(__file__).parent / "fixtures"
SOG_FORMAT_VERSION = 2
FIXTURE_SPLAT_COUNT = 8


def resource(path: str, content: bytes) -> AssetResource:
    """Build a resource whose integrity metadata matches its content."""
    return AssetResource(
        path=path,
        url=f"https://assets.example.invalid/{path}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def scene_reference(*resources: AssetResource) -> SceneReference:
    """Build the smallest useful test scene reference."""
    return SceneReference(
        asset_id="test-room-v1",
        display_name="Test room",
        format="sog",
        entrypoint=resources[0].path,
        resources=resources,
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Test Room by Example Author",
        attribution_url="https://example.invalid/test-room",
    )


@pytest.mark.parametrize("path", [r"..\secret.sog", r"C:\scene.sog", "data//scene.sog"])
def test_manifest_rejects_non_posix_resource_paths(path: str) -> None:
    """Keep platform-specific path parsing outside the managed manifest."""
    payload = {
        "path": path,
        "url": "https://example.com/scene.sog",
        "size": 4,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValidationError, match="normalized relative path"):
        AssetResource.model_validate(payload)


def test_asset_install_is_atomic_and_reuses_a_valid_cache(tmp_path: Path) -> None:
    """Expose only complete verified scenes and keep them usable offline."""
    files = {"meta.json": b"metadata", "sh0.webp": b"pixels"}
    reference = scene_reference(*(resource(path, content) for path, content in files.items()))
    requests: list[str] = []

    def open_resource(url: str) -> BytesIO:
        requests.append(url)
        return BytesIO(files[Path(url).name])

    installer = AssetInstaller(tmp_path, opener=open_resource)
    installed = installer.install(reference)

    assert installed == tmp_path / reference.asset_id
    assert installer.is_ready(reference)
    assert not list(tmp_path.glob("*.part"))

    offline = AssetInstaller(
        tmp_path,
        opener=lambda _url: pytest.fail("valid cache should not use the network"),
    )
    assert offline.install(reference) == installed
    assert len(requests) == len(files)


def test_checked_in_sample_manifest_has_a_licensed_sog_scene() -> None:
    """Keep the prepared public sample explicit and reviewable."""
    manifest = load_sample_manifest()

    assert manifest.schema_version == 1
    assert len(manifest.scenes) == 1
    assert manifest.scenes[0].format == "sog"
    assert manifest.scenes[0].license_name == "CC BY 4.0"
    assert manifest.scenes[0].expected_size == SAMPLE_DOWNLOAD_SIZE


def test_lightweight_sog_fixture_has_runtime_resources() -> None:
    """Keep the pinned PLY-to-SOG interchange result available to smoke tests."""
    fixture = FIXTURES / "test-scene.sog"

    assert zipfile.is_zipfile(fixture)
    with zipfile.ZipFile(fixture) as archive:
        metadata = json.loads(archive.read("meta.json"))
        assert metadata["version"] == SOG_FORMAT_VERSION
        assert metadata["count"] == FIXTURE_SPLAT_COUNT
        assert {"means_l.webp", "means_u.webp", "scales.webp", "quats.webp", "sh0.webp"} <= set(
            archive.namelist(),
        )


def test_bad_checksum_never_exposes_a_partial_asset(tmp_path: Path) -> None:
    """Discard a failed staging directory instead of publishing corrupt bytes."""
    reference = scene_reference(resource("scene.sog", b"expected"))
    installer = AssetInstaller(tmp_path, opener=lambda _url: BytesIO(b"tampered"))

    with pytest.raises(ValueError, match="integrity"):
        installer.install(reference)

    assert not installer.is_ready(reference)
    assert not (tmp_path / reference.asset_id).exists()
    assert not list(tmp_path.glob("*.part"))


def test_interrupted_download_can_retry_cleanly(tmp_path: Path) -> None:
    """Clean staging files after interruption so a later attempt can succeed."""
    content = b"complete scene"
    reference = scene_reference(resource("scene.sog", content))
    attempts = iter((OSError("connection dropped"), BytesIO(content)))

    def open_resource(_url: str) -> BytesIO:
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    installer = AssetInstaller(tmp_path, opener=open_resource)
    with pytest.raises(OSError, match="dropped"):
        installer.install(reference)

    assert installer.install(reference).is_dir()
    assert not list(tmp_path.glob("*.part"))


def test_managed_scene_urls_reject_unknown_files_and_traversal(tmp_path: Path) -> None:
    """Resolve only manifest-owned files below a verified managed root."""
    content = b"scene"
    reference = scene_reference(resource("runtime/scene.sog", content))
    installer = AssetInstaller(tmp_path, opener=lambda _url: BytesIO(content))
    installer.install(reference)
    resolver = ManagedSceneResolver(installer, [reference])

    resolved = resolver.resolve(QUrl("bbscene://test-room-v1/runtime/scene.sog"))

    assert resolved == tmp_path / reference.asset_id / "runtime" / "scene.sog"
    assert resolver.resolve(QUrl("bbscene://test-room-v1/runtime/other.sog")) is None
    assert resolver.resolve(QUrl("bbscene://test-room-v1/%2e%2e/secret")) is None
    assert resolver.resolve(QUrl("file:///secret.sog")) is None


def test_viewpoint_round_trips_through_application_data(tmp_path: Path) -> None:
    """Persist a complete room-scoped camera without numerical drift."""
    viewpoint = Viewpoint(
        position=Vector3(x=0.2, y=1.4, z=-2.1),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        orbit_target=Vector3(x=0.0, y=1.1, z=0.0),
        field_of_view=42.5,
        near_clip=0.05,
        far_clip=40.0,
        aspect_ratio=16 / 9,
        crop=CropRegion(left=0.05, top=0.02, right=0.95, bottom=0.98),
        subject_region=SubjectRegion(x=0.35, y=0.2, width=0.3, height=0.7),
        subject_depth=2.4,
        focus_depth=2.6,
    )
    store = ViewpointStore(tmp_path / "viewpoints-v1.json")

    store.save("room-1", viewpoint)

    restored = ViewpointStore(tmp_path / "viewpoints-v1.json").load("room-1")
    assert restored is not None
    assert restored.model_dump() == viewpoint.model_dump()
    assert restored.camera_fingerprint == viewpoint.camera_fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [("field_of_view", float("nan")), ("near_clip", 0.0), ("far_clip", float("inf"))],
)
def test_viewpoint_rejects_non_finite_and_invalid_camera_values(field: str, value: float) -> None:
    """Reject cameras that cannot be rendered or serialized safely."""
    values = Viewpoint().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        Viewpoint.model_validate(values)
