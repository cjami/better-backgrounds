"""Feature-first: Tests for Phase 3 scene assets and persisted viewpoints."""

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
    DepthOfFieldSettings,
    ManagedSceneResolver,
    Quaternion,
    SceneCatalogue,
    SceneProvenance,
    SceneReference,
    SubjectRegion,
    Vector3,
    Viewpoint,
    ViewpointStore,
    colmap_scene_transform,
    load_sample_manifest,
)

SAMPLE_DOWNLOAD_SIZE = 15_011_322
FIXTURES = Path(__file__).parents[1] / "fixtures"
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


def test_asset_cache_survives_presentation_metadata_changes(tmp_path: Path) -> None:
    """Do not redownload unchanged bytes when a prepared viewpoint is corrected."""
    content = b"scene"
    reference = scene_reference(resource("scene.sog", content))
    installer = AssetInstaller(tmp_path, opener=lambda _url: BytesIO(content))
    installed = installer.install(reference)
    changed = reference.model_copy(update={"display_name": "Renamed test room"})
    offline = AssetInstaller(
        tmp_path,
        opener=lambda _url: pytest.fail("presentation changes must reuse verified bytes"),
    )

    assert offline.install(changed) == installed


def test_checked_in_sample_manifest_has_a_licensed_sog_scene() -> None:
    """Keep the prepared public sample explicit and reviewable."""
    manifest = load_sample_manifest()

    assert manifest.schema_version == 1
    assert len(manifest.scenes) == 1
    assert manifest.scenes[0].format == "sog"
    assert manifest.scenes[0].license_name == "CC BY 4.0"
    assert manifest.scenes[0].expected_size == SAMPLE_DOWNLOAD_SIZE
    assert manifest.scenes[0].default_viewpoint.scene_transform == colmap_scene_transform()


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


def test_generated_scene_is_adopted_and_catalogued_offline(tmp_path: Path) -> None:
    """Publish local output without inventing a network resource URL."""
    source = tmp_path / "generated"
    source.mkdir()
    content = b'{"version":2,"count":1}'
    (source / "meta.json").write_bytes(content)
    local_resource = AssetResource(
        path="meta.json",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    reference = SceneReference(
        asset_id="generated-room-v1",
        display_name="Generated room",
        format="sog",
        entrypoint="meta.json",
        resources=(local_resource,),
        license_name="User-provided capture",
        attribution="Generated locally",
    )
    installer = AssetInstaller(tmp_path / "cache")
    catalogue = SceneCatalogue(tmp_path / "data" / "catalogue.json")

    installer.adopt(reference, source)
    catalogue.save(reference)

    assert installer.is_ready(reference)
    assert catalogue.find(reference.asset_id) == reference


def test_legacy_generated_catalogue_migrates_colmap_orientation(tmp_path: Path) -> None:
    """Repair generated rooms published before coordinate normalization was recorded."""
    reference = scene_reference(resource("meta.json", b"metadata"))
    path = tmp_path / "catalogue.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenes": [reference.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )

    migrated = SceneCatalogue(path).find(reference.asset_id)

    assert migrated is not None
    assert migrated.default_viewpoint.scene_transform == colmap_scene_transform()


def test_v2_catalogue_is_read_without_reapplying_colmap_orientation(tmp_path: Path) -> None:
    """Preserve already-normalized v2 rooms while exposing them through the v3 reader."""
    reference = scene_reference(resource("meta.json", b"metadata"))
    path = tmp_path / "catalogue.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scenes": [reference.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )

    assert SceneCatalogue(path).find(reference.asset_id) == reference


def test_generated_catalogue_is_saved_as_schema_v3(tmp_path: Path) -> None:
    """Publish the current PLY-capable scene contract under schema v3."""
    reference = scene_reference(resource("meta.json", b"metadata"))
    path = tmp_path / "catalogue.json"

    SceneCatalogue(path).save(reference)

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3


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
        depth_of_field=DepthOfFieldSettings(blur_strength=0.65),
    )
    store = ViewpointStore(tmp_path / "viewpoints-v1.json")

    store.save("room-1", viewpoint)

    restored = ViewpointStore(tmp_path / "viewpoints-v1.json").load("room-1")
    assert restored is not None
    assert restored.model_dump() == viewpoint.model_dump()
    assert restored.camera_fingerprint == viewpoint.camera_fingerprint


def test_v1_viewpoint_keeps_original_subject_framing_and_depth() -> None:
    """Preserve the original full-frame subject behavior while adding DOF."""
    payload = Viewpoint().model_dump(mode="json")
    payload.pop("depth_of_field")
    payload.update(
        schema_version=1,
        subject_region=SubjectRegion(x=0.2, y=0.1, width=0.4, height=0.6).model_dump(),
        subject_depth=3.0,
        focus_depth=2.6,
    )

    migrated = Viewpoint.model_validate(payload)

    assert migrated.schema_version == 4
    assert migrated.subject_region == SubjectRegion(x=0.2, y=0.1, width=0.4, height=0.6)
    assert migrated.depth_of_field.blur_strength == 0.0


def test_v2_placement_viewpoint_returns_to_original_subject_framing() -> None:
    """Load settings saved by the withdrawn placement implementation safely."""
    payload = Viewpoint().model_dump(mode="json")
    payload.pop("subject_region")
    payload.update(
        schema_version=2,
        subject_placement={
            "anchor_x": 0.5,
            "anchor_y": 0.9,
            "reference_height": 0.7,
            "scale": 1.5,
            "reference_depth": 2.4,
            "depth": 3.0,
            "occlusion_enabled": True,
        },
    )

    migrated = Viewpoint.model_validate(payload)

    assert migrated.schema_version == 4
    assert migrated.subject_region.x == pytest.approx(0.35)
    assert migrated.subject_region.y == pytest.approx(0.2)
    assert migrated.subject_region.width == pytest.approx(0.3)
    assert migrated.subject_region.height == pytest.approx(0.7)
    assert migrated.depth_of_field.blur_strength == 0.0


def test_v3_depth_controls_collapse_to_one_background_blur_amount() -> None:
    """Preserve enabled blur while discarding invisible calibration controls."""
    payload = Viewpoint().model_dump(mode="json")
    payload.update(
        schema_version=3,
        subject_depth=3.4,
        depth_of_field={"enabled": True, "focus_range": 0.3, "blur_strength": 0.7},
    )

    migrated = Viewpoint.model_validate(payload)

    assert migrated.schema_version == 4
    assert migrated.depth_of_field == DepthOfFieldSettings(blur_strength=0.7)


def test_sharp_provenance_exposes_metric_depth_capability() -> None:
    """Gate depth effects on explicit local SHARP provenance."""
    reference = scene_reference(resource("scene.ply", b"scene")).model_copy(
        update={
            "format": "ply",
            "provenance": SceneProvenance(
                source_kind="upload",
                source_sha256="1" * 64,
                source_size=(1280, 720),
                builder_revision="2" * 40,
                checkpoint_sha256="3" * 64,
                device="cuda",
                inference_ms=1200,
                license_name="Apple SHARP Research License",
            ),
        },
    )

    assert reference.supports_metric_depth
    assert not reference.model_copy(update={"format": "sog"}).supports_metric_depth
    assert not reference.model_copy(update={"provenance": None}).supports_metric_depth


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
