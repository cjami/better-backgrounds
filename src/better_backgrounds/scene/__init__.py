"""Managed scene assets and room-scoped virtual camera state."""

from better_backgrounds.scene.assets import AssetInstaller, SceneAssetManifest, load_sample_manifest
from better_backgrounds.scene.catalogue import SceneCatalogue
from better_backgrounds.scene.library import SceneLibrary
from better_backgrounds.scene.models import (
    APP_SCHEME,
    SCENE_SCHEME,
    AssetResource,
    CameraBounds,
    CropRegion,
    DepthOfFieldSettings,
    Quaternion,
    SceneProvenance,
    SceneReference,
    SceneTransform,
    SubjectRegion,
    Vector3,
    Viewpoint,
    colmap_scene_transform,
    normalize_colmap_scene_reference,
    sharp_scene_transform,
)
from better_backgrounds.scene.resolver import ManagedSceneResolver
from better_backgrounds.scene.selection import RoomSelectionStore
from better_backgrounds.scene.snapshots import SnapshotPaths, SnapshotStore
from better_backgrounds.scene.viewpoints import ViewpointStore

__all__ = [
    "APP_SCHEME",
    "SCENE_SCHEME",
    "AssetInstaller",
    "AssetResource",
    "CameraBounds",
    "CropRegion",
    "DepthOfFieldSettings",
    "ManagedSceneResolver",
    "Quaternion",
    "RoomSelectionStore",
    "SceneAssetManifest",
    "SceneCatalogue",
    "SceneLibrary",
    "SceneProvenance",
    "SceneReference",
    "SceneTransform",
    "SnapshotPaths",
    "SnapshotStore",
    "SubjectRegion",
    "Vector3",
    "Viewpoint",
    "ViewpointStore",
    "colmap_scene_transform",
    "load_sample_manifest",
    "normalize_colmap_scene_reference",
    "sharp_scene_transform",
]
