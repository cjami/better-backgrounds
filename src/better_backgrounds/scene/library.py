"""Application scene-library composition service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from better_backgrounds.scene.assets import AssetInstaller, load_sample_manifest
from better_backgrounds.scene.catalogue import SceneCatalogue
from better_backgrounds.scene.resolver import ManagedSceneResolver
from better_backgrounds.scene.selection import RoomSelectionStore
from better_backgrounds.scene.snapshots import SnapshotStore
from better_backgrounds.scene.viewpoints import ViewpointStore

if TYPE_CHECKING:
    from pathlib import Path

    from better_backgrounds.scene.models import SceneReference

DEFAULT_ROOMS = (
    "Table Tennis Room — Sample",
    "Loft — North Window",
    "Studio — West Wall",
    "Living room",
    "Bookshelf corner",
)


class SceneLibrary:
    """Own sample and generated scenes, stable room IDs, assets, and viewpoints."""

    def __init__(self, cache_root: Path, data_root: Path) -> None:
        """Load the persisted catalogue and construct managed asset services."""
        self.cache_root = cache_root
        self.data_root = data_root
        self.sample_scene = load_sample_manifest().scenes[0]
        self.catalogue = SceneCatalogue(data_root / "scene-catalogue-v1.json")
        self.generated_scenes = {scene.asset_id: scene for scene in self.catalogue.scenes()}
        self.assets = AssetInstaller(cache_root)
        self.snapshots = SnapshotStore(cache_root / "rendered-snapshots-v1")
        self.resolver = ManagedSceneResolver(
            self.assets,
            [self.sample_scene, *self.generated_scenes.values()],
        )
        self.viewpoints = ViewpointStore(data_root / "viewpoints-v1.json")
        self.selection = RoomSelectionStore(data_root / "selected-room-v1.json")
        generated_rooms = [scene.display_name for scene in self.generated_scenes.values()]
        self.rooms = [
            *generated_rooms,
            *[room for room in DEFAULT_ROOMS if room not in generated_rooms],
        ]
        self.room_ids = {room: self._initial_room_id(room) for room in self.rooms}

    def scene_for_room(self, room: str) -> SceneReference | None:
        """Resolve a display room to its sample or generated scene."""
        if room == self.sample_scene.display_name:
            return self.sample_scene
        room_id = self.room_ids.get(room)
        return self.generated_scenes.get(room_id) if room_id is not None else None

    def scene_for_id(self, asset_id: str) -> SceneReference | None:
        """Resolve a stable scene identifier independently of the selected room."""
        if asset_id == self.sample_scene.asset_id:
            return self.sample_scene
        return self.generated_scenes.get(asset_id)

    def register(self, scene_id: str, fallback_name: str) -> tuple[str, str]:
        """Register a completed generated scene and return its room name and ID."""
        scene = self.catalogue.find(scene_id)
        if scene is not None and self.assets.is_ready(scene):
            self.generated_scenes[scene.asset_id] = scene
            self.resolver.register(scene)
        room_name = scene.display_name if scene is not None else fallback_name
        room_id = scene.asset_id if scene is not None else self.room_id(room_name)
        if room_name not in self.rooms:
            self.rooms.insert(0, room_name)
        self.room_ids[room_name] = room_id
        return room_name, room_id

    @staticmethod
    def room_id(room_name: str) -> str:
        """Create the stable fallback identifier used by legacy rooms."""
        value = "".join(
            character.lower() if character.isalnum() else "-" for character in room_name
        )
        return "-".join(part for part in value.split("-") if part)

    def _initial_room_id(self, room: str) -> str:
        if room == self.sample_scene.display_name:
            return self.sample_scene.asset_id
        generated = next(
            (
                scene.asset_id
                for scene in self.generated_scenes.values()
                if scene.display_name == room
            ),
            None,
        )
        return generated or self.room_id(room)
