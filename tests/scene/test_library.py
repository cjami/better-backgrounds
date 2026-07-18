"""Focused tests for scene registration and stable room identifiers."""

from typing import TYPE_CHECKING

from better_backgrounds.scene import SceneLibrary

if TYPE_CHECKING:
    from pathlib import Path


def test_scene_library_registers_fallback_room_with_stable_id(tmp_path: Path) -> None:
    """Keep a completed room selectable when no catalogue scene was published."""
    library = SceneLibrary(tmp_path / "cache", tmp_path / "data")

    room_name, room_id = library.register("missing-scene", "North Window Room")

    assert room_name == "North Window Room"
    assert room_id == "north-window-room"
    assert library.room_ids[room_name] == room_id
    assert library.rooms[0] == room_name
    assert library.scene_for_room(room_name) is None
