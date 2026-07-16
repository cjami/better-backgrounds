"""Tests for input-camera discovery and persisted selection."""

from typing import TYPE_CHECKING

from better_backgrounds.input_camera import InputCameraSelectionStore

if TYPE_CHECKING:
    from pathlib import Path


def test_input_camera_selection_round_trips_atomically(tmp_path: Path) -> None:
    """Restore the stable device identifier selected by the user."""
    store = InputCameraSelectionStore(tmp_path / "input-camera-v1.json")

    store.save("camera-b")

    assert store.load() == "camera-b"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_input_camera_selection_is_ignored(tmp_path: Path) -> None:
    """Recover from corrupt or incompatible preference data."""
    path = tmp_path / "input-camera-v1.json"
    path.write_text('{"schema_version":2,"device_id":"camera-b"}', encoding="utf-8")

    assert InputCameraSelectionStore(path).load() is None
