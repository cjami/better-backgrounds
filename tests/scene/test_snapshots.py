"""Rendered room snapshots remain exact scene-derived cache entries."""

from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image

from better_backgrounds.scene import SnapshotStore, Viewpoint, load_sample_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _png(viewpoint: Viewpoint, color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, round(320 / viewpoint.aspect_ratio)), color).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def test_snapshot_round_trips_for_the_exact_scene_and_viewpoint(tmp_path: Path) -> None:
    """Resolve the verified current-view render for its scene and viewpoint."""
    scene = load_sample_manifest().scenes[0]
    viewpoint = scene.default_viewpoint
    store = SnapshotStore(tmp_path)

    published = store.save(
        scene,
        viewpoint,
        _png(viewpoint, (20, 30, 40)),
    )

    assert store.load(scene, viewpoint) == published


def test_snapshot_viewpoint_and_scene_content_are_part_of_the_cache_key(tmp_path: Path) -> None:
    """Never present pixels from different geometry or camera settings."""
    scene = load_sample_manifest().scenes[0]
    viewpoint = scene.default_viewpoint
    store = SnapshotStore(tmp_path)
    store.save(scene, viewpoint, _png(viewpoint, (20, 30, 40)))
    moved = viewpoint.model_copy(update={"field_of_view": 55.0})
    rebuilt = scene.model_copy(
        update={
            "resources": (
                scene.resources[0].model_copy(update={"sha256": "f" * 64}),
                *scene.resources[1:],
            ),
        },
    )

    assert store.load(scene, moved) is None
    assert store.load(rebuilt, viewpoint) is None


def test_snapshot_cache_ignores_corrupt_published_pixels(tmp_path: Path) -> None:
    """Treat a damaged derived image as a cache miss instead of showing it."""
    scene = load_sample_manifest().scenes[0]
    viewpoint = scene.default_viewpoint
    store = SnapshotStore(tmp_path)
    published = store.save(
        scene,
        viewpoint,
        _png(viewpoint, (20, 30, 40)),
    )
    published.background.write_bytes(b"damaged")

    assert store.load(scene, viewpoint) is None
