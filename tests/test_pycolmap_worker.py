"""Isolated PyCOLMAP stage adapter tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from better_backgrounds import pycolmap_worker

if TYPE_CHECKING:
    import pytest


def test_feature_extraction_uses_portable_cpu_and_per_image_cameras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep wheel behavior aligned with the reviewed CLI policy."""
    called: dict[str, object] = {}

    def extract_features(*args: object, **kwargs: object) -> None:
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(pycolmap_worker.pycolmap, "extract_features", extract_features)
    database = tmp_path / "database.db"
    images = tmp_path / "images"

    pycolmap_worker.extract_features(database, images)

    kwargs = cast("dict[str, object]", called["kwargs"])
    assert kwargs["camera_mode"] == pycolmap_worker.per_image_camera_mode()
    assert kwargs["device"] == pycolmap_worker.cpu_device()
    options = cast("pycolmap_worker.FeatureOptions", kwargs["extraction_options"])
    assert not options.use_gpu
    assert options.num_threads == pycolmap_worker.THREAD_LIMIT


def test_sequential_matching_uses_bounded_cpu_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve temporal matching without a graphics context."""
    called: dict[str, object] = {}

    def match_sequential(*args: object, **kwargs: object) -> None:
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(pycolmap_worker.pycolmap, "match_sequential", match_sequential)

    pycolmap_worker.match_sequential(tmp_path / "database.db")

    kwargs = cast("dict[str, object]", called["kwargs"])
    assert kwargs["device"] == pycolmap_worker.cpu_device()
    matching = cast("pycolmap_worker.FeatureOptions", kwargs["matching_options"])
    pairing = cast("pycolmap_worker.PairingOptions", kwargs["pairing_options"])
    assert not matching.use_gpu
    assert matching.num_threads == pycolmap_worker.THREAD_LIMIT
    assert pairing.overlap == pycolmap_worker.SEQUENTIAL_OVERLAP
    assert pairing.loop_detection


def test_mapping_writes_models_to_the_requested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep PyCOLMAP outputs compatible with resumable model selection."""
    called: dict[str, object] = {}

    def incremental_mapping(*args: object, **kwargs: object) -> dict[int, object]:
        called["args"] = args
        called["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(pycolmap_worker.pycolmap, "incremental_mapping", incremental_mapping)
    output = tmp_path / "sparse"

    pycolmap_worker.map_images(tmp_path / "database.db", tmp_path / "images", output)

    assert output.is_dir()
    kwargs = cast("dict[str, object]", called["kwargs"])
    assert kwargs["database_path"] == tmp_path / "database.db"
    assert kwargs["image_path"] == tmp_path / "images"
    assert kwargs["output_path"] == output
    options = cast("pycolmap_worker.MappingOptions", kwargs["options"])
    assert options.num_threads == pycolmap_worker.THREAD_LIMIT
    assert options.random_seed == 0


def test_frozen_worker_command_reuses_the_desktop_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep PyCOLMAP stages runnable inside the packaged application."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = pycolmap_worker.worker_command()

    assert command == (str(Path(sys.argv[0]).resolve()), "--pycolmap-worker")
