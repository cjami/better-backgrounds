"""Isolated PyCOLMAP stages for cancellable reconstruction jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pycolmap

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PYCOLMAP_VERSION = "4.1.0"
THREAD_LIMIT = 8
SEQUENTIAL_OVERLAP = 10


class FeatureOptions(Protocol):
    """Typed portion of PyCOLMAP feature options used by the adapter."""

    num_threads: int
    use_gpu: bool


class PairingOptions(Protocol):
    """Typed portion of sequential pairing options used by the adapter."""

    overlap: int
    loop_detection: bool


class MappingOptions(Protocol):
    """Typed portion of incremental mapping options used by the adapter."""

    num_threads: int
    random_seed: int


class ReconstructionModel(Protocol):
    """Model export operation required from PyCOLMAP."""

    def write_text(self, output: Path) -> None:
        """Write a text-format COLMAP model."""


def _api(name: str) -> object:
    return getattr(pycolmap, name)


def _factory(name: str) -> object:
    return _api(name)


def _member(container: str, name: str) -> object:
    return getattr(_api(container), name)


def cpu_device() -> object:
    """Return PyCOLMAP's portable CPU device value."""
    return _member("Device", "cpu")


def per_image_camera_mode() -> object:
    """Return the camera policy used for uncalibrated video frames."""
    return _member("CameraMode", "PER_IMAGE")


def extraction_options() -> FeatureOptions:
    """Use deterministic, display-independent SIFT extraction."""
    factory = cast("Callable[..., FeatureOptions]", _factory("FeatureExtractionOptions"))
    return factory(
        num_threads=THREAD_LIMIT,
        use_gpu=False,
    )


def matching_options() -> FeatureOptions:
    """Use the CPU matcher supported by every reviewed wheel."""
    factory = cast("Callable[..., FeatureOptions]", _factory("FeatureMatchingOptions"))
    return factory(
        num_threads=THREAD_LIMIT,
        use_gpu=False,
    )


def pairing_options() -> PairingOptions:
    """Match nearby frames and retain the reviewed loop policy."""
    factory = cast("Callable[..., PairingOptions]", _factory("SequentialPairingOptions"))
    return factory(
        overlap=SEQUENTIAL_OVERLAP,
        loop_detection=True,
        num_threads=THREAD_LIMIT,
    )


def mapping_options() -> MappingOptions:
    """Bound mapping concurrency and keep initialization reproducible."""
    factory = cast("Callable[..., MappingOptions]", _factory("IncrementalPipelineOptions"))
    return factory(
        num_threads=THREAD_LIMIT,
        random_seed=0,
    )


def extract_features(database: Path, images: Path) -> None:
    """Extract per-image-camera SIFT features into a COLMAP database."""
    extract = cast("Callable[..., None]", _api("extract_features"))
    reader_factory = cast("Callable[..., object]", _factory("ImageReaderOptions"))
    extract(
        database_path=database,
        image_path=images,
        camera_mode=per_image_camera_mode(),
        reader_options=reader_factory(camera_model="SIMPLE_RADIAL"),
        extraction_options=extraction_options(),
        device=cpu_device(),
    )


def match_sequential(database: Path) -> None:
    """Match temporally adjacent features on the portable CPU backend."""
    match = cast("Callable[..., None]", _api("match_sequential"))
    match(
        database_path=database,
        matching_options=matching_options(),
        pairing_options=pairing_options(),
        device=cpu_device(),
    )


def map_images(database: Path, images: Path, output: Path) -> None:
    """Write every useful incremental reconstruction under one output root."""
    output.mkdir(parents=True, exist_ok=True)
    mapping = cast("Callable[..., object]", _api("incremental_mapping"))
    mapping(
        database_path=database,
        image_path=images,
        output_path=output,
        options=mapping_options(),
    )


def export_text_model(source: Path, output: Path) -> None:
    """Export a binary sparse model for stable quality inspection."""
    output.mkdir(parents=True, exist_ok=True)
    factory = cast("Callable[[Path], ReconstructionModel]", _factory("Reconstruction"))
    factory(source).write_text(output)


def worker_command() -> tuple[str, ...]:
    """Return a stage prefix that works in source and frozen applications."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return (str(Path(sys.argv[0]).resolve()), "--pycolmap-worker")
    return (sys.executable, "-m", "better_backgrounds.pycolmap_worker")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    extraction = subparsers.add_parser("feature-extractor")
    extraction.add_argument("--database-path", type=Path, required=True)
    extraction.add_argument("--image-path", type=Path, required=True)

    matching = subparsers.add_parser("sequential-matcher")
    matching.add_argument("--database-path", type=Path, required=True)

    mapping = subparsers.add_parser("mapper")
    mapping.add_argument("--database-path", type=Path, required=True)
    mapping.add_argument("--image-path", type=Path, required=True)
    mapping.add_argument("--output-path", type=Path, required=True)

    conversion = subparsers.add_parser("model-converter")
    conversion.add_argument("--input-path", type=Path, required=True)
    conversion.add_argument("--output-path", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one bounded stage and return a process-compatible status."""
    version = str(_api("__version__"))
    if version != PYCOLMAP_VERSION:
        sys.stderr.write(
            f"Expected PyCOLMAP {PYCOLMAP_VERSION}, found {version}.\n",
        )
        return 2
    values = _parser().parse_args(arguments)
    try:
        if values.stage == "feature-extractor":
            extract_features(values.database_path, values.image_path)
        elif values.stage == "sequential-matcher":
            match_sequential(values.database_path)
        elif values.stage == "mapper":
            map_images(values.database_path, values.image_path, values.output_path)
        else:
            export_text_model(values.input_path, values.output_path)
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
