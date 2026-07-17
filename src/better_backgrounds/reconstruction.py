"""Resumable native reconstruction commands, manifests, and artifact validation."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from collections.abc import Mapping, Sequence  # noqa: TC003
from contextlib import suppress
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

FILE_CHUNK_SIZE = 1024 * 1024
COLMAP_BINARY_COUNT_SIZE = 8
MINIMUM_PLY_SIZE = 128
MINIMUM_TEMPORAL_INDICES = 2
POINT_ERROR_MINIMUM_FIELDS = 8


class ReconstructionModel(BaseModel):
    """Reject undeclared values in durable reconstruction state."""

    model_config = ConfigDict(extra="forbid")


class ReconstructionQuality(StrEnum):
    """Name one bounded reconstruction cost and quality policy."""

    PREVIEW = "preview"
    BALANCED = "balanced"
    QUALITY = "quality"


class ReconstructionPreset(ReconstructionModel):
    """Keep every expensive quality lever together at one boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quality: ReconstructionQuality
    selected_frames: int = Field(ge=60, le=100)
    maximum_image_edge: int = Field(ge=1280, le=1920)
    brush_steps: int = Field(ge=1)


RECONSTRUCTION_PRESETS: Mapping[ReconstructionQuality, ReconstructionPreset] = MappingProxyType(
    {
        ReconstructionQuality.PREVIEW: ReconstructionPreset(
            quality=ReconstructionQuality.PREVIEW,
            selected_frames=60,
            maximum_image_edge=1280,
            brush_steps=3_000,
        ),
        ReconstructionQuality.BALANCED: ReconstructionPreset(
            quality=ReconstructionQuality.BALANCED,
            selected_frames=80,
            maximum_image_edge=1600,
            brush_steps=6_000,
        ),
        ReconstructionQuality.QUALITY: ReconstructionPreset(
            quality=ReconstructionQuality.QUALITY,
            selected_frames=100,
            maximum_image_edge=1920,
            brush_steps=12_000,
        ),
    }
)


def reconstruction_preset(
    quality: ReconstructionQuality | str,
) -> ReconstructionPreset:
    """Resolve a validated quality name to its immutable work budget."""
    return RECONSTRUCTION_PRESETS[ReconstructionQuality(quality)]


class StageArtifact(ReconstructionModel):
    """Record one completed stage and all outputs needed to reuse it."""

    fingerprint: str = Field(min_length=1, max_length=128)
    outputs: tuple[str, ...] = Field(min_length=1)


class JobManifest(ReconstructionModel):
    """Version resumable work without treating partial output as complete."""

    schema_version: Literal[1] = 1
    job_id: str = Field(min_length=1, max_length=128)
    input_path: Path
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_fingerprint: str = Field(min_length=1, max_length=128)
    status: Literal["running", "cancelled", "failed", "completed"] = "running"
    current_stage: str | None = Field(default=None, max_length=80)
    stages: dict[str, StageArtifact] = Field(default_factory=dict)
    scene_id: str | None = Field(default=None, max_length=128)

    def can_resume(self, source: Path, configuration_fingerprint: str, job_root: Path) -> bool:
        """Return whether source, settings, and every claimed output still match."""
        if (
            self.input_fingerprint != fingerprint_file(source)
            or self.configuration_fingerprint != configuration_fingerprint
        ):
            return False
        return all(
            all(_safe_output(job_root, output).is_file() for output in artifact.outputs)
            for artifact in self.stages.values()
        )

    def stage_is_current(
        self,
        stage: str,
        fingerprint: str,
        job_root: Path,
    ) -> bool:
        """Return whether one stage's fingerprint and complete files can be reused."""
        artifact = self.stages.get(stage)
        return bool(
            artifact is not None
            and artifact.fingerprint == fingerprint
            and all(_safe_output(job_root, output).is_file() for output in artifact.outputs)
        )


class JobManifestStore:
    """Persist job state through atomic document replacement."""

    def __init__(self, path: Path) -> None:
        """Own one job.json path."""
        self.path = path

    def load(self) -> JobManifest | None:
        """Return a valid manifest or ignore incomplete/corrupt state."""
        try:
            return JobManifest.model_validate_json(self.path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return None

    def save(self, manifest: JobManifest) -> None:
        """Atomically replace the durable job state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)


class ColmapQuality(ReconstructionModel):
    """Summarize camera registration and sparse-model quality evidence."""

    registered_images: int = Field(ge=0)
    registered_proportion: float = Field(ge=0, le=1, allow_inf_nan=False)
    temporal_span: float = Field(ge=0, le=1, allow_inf_nan=False)
    longest_gap: int = Field(ge=0)
    median_reprojection_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)


def _safe_output(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        msg = f"job output escapes its root: {relative}"
        raise ValueError(msg)
    return candidate


def fingerprint_file(path: Path) -> str:
    """Hash an input or artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(FILE_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_configuration(configuration: Mapping[str, object]) -> str:
    """Hash stable JSON settings used to decide cache compatibility."""
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class ReconstructionCommands:
    """Construct native argv without a shell or path interpolation."""

    def __init__(
        self,
        *,
        ffmpeg: Path,
        pycolmap: Sequence[str],
        brush: Path,
        splat_transform: Path,
    ) -> None:
        """Pin every executable at the adapter boundary."""
        if not pycolmap:
            msg = "PyCOLMAP worker command cannot be empty"
            raise ValueError(msg)
        self.ffmpeg = ffmpeg
        self.pycolmap = tuple(pycolmap)
        self.brush = brush
        self.splat_transform = splat_transform

    def extract_frames(
        self,
        video: Path,
        output: Path,
        *,
        fps: float,
        maximum_image_edge: int,
    ) -> list[str]:
        """Build deterministic timestamp-normalized candidate extraction."""
        if maximum_image_edge <= 0:
            msg = "maximum image edge must be positive"
            raise ValueError(msg)
        scale = (
            f"scale=w='if(gte(iw,ih),min(iw,{maximum_image_edge}),-2)'"
            f":h='if(gte(iw,ih),-2,min(ih,{maximum_image_edge}))'"
        )
        return [
            str(self.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-i",
            str(video),
            "-map_metadata",
            "-1",
            "-vf",
            f"fps={fps:.6g},setpts=N/({fps:.6g}*TB),{scale}",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(output / "%06d.png"),
        ]

    def feature_extraction(self, images: Path, database: Path) -> list[str]:
        """Build explicit feature extraction with conservative camera policy."""
        return [
            *self.pycolmap,
            "feature-extractor",
            "--database-path",
            str(database),
            "--image-path",
            str(images),
        ]

    def sequential_matching(self, database: Path) -> list[str]:
        """Build overlap-aware sequential matching."""
        return [
            *self.pycolmap,
            "sequential-matcher",
            "--database-path",
            str(database),
        ]

    def mapping(self, images: Path, database: Path, sparse: Path) -> list[str]:
        """Build sparse mapping in an isolated output directory."""
        return [
            *self.pycolmap,
            "mapper",
            "--database-path",
            str(database),
            "--image-path",
            str(images),
            "--output-path",
            str(sparse),
        ]

    def model_to_text(self, source: Path, destination: Path) -> list[str]:
        """Build a stable text conversion for quality validation."""
        return [
            *self.pycolmap,
            "model-converter",
            "--input-path",
            str(source),
            "--output-path",
            str(destination),
        ]

    def brush_training(self, dataset: Path, output: Path, *, iterations: int) -> list[str]:
        """Build bounded Brush training and PLY export."""
        return [
            str(self.brush),
            str(dataset),
            "--total-steps",
            str(iterations),
            "--export-every",
            str(iterations),
            "--export-path",
            str(output.parent),
            "--export-name",
            output.name,
        ]

    def convert_sog(
        self,
        source: Path,
        destination: Path,
        *,
        gpu: Literal["cpu"] | None = None,
    ) -> list[str]:
        """Build the npm-pinned PLY-to-SOG conversion."""
        command = [
            str(self.splat_transform),
            "-q",
            "-w",
        ]
        if gpu is not None:
            command.extend(("-g", gpu))
        command.extend(("--max-workers", "0", str(source), str(destination)))
        return command


def registered_image_count(model: Path) -> int:
    """Return the registered image count from a COLMAP text or binary model."""
    binary = model / "images.bin"
    if binary.is_file() and binary.stat().st_size >= COLMAP_BINARY_COUNT_SIZE:
        with binary.open("rb") as source:
            return int(struct.unpack("<Q", source.read(COLMAP_BINARY_COUNT_SIZE))[0])
    text = model / "images.txt"
    if not text.is_file():
        return 0
    content = text.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Number of images:\s*(\d+)", content)
    if match is not None:
        return int(match.group(1))
    data_lines = [line for line in content.splitlines() if line and not line.startswith("#")]
    return len(data_lines) // 2


def select_largest_model(root: Path) -> Path | None:
    """Return the deterministic COLMAP model with most registered images."""
    models = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    useful = [(model, registered_image_count(model)) for model in models]
    useful = [(model, count) for model, count in useful if count > 0]
    if not useful:
        return None
    return max(useful, key=lambda item: (item[1], item[0].name))[0]


def analyse_colmap_model(model: Path, selected_count: int) -> ColmapQuality:
    """Measure registered coverage, continuity, temporal span, and reprojection error."""
    if selected_count <= 0:
        msg = "selected frame count must be positive"
        raise ValueError(msg)
    images = model / "images.txt"
    if not images.is_file():
        msg = "COLMAP text model is missing images.txt"
        raise ValueError(msg)
    data_lines = [
        line
        for line in images.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and not line.startswith("#")
    ]
    image_header = re.compile(
        r"^\d+\s+(?:[-+eE.\d]+\s+){7}\d+\s+\S+$",
    )
    image_lines = [line for line in data_lines if image_header.fullmatch(line)]
    indices: list[int] = []
    for line in image_lines:
        name = line.split()[-1]
        with suppress(ValueError):
            indices.append(int(Path(name).stem))
    indices.sort()
    registered = len(image_lines)
    if len(indices) >= MINIMUM_TEMPORAL_INDICES:
        temporal_span = (indices[-1] - indices[0]) / max(selected_count - 1, 1)
        longest_gap = max(second - first for first, second in pairwise(indices))
    else:
        temporal_span = 0.0
        longest_gap = selected_count
    errors: list[float] = []
    points = model / "points3D.txt"
    if points.is_file():
        for line in points.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= POINT_ERROR_MINIMUM_FIELDS:
                with suppress(ValueError):
                    errors.append(float(parts[7]))
    median_error = float(np.median(errors)) if errors else None
    return ColmapQuality(
        registered_images=registered,
        registered_proportion=min(registered / selected_count, 1.0),
        temporal_span=min(max(temporal_span, 0.0), 1.0),
        longest_gap=longest_gap,
        median_reprojection_error=median_error,
    )


def validate_ply(path: Path) -> None:
    """Reject missing, truncated, or non-splat PLY output."""
    if not path.is_file() or path.stat().st_size < MINIMUM_PLY_SIZE:
        msg = "Brush did not produce a usable PLY file"
        raise ValueError(msg)
    with path.open("rb") as source:
        header = source.read(min(path.stat().st_size, 64 * 1024))
    end = header.find(b"end_header")
    if not header.startswith(b"ply\n") or end < 0:
        msg = "Brush produced an invalid PLY header"
        raise ValueError(msg)
    header_text = header[:end].decode("ascii", errors="replace")
    match = re.search(r"element vertex\s+(\d+)", header_text)
    if match is None or int(match.group(1)) <= 0:
        msg = "Brush PLY contains no splats"
        raise ValueError(msg)


def validate_sog(path: Path) -> None:
    """Reject incomplete SOG output before scene publication."""
    if not path.is_file() or not zipfile.is_zipfile(path):
        msg = "SplatTransform did not produce a readable SOG archive"
        raise ValueError(msg)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            "meta.json",
            "means_l.webp",
            "means_u.webp",
            "scales.webp",
            "quats.webp",
            "sh0.webp",
        }
        if not required.issubset(names):
            msg = "SOG archive is missing required renderer resources"
            raise ValueError(msg)
        metadata = json.loads(archive.read("meta.json"))
        if not isinstance(metadata, dict) or int(metadata.get("count", 0)) <= 0:
            msg = "SOG metadata contains no splats"
            raise ValueError(msg)


def relative_outputs(root: Path, paths: Sequence[Path]) -> tuple[str, ...]:
    """Convert validated job-owned paths to portable manifest values."""
    return tuple(path.resolve().relative_to(root.resolve()).as_posix() for path in paths)
