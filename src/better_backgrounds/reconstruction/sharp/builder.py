"""Atomic image-to-scene reconstruction with Apple SHARP."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from better_backgrounds.jobs.events import JobEvent, ProgressEvent, WarningEvent
from better_backgrounds.reconstruction.images import (
    SceneImageDiagnostics,
    SceneImageSelection,
    inspect_scene_image,
    sha256_file,
)
from better_backgrounds.reconstruction.sharp.checkpoint import (
    SharpCheckpointInstaller,
    load_sharp_checkpoint_manifest,
    probe_sharp_capabilities,
)
from better_backgrounds.reconstruction.sharp.contracts import (
    SceneBuildRequest,
    SharpCancelledError,
    SharpCapabilities,
    SharpCheckpointManifest,
    SharpPlyMetadata,
    SharpPredictor,
)
from better_backgrounds.reconstruction.sharp.ply import validate_sharp_ply
from better_backgrounds.reconstruction.sharp.runtime import run_sharp_inference
from better_backgrounds.scene import (
    AssetInstaller,
    AssetResource,
    CameraBounds,
    SceneProvenance,
    SceneReference,
    Vector3,
    Viewpoint,
    sharp_scene_transform,
)

EmitEvent = Callable[[JobEvent], None]
CancellationCheck = Callable[[], bool]
SUPPORTED_OUTPUT_ASPECT_RATIOS = (16 / 9, 4 / 3, 1.0)


def _sharp_viewpoint(metadata: SharpPlyMetadata) -> Viewpoint:
    width, height = metadata.image_size
    source_aspect_ratio = width / height
    output_aspect_ratio = min(
        SUPPORTED_OUTPUT_ASPECT_RATIOS,
        key=lambda candidate: abs(candidate - source_aspect_ratio),
    )
    return Viewpoint(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orbit_target=Vector3(x=0.0, y=0.0, z=-2.0),
        field_of_view=min(max(metadata.field_of_view, 24.0), 90.0),
        aspect_ratio=output_aspect_ratio,
        horizon=0.0,
        near_clip=0.03,
        far_clip=30.0,
        scene_transform=sharp_scene_transform(),
        safe_camera_region=CameraBounds(
            minimum=Vector3(x=-0.35, y=-0.25, z=-0.35),
            maximum=Vector3(x=0.35, y=0.25, z=0.35),
        ),
    )


def _normalized_input(source: Path, destination: Path) -> None:
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        if "A" in oriented.getbands() or "transparency" in opened.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            normalized = background.convert("RGB")
        else:
            normalized = oriented.convert("RGB")
        normalized.save(destination, format="PNG")


def _preview_image(normalized: Path, destination: Path) -> None:
    with Image.open(normalized) as opened:
        preview = opened.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    preview.save(destination, format="WEBP", quality=88, method=6)


def _scene_id(selection: SceneImageSelection, source_sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(selection.display_name).stem.lower()).strip("-")
    return f"{(stem or 'room')[:48]}-{source_sha256[:10]}"


class SharpSceneBuilder:
    """Build, validate, and atomically adopt one upload-first SHARP scene."""

    def __init__(
        self,
        *,
        predictor: SharpPredictor = run_sharp_inference,
        manifest: SharpCheckpointManifest | None = None,
    ) -> None:
        """Inject only the heavyweight inference edge and pinned manifest."""
        self._predictor = predictor
        self._manifest = manifest or load_sharp_checkpoint_manifest()

    def build(
        self,
        request: SceneBuildRequest,
        emit_event: EmitEvent,
        is_cancelled: CancellationCheck,
    ) -> SceneReference:
        """Run every gate and return only a verified managed SceneReference."""
        selection = request.selection
        source = selection.source_path
        if source is None:
            msg = "A SHARP build requires a selected room image"
            raise ValueError(msg)
        staging = request.config.output_root / f".sharp-{request.job_id}-{uuid4().hex}.part"
        try:
            self._progress(emit_event, request.job_id, "validation", 0.02, "Validating image")
            diagnostics = inspect_scene_image(source)
            source_sha256 = sha256_file(source)
            staging.mkdir(parents=True)
            normalized = staging / "input.png"
            preview = staging / "preview.webp"
            scene_path = staging / "scene.ply"
            _normalized_input(source, normalized)
            _preview_image(normalized, preview)
            for index, concern in enumerate(diagnostics.warnings):
                emit_event(
                    WarningEvent(
                        job_id=request.job_id,
                        code=f"image_warning_{index + 1}",
                        message=concern,
                    )
                )
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "model_preparation",
                0.16,
                "Verifying the pinned SHARP checkpoint",
            )
            checkpoint = SharpCheckpointInstaller(
                request.config.checkpoint_path.parent,
                manifest=self._manifest,
            )
            checkpoint.validate(request.config.checkpoint_path)
            capabilities = (
                probe_sharp_capabilities(request.config.device)
                if self._predictor is run_sharp_inference
                else SharpCapabilities(
                    device_type=(
                        "cpu" if request.config.device == "auto" else request.config.device
                    ),
                    accelerated=request.config.device not in {"auto", "cpu"},
                )
            )
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "model_loading",
                0.26,
                f"Loading SHARP on {capabilities.device_type.upper()}",
            )

            def model_loaded() -> None:
                self._progress(
                    emit_event,
                    request.job_id,
                    "inference",
                    0.38,
                    "Predicting metric 3D Gaussians",
                )

            inference_ms = self._predictor(
                normalized,
                diagnostics.focal_length_px,
                request.config.checkpoint_path,
                capabilities.device_type,
                scene_path,
                model_loaded,
            )
            self._cancel_if_requested(is_cancelled)

            self._progress(
                emit_event,
                request.job_id,
                "ply_validation",
                0.75,
                "Validating Gaussian scene and camera metadata",
            )
            metadata = validate_sharp_ply(
                scene_path,
                expected_image_size=(diagnostics.width, diagnostics.height),
            )
            self._cancel_if_requested(is_cancelled)
            normalized.unlink(missing_ok=True)
            reference = self._reference(
                selection,
                staging,
                source_sha256,
                diagnostics,
                metadata,
                capabilities,
                inference_ms,
            )
            self._progress(
                emit_event,
                request.job_id,
                "publication",
                0.9,
                "Publishing verified scene resources",
            )
            AssetInstaller(request.config.output_root).adopt(reference, staging)
            self._progress(
                emit_event,
                request.job_id,
                "preview_generation",
                0.98,
                "Preview ready",
            )
            return reference
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _reference(
        self,
        selection: SceneImageSelection,
        staging: Path,
        source_sha256: str,
        diagnostics: SceneImageDiagnostics,
        metadata: SharpPlyMetadata,
        capabilities: SharpCapabilities,
        inference_ms: float,
    ) -> SceneReference:
        resources = tuple(
            AssetResource(
                path=path.relative_to(staging).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in (staging / "scene.ply", staging / "preview.webp")
        )
        display_name = Path(selection.display_name).stem.replace("_", " ").strip().title()
        return SceneReference(
            asset_id=_scene_id(selection, source_sha256),
            display_name=display_name or "Room",
            format="ply",
            entrypoint="scene.ply",
            resources=resources,
            license_name="User source; SHARP research-model output",
            license_url=self._manifest.license_url,
            attribution=f"Generated locally with Apple SHARP from {selection.display_name}",
            attribution_url="https://github.com/apple/ml-sharp",
            preview="preview.webp",
            default_viewpoint=_sharp_viewpoint(metadata),
            provenance=SceneProvenance(
                source_kind=selection.source_kind,
                source_sha256=source_sha256,
                source_size=(diagnostics.width, diagnostics.height),
                builder_revision=self._manifest.builder_revision,
                checkpoint_sha256=self._manifest.sha256,
                device=capabilities.device_type,
                inference_ms=inference_ms,
                license_name=self._manifest.license_name,
                license_url=self._manifest.license_url,
            ),
        )

    @staticmethod
    def _progress(
        emit: EmitEvent,
        job_id: str,
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        emit(ProgressEvent(job_id=job_id, stage=stage, progress=progress, message=message))

    @staticmethod
    def _cancel_if_requested(is_cancelled: CancellationCheck) -> None:
        if is_cancelled():
            msg = "SHARP room build was cancelled"
            raise SharpCancelledError(msg)
