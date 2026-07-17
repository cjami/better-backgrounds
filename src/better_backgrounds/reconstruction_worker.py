"""Cancellable worker for local video analysis and Gaussian reconstruction."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

import cv2
import psutil
from pydantic import ValidationError

from better_backgrounds.managed_tools import safe_extract_archive
from better_backgrounds.protocol import (
    CancelControl,
    CancelledEvent,
    ErrorEvent,
    JobEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
)
from better_backgrounds.reconstruction import (
    JobManifest,
    JobManifestStore,
    ReconstructionCommands,
    ReconstructionQuality,
    StageArtifact,
    analyse_colmap_model,
    fingerprint_configuration,
    fingerprint_file,
    reconstruction_preset,
    registered_image_count,
    relative_outputs,
    select_largest_model,
    validate_ply,
    validate_sog,
)
from better_backgrounds.scene import (
    AssetInstaller,
    AssetResource,
    SceneCatalogue,
    SceneReference,
    normalize_colmap_scene_reference,
)
from better_backgrounds.video_analysis import (
    CaptureDiagnostics,
    FrameMetrics,
    analyse_metrics,
    probe_video,
    score_frame,
    select_frames,
    validate_probe,
)

EventSink = Callable[[JobEvent], None]
NATIVE_TIMEOUT_SECONDS = 45 * 60
MINIMUM_REGISTERED_IMAGES = 30
MINIMUM_REGISTERED_PROPORTION = 0.5
MINIMUM_TEMPORAL_SPAN = 0.6
MAXIMUM_CONTINUITY_GAP = 12
MAXIMUM_REPROJECTION_ERROR = 4.0


class ReconstructionCancelledError(RuntimeError):
    """Stop a cooperative worker without reporting a failure."""


class NativeCommandError(RuntimeError):
    """Report a bounded native-stage failure."""

    def __init__(self, stage: str, return_code: int | None, log_path: Path) -> None:
        """Keep user-facing details safe while retaining the complete log."""
        code = "timed out" if return_code is None else f"exited with code {return_code}"
        super().__init__(f"{stage} {code}")
        self.stage = stage
        self.return_code = return_code
        self.log_path = log_path


class NativeProcessRunner:
    """Run one native command with cancellation, timeout, and tree cleanup."""

    def __init__(self, cancelled: threading.Event, log_path: Path) -> None:
        """Write native output to one durable job log."""
        self.cancelled = cancelled
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def run(self, command: Sequence[str], *, stage: str, timeout: float) -> None:
        """Execute argv without a shell and stop all descendants when requested."""
        if self.cancelled.is_set():
            raise ReconstructionCancelledError
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab", buffering=0) as log:
            log.write(f"\n[{stage}] {json.dumps(list(command), ensure_ascii=False)}\n".encode())
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(  # noqa: S603
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=self.log_path.parent,
                shell=False,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
            with self._lock:
                self._process = process
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if self.cancelled.wait(0.1):
                        self._terminate_tree(process)
                        raise ReconstructionCancelledError
                    if time.monotonic() >= deadline:
                        self._terminate_tree(process)
                        raise NativeCommandError(stage, None, self.log_path)
                if process.returncode != 0:
                    raise NativeCommandError(stage, process.returncode, self.log_path)
            finally:
                with self._lock:
                    self._process = None

    def cancel(self) -> None:
        """Signal cancellation and immediately stop the active native tree."""
        self.cancelled.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate_tree(process)

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
        with suppress(psutil.Error):
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                with suppress(psutil.Error):
                    child.terminate()
            with suppress(psutil.Error):
                parent.terminate()
            _gone, alive = psutil.wait_procs([*children, parent], timeout=2)
            for item in alive:
                with suppress(psutil.Error):
                    item.kill()
        if process.poll() is None:
            with suppress(OSError):
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)


class ReconstructionWorker:
    """Run and persist one defensible reconstruction pipeline."""

    def __init__(
        self,
        *,
        job_id: str,
        video: Path,
        job_root: Path,
        scene_cache_root: Path,
        catalogue_path: Path,
        ffprobe: Path,
        commands: ReconstructionCommands,
        emit: EventSink,
        resume: bool = False,
        quality: ReconstructionQuality = ReconstructionQuality.BALANCED,
    ) -> None:
        """Receive only trusted, application-owned roots and explicit tools."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            msg = "invalid job identifier"
            raise ValueError(msg)
        self.job_id = job_id
        self.video = video.resolve()
        self.job_root = job_root.resolve()
        self.scene_cache_root = scene_cache_root.resolve()
        self.catalogue_path = catalogue_path.resolve()
        self.ffprobe = ffprobe
        self.commands = commands
        self.emit = emit
        self.resume = resume
        self.preset = reconstruction_preset(quality)
        self.cancelled = threading.Event()
        self.log_path = self.job_root / "reconstruction.log"
        self.runner = NativeProcessRunner(self.cancelled, self.log_path)
        self.store = JobManifestStore(self.job_root / "job.json")
        self.configuration = fingerprint_configuration(
            {
                "schema_version": 1,
                "candidate_limit": 180,
                "selected_minimum": 60,
                "quality": self.preset.quality.value,
                "selected_maximum": self.preset.selected_frames,
                "maximum_image_edge": self.preset.maximum_image_edge,
                "camera_model": "SIMPLE_RADIAL",
                "train_iterations": self.preset.brush_steps,
                "sog": "npm-lock",
            }
        )
        self.manifest: JobManifest | None = None

    def run(self) -> int:
        """Reach exactly one terminal event and preserve resumable state."""
        try:
            self._prepare_manifest()
            diagnostics = self._analyse()
            self._require_suitable(diagnostics)
            model = self._estimate_camera_poses(diagnostics.selected_frames)
            ply = self._train(model)
            scene = self._convert_and_publish(ply)
            self._set_manifest(status="completed", current_stage=None, scene_id=scene.asset_id)
            self.emit(
                ResultEvent(
                    job_id=self.job_id,
                    scene_id=scene.asset_id,
                    message=f"{scene.display_name} is ready.",
                )
            )
        except ReconstructionCancelledError:
            self._set_manifest(status="cancelled")
            self.emit(
                CancelledEvent(
                    job_id=self.job_id,
                    message="Room build cancelled; completed stages are available to resume.",
                )
            )
            return 130
        except (OSError, ValueError, NativeCommandError) as error:
            self._set_manifest(status="failed")
            self.emit(
                ErrorEvent(
                    job_id=self.job_id,
                    code=self._error_code(error),
                    message=self._safe_error_message(error),
                    recovery_action=self._recovery_action(error),
                    log_reference=str(self.log_path),
                )
            )
            return 1
        else:
            return 0

    def cancel(self) -> None:
        """Cooperatively cancel the worker and its current native process."""
        self.runner.cancel()

    def _prepare_manifest(self) -> None:
        if not self.video.is_file():
            msg = "The selected video no longer exists. Choose it again."
            raise ValueError(msg)
        self.job_root.mkdir(parents=True, exist_ok=True)
        existing = self.store.load() if self.resume else None
        if existing is not None and existing.can_resume(
            self.video, self.configuration, self.job_root
        ):
            self.manifest = existing.model_copy(update={"status": "running"})
        else:
            self.manifest = JobManifest(
                job_id=self.job_id,
                input_path=self.video,
                input_fingerprint=fingerprint_file(self.video),
                configuration_fingerprint=self.configuration,
            )
        self.store.save(self.manifest)

    def _analyse(self) -> CaptureDiagnostics:
        self._progress("validation", 0.02, "Checking video format and capture limits")
        probe = probe_video(self.video, self.ffprobe)
        if validate_probe(probe):
            return analyse_metrics(probe, (), ())
        extraction_fingerprint = fingerprint_configuration(
            {
                "input": self._manifest().input_fingerprint,
                "duration": probe.duration_seconds,
                "maximum_image_edge": self.preset.maximum_image_edge,
                "selected_maximum": self.preset.selected_frames,
                "version": 2,
            }
        )
        diagnostics_path = self.job_root / "analysis.json"
        selected_root = self.job_root / "images"
        if self._manifest().stage_is_current(
            "extract_frames",
            extraction_fingerprint,
            self.job_root,
        ):
            diagnostics = CaptureDiagnostics.model_validate_json(
                diagnostics_path.read_text(encoding="utf-8"),
            )
            self._progress("frame_selection", 0.24, "Reused verified frame analysis")
            return diagnostics
        candidate_root = self.job_root / "candidates"
        self._replace_directory(candidate_root)
        self._replace_directory(selected_root)
        sampling_rate = min(6.0, 100.0 / probe.duration_seconds)
        self._progress("frame_selection", 0.08, "Extracting timestamped candidate frames")
        self.runner.run(
            self.commands.extract_frames(
                self.video,
                candidate_root,
                fps=sampling_rate,
                maximum_image_edge=self.preset.maximum_image_edge,
            ),
            stage="frame extraction",
            timeout=10 * 60,
        )
        candidates = self._score_candidates(candidate_root, sampling_rate)
        selected = select_frames(candidates, maximum=self.preset.selected_frames)
        diagnostics = analyse_metrics(probe, candidates, selected)
        for warning in diagnostics.warnings:
            self.emit(
                WarningEvent(
                    job_id=self.job_id,
                    code="capture_quality",
                    message=warning,
                )
            )
        for output_index, metric in enumerate(selected):
            source = candidate_root / f"{metric.index:06d}.png"
            shutil.copyfile(source, selected_root / f"{output_index:06d}.png")
        diagnostics_path.write_text(diagnostics.model_dump_json(indent=2), encoding="utf-8")
        outputs = [diagnostics_path, *sorted(selected_root.glob("*.png"))]
        self._complete_stage("extract_frames", extraction_fingerprint, outputs)
        self._progress(
            "frame_selection",
            0.25,
            f"Selected {len(selected)} useful frames from {len(candidates)} candidates",
        )
        return diagnostics

    def _score_candidates(self, root: Path, fps: float) -> tuple[FrameMetrics, ...]:
        metrics: list[FrameMetrics] = []
        previous = None
        for path in sorted(root.glob("*.png")):
            self._check_cancelled()
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            frame_index = int(path.stem)
            metrics.append(
                score_frame(
                    image,
                    previous,
                    index=frame_index,
                    timestamp_seconds=frame_index / fps,
                )
            )
            previous = image
        return tuple(metrics)

    def _require_suitable(self, diagnostics: CaptureDiagnostics) -> None:
        if diagnostics.suitable:
            return
        issue = diagnostics.issues[0]
        msg = f"{issue.message} {issue.recovery_action}"
        raise ValueError(msg)

    def _estimate_camera_poses(self, selected_count: int) -> Path:
        stage_fingerprint = fingerprint_configuration(
            {
                "frames": self._manifest().stages["extract_frames"].fingerprint,
                "pycolmap": "4.1.0",
                "camera_model": "SIMPLE_RADIAL",
                "feature_backend": "sift-cpu",
            }
        )
        selected_root = self.job_root / "images"
        database = self.job_root / "database.db"
        sparse = self.job_root / "sparse"
        selected_model_path = self.job_root / "selected-model.txt"
        if self._manifest().stage_is_current("camera_estimation", stage_fingerprint, self.job_root):
            model = Path(selected_model_path.read_text(encoding="utf-8").strip())
            if model.is_dir():
                self._progress("camera_estimation", 0.56, "Reused verified camera reconstruction")
                return model
        database.unlink(missing_ok=True)
        self._replace_directory(sparse)
        self._progress("camera_estimation", 0.28, "Detecting image features")
        self.runner.run(
            self.commands.feature_extraction(selected_root, database),
            stage="COLMAP feature extraction",
            timeout=20 * 60,
        )
        self._progress("camera_estimation", 0.36, "Matching overlapping camera views")
        self.runner.run(
            self.commands.sequential_matching(database),
            stage="COLMAP sequential matching",
            timeout=20 * 60,
        )
        self._progress("camera_estimation", 0.46, "Estimating camera poses and sparse geometry")
        self.runner.run(
            self.commands.mapping(selected_root, database, sparse),
            stage="COLMAP mapping",
            timeout=30 * 60,
        )
        model = select_largest_model(sparse)
        if model is None:
            msg = "The capture did not produce a connected camera reconstruction."
            raise ValueError(msg)
        registered = registered_image_count(model)
        if (
            registered < MINIMUM_REGISTERED_IMAGES
            or registered / selected_count < MINIMUM_REGISTERED_PROPORTION
        ):
            msg = f"Only {registered} of {selected_count} frames registered into one model."
            raise ValueError(msg)
        text_model = self.job_root / "model-text"
        self._replace_directory(text_model)
        self.runner.run(
            self.commands.model_to_text(model, text_model),
            stage="COLMAP model validation export",
            timeout=5 * 60,
        )
        quality = analyse_colmap_model(text_model, selected_count)
        if quality.temporal_span < MINIMUM_TEMPORAL_SPAN:
            msg = "Registered cameras do not cover enough of the capture's timeline."
            raise ValueError(msg)
        if quality.longest_gap > MAXIMUM_CONTINUITY_GAP:
            msg = "The camera reconstruction fragments across a long part of the capture."
            raise ValueError(msg)
        if (
            quality.median_reprojection_error is not None
            and quality.median_reprojection_error > MAXIMUM_REPROJECTION_ERROR
        ):
            msg = "Camera reprojection error is too high for stable splat training."
            raise ValueError(msg)
        quality_path = self.job_root / "camera-quality.json"
        quality_path.write_text(quality.model_dump_json(indent=2), encoding="utf-8")
        selected_model_path.write_text(str(model), encoding="utf-8")
        model_outputs = [path for path in model.iterdir() if path.is_file()]
        text_outputs = [path for path in text_model.iterdir() if path.is_file()]
        self._complete_stage(
            "camera_estimation",
            stage_fingerprint,
            [selected_model_path, quality_path, *model_outputs, *text_outputs],
        )
        return model

    def _train(self, model: Path) -> Path:
        stage_fingerprint = fingerprint_configuration(
            {
                "camera": self._manifest().stages["camera_estimation"].fingerprint,
                "brush": "0.3.0",
                "iterations": self.preset.brush_steps,
            }
        )
        ply = self.job_root / "training" / "room.ply"
        if self._manifest().stage_is_current("training", stage_fingerprint, self.job_root):
            validate_ply(ply)
            self._progress("scene_training", 0.82, "Reused verified Brush training output")
            return ply
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.unlink(missing_ok=True)
        dataset = self.job_root / "dataset"
        self._prepare_brush_dataset(dataset, model)
        self._progress(
            "scene_training",
            None,
            (
                f"Training the {self.preset.quality.value} room splat "
                f"for {self.preset.brush_steps:,} steps"
            ),
        )
        self.runner.run(
            self.commands.brush_training(
                dataset,
                ply,
                iterations=self.preset.brush_steps,
            ),
            stage="Brush training",
            timeout=NATIVE_TIMEOUT_SECONDS,
        )
        validate_ply(ply)
        self._complete_stage("training", stage_fingerprint, [ply])
        return ply

    def _prepare_brush_dataset(self, dataset: Path, model: Path) -> None:
        self._replace_directory(dataset)
        shutil.copytree(self.job_root / "images", dataset / "images")
        sparse = dataset / "sparse" / "0"
        sparse.parent.mkdir(parents=True)
        shutil.copytree(model, sparse)

    def _convert_and_publish(self, ply: Path) -> SceneReference:
        stage_fingerprint = fingerprint_configuration(
            {
                "ply": fingerprint_file(ply),
                "splat_transform": "2.7.1",
                "gpu_policy": "automatic-with-cpu-fallback-v1",
            }
        )
        sog = self.job_root / "scene.sog"
        scene_source = self.job_root / "scene"
        scene_id = _scene_id(self.video, self._manifest().input_fingerprint)
        if not self._manifest().stage_is_current("conversion", stage_fingerprint, self.job_root):
            sog.unlink(missing_ok=True)
            shutil.rmtree(scene_source, ignore_errors=True)
            self._progress("runtime_conversion", 0.88, "Converting the trained splat to SOG")
            self._convert_sog(ply, sog)
            validate_sog(sog)
            safe_extract_archive(sog, scene_source, "zip")
            self._add_preview(scene_source)
            outputs = [sog, *[path for path in scene_source.rglob("*") if path.is_file()]]
            self._complete_stage("conversion", stage_fingerprint, outputs)
        scene = build_generated_scene_reference(scene_id, self.video, scene_source)
        self._progress("runtime_conversion", 0.96, "Verifying and registering the completed room")
        AssetInstaller(self.scene_cache_root).adopt(scene, scene_source)
        SceneCatalogue(self.catalogue_path).save(scene)
        return scene

    def _convert_sog(self, ply: Path, sog: Path) -> None:
        try:
            self.runner.run(
                self.commands.convert_sog(ply, sog),
                stage="GPU SOG conversion",
                timeout=20 * 60,
            )
        except NativeCommandError as error:
            if error.return_code is None:
                raise
            sog.unlink(missing_ok=True)
            self.emit(
                WarningEvent(
                    job_id=self.job_id,
                    code="gpu_fallback",
                    message="GPU conversion was unavailable; retrying SOG conversion on CPU.",
                )
            )
            self.runner.run(
                self.commands.convert_sog(ply, sog, gpu="cpu"),
                stage="CPU SOG conversion fallback",
                timeout=20 * 60,
            )

    def _add_preview(self, scene_root: Path) -> None:
        candidates = sorted((self.job_root / "images").glob("*.png"))
        if not candidates:
            return
        image = cv2.imread(str(candidates[len(candidates) // 2]), cv2.IMREAD_COLOR)
        if image is None:
            return
        preview = scene_root / "preview.webp"
        if not cv2.imwrite(str(preview), image, [cv2.IMWRITE_WEBP_QUALITY, 82]):
            msg = "Could not create the generated room preview"
            raise ValueError(msg)

    def _complete_stage(self, stage: str, fingerprint: str, outputs: Sequence[Path]) -> None:
        manifest = self._manifest()
        stages = dict(manifest.stages)
        stages[stage] = StageArtifact(
            fingerprint=fingerprint,
            outputs=relative_outputs(self.job_root, outputs),
        )
        self.manifest = manifest.model_copy(update={"stages": stages, "current_stage": stage})
        self.store.save(self.manifest)

    def _set_manifest(self, **changes: object) -> None:
        if self.manifest is None:
            return
        self.manifest = self.manifest.model_copy(update=changes)
        self.store.save(self.manifest)

    def _manifest(self) -> JobManifest:
        if self.manifest is None:
            msg = "job manifest has not been initialized"
            raise RuntimeError(msg)
        return self.manifest

    def _progress(self, stage: str, progress: float | None, message: str) -> None:
        self._check_cancelled()
        self._set_manifest(current_stage=stage)
        self.emit(
            ProgressEvent(
                job_id=self.job_id,
                stage=stage,
                progress=progress,
                message=message,
            )
        )

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise ReconstructionCancelledError

    @staticmethod
    def _replace_directory(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

    @staticmethod
    def _error_code(error: Exception) -> str:
        if isinstance(error, NativeCommandError):
            return "native_tool_failed"
        text = str(error).lower()
        if "video" in text or "capture" in text or "frame" in text:
            return "unsuitable_capture"
        return "reconstruction_failed"

    @staticmethod
    def _safe_error_message(error: Exception) -> str:
        if isinstance(error, NativeCommandError):
            return f"The native {error.stage} stage could not complete."
        return str(error)[:500]

    @staticmethod
    def _recovery_action(error: Exception) -> str:
        if isinstance(error, NativeCommandError):
            return "Run doctor, review the build log, then retry the saved job."
        return "Review the capture guidance, then retry or choose another video."


def _scene_id(video: Path, input_fingerprint: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-") or "room"
    return f"{stem[:48]}-{input_fingerprint[:10]}"


def build_generated_scene_reference(
    scene_id: str,
    video: Path,
    root: Path,
) -> SceneReference:
    """Describe one generated COLMAP/Brush scene in renderer coordinates."""
    resources = tuple(
        AssetResource(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=fingerprint_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    preview = "preview.webp" if (root / "preview.webp").is_file() else None
    return normalize_colmap_scene_reference(
        SceneReference(
            asset_id=scene_id,
            display_name=video.stem.replace("_", " ").strip().title(),
            format="sog",
            entrypoint="meta.json",
            resources=resources,
            license_name="User-provided capture",
            attribution=f"Generated locally from {video.name}",
            preview=preview,
        )
    )


def emit_stdout(event: JobEvent) -> None:
    """Write one validated NDJSON event to the supervisor."""
    sys.stdout.write(event.model_dump_json() + "\n")
    sys.stdout.flush()


def watch_control(worker: ReconstructionWorker) -> threading.Thread:
    """Watch stdin without blocking native work and accept only this job's cancel."""

    def read() -> None:
        for line in sys.stdin:
            try:
                control = CancelControl.model_validate_json(line)
            except ValidationError:
                continue
            if control.job_id == worker.job_id:
                worker.cancel()
                return

    thread = threading.Thread(target=read, name="reconstruction-control", daemon=True)
    thread.start()
    return thread
