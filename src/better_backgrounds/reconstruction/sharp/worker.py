"""NDJSON worker entry points for SHARP checkpoint preparation and scene builds."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import ValidationError

from better_backgrounds.jobs.events import (
    CancelControl,
    CancelledEvent,
    ErrorEvent,
    JobEvent,
    ProgressEvent,
    ResultEvent,
)
from better_backgrounds.reconstruction.sharp import (
    SceneBuildRequest,
    SceneImageSelection,
    SharpBuildConfig,
    SharpCancelledError,
    SharpCheckpointInstaller,
    SharpSceneBuilder,
)
from better_backgrounds.scene import SceneCatalogue

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import TextIO

    from better_backgrounds.reconstruction.sharp.runtime import SharpDeviceRequest

EventEmitter = Callable[[JobEvent], None]
ERROR_INVALID_HANDLE = 6


def emit_stdout(event: JobEvent) -> None:
    """Write one validated NDJSON event to the supervisor."""
    sys.stdout.write(event.model_dump_json() + "\n")
    sys.stdout.flush()


class SharpBuildWorker:
    """Own one cancellable image-to-scene build inside a dedicated process."""

    def __init__(
        self,
        *,
        job_id: str,
        image: Path,
        source_kind: str,
        device: SharpDeviceRequest,
        checkpoint_path: Path,
        scene_cache_root: Path,
        catalogue_path: Path,
        emit: EventEmitter = emit_stdout,
        builder: SharpSceneBuilder | None = None,
    ) -> None:
        """Keep worker dependencies serializable and independently testable."""
        self.job_id = job_id
        self.image = image
        self.source_kind = source_kind
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.scene_cache_root = scene_cache_root
        self.catalogue_path = catalogue_path
        self.emit = emit
        self.builder = builder or SharpSceneBuilder()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        self._cancelled.set()

    def run(self) -> int:
        """Build and catalogue one room, emitting exactly one terminal event."""
        try:
            source_kind = "camera" if self.source_kind == "camera" else "upload"
            reference = self.builder.build(
                SceneBuildRequest(
                    job_id=self.job_id,
                    selection=SceneImageSelection(
                        display_name=self.image.name,
                        source_path=self.image,
                        source_kind=source_kind,
                    ),
                    config=SharpBuildConfig(
                        device=self.device,
                        checkpoint_path=self.checkpoint_path,
                        output_root=self.scene_cache_root,
                    ),
                ),
                self.emit,
                self._cancelled.is_set,
            )
            SceneCatalogue(self.catalogue_path).save(reference)
        except SharpCancelledError:
            self.emit(
                CancelledEvent(job_id=self.job_id, message="SHARP room build cancelled."),
            )
            return 0
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.emit(self._error_event(error))
            return 1
        self.emit(
            ResultEvent(
                job_id=self.job_id,
                scene_id=reference.asset_id,
                message="SHARP room is ready.",
            ),
        )
        return 0

    def _error_event(self, error: Exception) -> ErrorEvent:
        text = str(error)
        lowered = text.lower()
        if "checkpoint" in lowered or "license" in lowered:
            code = "sharp_checkpoint_failed"
            recovery = "Prepare the pinned SHARP checkpoint, then retry."
        elif "image" in lowered or "jpeg" in lowered or "png" in lowered or "webp" in lowered:
            code = "sharp_image_invalid"
            recovery = "Choose another JPEG, PNG, or WebP room image."
        elif "ply" in lowered or "gaussian" in lowered or "intrinsic" in lowered:
            code = "sharp_output_invalid"
            recovery = "Retry once, then review the SHARP compatibility report."
        else:
            code = "sharp_build_failed"
            recovery = "Review the worker log and runtime compatibility, then retry."
        return ErrorEvent(
            job_id=self.job_id,
            code=code,
            message=text[:500] or "SHARP could not build this room.",
            recovery_action=recovery,
        )


class SharpCheckpointWorker:
    """Prepare the license-gated checkpoint through the shared worker protocol."""

    def __init__(
        self,
        *,
        job_id: str,
        model_root: Path,
        license_accepted: bool,
        emit: EventEmitter = emit_stdout,
        installer: SharpCheckpointInstaller | None = None,
    ) -> None:
        """Configure one bounded model-preparation job."""
        self.job_id = job_id
        self.emit = emit
        self.installer = installer or SharpCheckpointInstaller(model_root)
        self.license_accepted = license_accepted
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation during the streamed download."""
        self._cancelled.set()

    def run(self) -> int:
        """Download and verify the checkpoint, emitting one terminal event."""
        try:
            self.emit(
                ProgressEvent(
                    job_id=self.job_id,
                    stage="model_preparation",
                    progress=0.0,
                    message="Preparing the pinned SHARP checkpoint",
                ),
            )
            self.installer.prepare(
                license_accepted=self.license_accepted,
                progress=self._progress,
                is_cancelled=self._cancelled.is_set,
            )
        except SharpCancelledError:
            self.emit(
                CancelledEvent(
                    job_id=self.job_id,
                    message="SHARP checkpoint preparation cancelled.",
                ),
            )
            return 0
        except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
            self.emit(
                ErrorEvent(
                    job_id=self.job_id,
                    code="sharp_checkpoint_failed",
                    message=str(error)[:500],
                    recovery_action="Check the license choice, connection, and free disk space.",
                ),
            )
            return 1
        self.emit(
            ResultEvent(
                job_id=self.job_id,
                scene_id="sharp-checkpoint",
                message="SHARP checkpoint is ready offline.",
            ),
        )
        return 0

    def _progress(self, completed: int, total: int) -> None:
        self.emit(
            ProgressEvent(
                job_id=self.job_id,
                stage="model_preparation",
                progress=completed / total,
                message=f"Downloading SHARP checkpoint ({completed / 1024**3:.1f} GiB)",
            ),
        )


def _windows_pipe_lines(stream: TextIO) -> Iterator[str]:
    """Poll redirected stdin without holding a blocking Windows pipe read."""
    import msvcrt  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    peek.restype = wintypes.BOOL
    pending = bytearray()
    while True:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_HANDLE:
                yield from stream
            return
        if available.value == 0:
            time.sleep(0.02)
            continue
        chunk = os.read(stream.fileno(), available.value)
        if not chunk:
            return
        pending.extend(chunk)
        while b"\n" in pending:
            raw_line, _, remainder = pending.partition(b"\n")
            pending = bytearray(remainder)
            yield raw_line.rstrip(b"\r").decode("utf-8", errors="replace")


def _control_lines(stream: TextIO) -> Iterator[str]:
    if os.name == "nt" and not stream.isatty():
        yield from _windows_pipe_lines(stream)
        return
    yield from stream


def watch_control(
    job_id: str,
    cancel: Callable[[], None],
    *,
    stdin: TextIO | None = None,
) -> threading.Thread:
    """Accept only a matching versioned cancellation command from stdin."""
    input_stream = stdin or sys.stdin

    def read() -> None:
        for line in _control_lines(input_stream):
            try:
                control = CancelControl.model_validate_json(line)
            except ValidationError:
                continue
            if control.job_id == job_id:
                cancel()
                return

    thread = threading.Thread(target=read, name="sharp-control", daemon=True)
    thread.start()
    return thread
