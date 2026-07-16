"""Supervise one versioned NDJSON worker process at a time."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

from better_backgrounds.ndjson import NdjsonDecodeError, NdjsonDecoder
from better_backgrounds.protocol import (
    CancelControl,
    CancelledEvent,
    ErrorEvent,
    JobEvent,
    is_terminal,
    parse_event_json,
)

if TYPE_CHECKING:
    from io import BufferedReader

EventCallback = Callable[[JobEvent], None]


class JobAlreadyRunningError(RuntimeError):
    """Raised when a runner is asked to own two processes."""


class JobRunner:
    """Own worker lifetime, protocol parsing, cancellation, and stderr bounds."""

    def __init__(
        self,
        on_event: EventCallback,
        *,
        cancellation_grace_seconds: float = 1.0,
        max_stderr_bytes: int = 64 * 1024,
    ) -> None:
        """Configure callbacks and hard resource bounds without starting work."""
        self._on_event = on_event
        self._grace_seconds = cancellation_grace_seconds
        self._max_stderr_bytes = max_stderr_bytes
        self._lock = threading.RLock()
        self._finished = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        self._terminal_sent = False
        self._stderr = bytearray()
        self._cancel_timer: threading.Timer | None = None

    @property
    def running(self) -> bool:
        """Return whether the owned process is still active."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def stderr_text(self) -> str:
        """Return the bounded stderr tail for diagnostics."""
        with self._lock:
            return bytes(self._stderr).decode("utf-8", errors="replace")

    def start(self, command: Sequence[str], *, job_id: str) -> None:
        """Start a worker with isolated process-group ownership."""
        with self._lock:
            if self.running:
                msg = "This runner already owns an active job."
                raise JobAlreadyRunningError(msg)
            self._finished.clear()
            self._job_id = job_id
            self._terminal_sent = False
            self._stderr.clear()
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self._process = subprocess.Popen(  # noqa: S603
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=creation_flags,
            )
            process = self._process

        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process,),
            name=f"job-stdout-{job_id}",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process,),
            name=f"job-stderr-{job_id}",
            daemon=True,
        )
        monitor_thread = threading.Thread(
            target=self._monitor,
            args=(process, stdout_thread, stderr_thread),
            name=f"job-monitor-{job_id}",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        monitor_thread.start()

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation, then enforce a bounded fallback."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or job_id != self._job_id:
                return False
            control = CancelControl(job_id=job_id).model_dump_json().encode("utf-8") + b"\n"
            if process.stdin is not None:
                with suppress(BrokenPipeError, OSError):
                    process.stdin.write(control)
                    process.stdin.flush()
            timer = threading.Timer(self._grace_seconds, self._force_cancel, args=(process, job_id))
            timer.daemon = True
            self._cancel_timer = timer
            timer.start()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until process cleanup and terminal event handling finish."""
        return self._finished.wait(timeout)

    def close(self) -> None:
        """Stop the active worker before releasing the runner."""
        with self._lock:
            job_id = self._job_id
            process = self._process
        if job_id is not None and process is not None and process.poll() is None:
            self.cancel(job_id)
            if not self.wait(self._grace_seconds + 1.0):
                self._terminate_tree(process)
                process.wait(timeout=2.0)

    def _read_stdout(self, process: subprocess.Popen[bytes]) -> None:
        decoder = NdjsonDecoder()
        stream = cast("BufferedReader | None", process.stdout)
        if stream is None:
            return
        try:
            while chunk := stream.read1(4096):
                for line in decoder.feed(chunk):
                    self._handle_line(line)
            for line in decoder.finish():
                self._handle_line(line)
        except (NdjsonDecodeError, ValidationError, ValueError) as error:
            self._protocol_failure(str(error), process)

    def _read_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stream = cast("BufferedReader | None", process.stderr)
        if stream is None:
            return
        while chunk := stream.read1(4096):
            with self._lock:
                self._stderr.extend(chunk)
                overflow = len(self._stderr) - self._max_stderr_bytes
                if overflow > 0:
                    del self._stderr[:overflow]

    def _handle_line(self, line: str) -> None:
        event = parse_event_json(line)
        with self._lock:
            if event.job_id != self._job_id:
                msg = "Worker emitted an event for an unexpected job."
                raise ValueError(msg)
        self._emit(event)

    def _emit(self, event: JobEvent) -> None:
        with self._lock:
            if self._terminal_sent:
                return
            if is_terminal(event):
                self._terminal_sent = True
                if self._cancel_timer is not None:
                    self._cancel_timer.cancel()
        self._on_event(event)

    def _protocol_failure(self, detail: str, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            job_id = self._job_id
        if job_id is None:
            return
        self._emit(
            ErrorEvent(
                job_id=job_id,
                code="invalid_worker_protocol",
                message="The worker returned invalid progress data.",
                recovery_action="Retry the operation and inspect its job log if it repeats.",
                log_reference=detail[:500],
            ),
        )
        self._terminate_tree(process)

    def _monitor(
        self,
        process: subprocess.Popen[bytes],
        stdout_thread: threading.Thread,
        stderr_thread: threading.Thread,
    ) -> None:
        return_code = process.wait()
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        with self._lock:
            job_id = self._job_id
            terminal_sent = self._terminal_sent
        if job_id is not None and not terminal_sent:
            self._emit(
                ErrorEvent(
                    job_id=job_id,
                    code="unexpected_worker_exit",
                    message=f"The worker stopped unexpectedly with exit code {return_code}.",
                    recovery_action="Retry the operation and inspect its job log if it repeats.",
                    log_reference=self.stderr_text[-500:] or None,
                ),
            )
        with self._lock:
            if self._process is process:
                self._process = None
        self._finished.set()

    def _force_cancel(self, process: subprocess.Popen[bytes], job_id: str) -> None:
        if process.poll() is not None:
            return
        self._emit(
            CancelledEvent(
                job_id=job_id,
                message="Build stopped after the cancellation grace period.",
                forced=True,
            ),
        )
        self._terminate_tree(process)

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],  # noqa: S607
                check=False,
                capture_output=True,
            )
            if process.poll() is None:
                process.kill()
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
