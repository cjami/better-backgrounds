"""Deterministic worker used to exercise desktop process behavior."""

from __future__ import annotations

import sys
import threading
import time
from typing import IO, Literal

from pydantic import ValidationError

from better_backgrounds.protocol import (
    CancelControl,
    CancelledEvent,
    ErrorEvent,
    ProgressEvent,
    ResultEvent,
    WarningEvent,
    WireModel,
)

FakeOutcome = Literal["success", "failure", "forced", "malformed", "unexpected-exit"]
STAGES = (
    ("validation", "Validating room video"),
    ("frame_selection", "Selecting sharp, well-spaced frames"),
    ("camera_estimation", "Estimating camera poses"),
    ("scene_training", "Training the spatial scene"),
    ("runtime_conversion", "Preparing the runtime scene"),
)


def _write_event(stream: IO[str], event: WireModel) -> None:
    stream.write(event.model_dump_json())
    stream.write("\n")
    stream.flush()


def _listen_for_cancel(job_id: str, cancelled: threading.Event, stream: IO[str]) -> None:
    for line in stream:
        try:
            control = CancelControl.model_validate_json(line)
        except ValidationError:
            continue
        if control.job_id == job_id:
            cancelled.set()
            return


def run_fake_job(
    job_id: str,
    *,
    outcome: FakeOutcome = "success",
    delay: float = 0.08,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Run a deterministic fake job and return a process-style exit code."""
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    if outcome == "malformed":
        output_stream.write("{not valid json}\n")
        output_stream.flush()
        return 0
    if outcome == "unexpected-exit":
        return 3

    cancelled = threading.Event()
    listener = threading.Thread(
        target=_listen_for_cancel,
        args=(job_id, cancelled, input_stream),
        name=f"fake-cancel-{job_id}",
        daemon=True,
    )
    listener.start()

    total_ticks = len(STAGES) * 2
    for index, (stage, message) in enumerate(STAGES):
        for tick in range(2):
            if cancelled.is_set() and outcome != "forced":
                _write_event(
                    output_stream,
                    CancelledEvent(job_id=job_id, message="Build cancelled safely."),
                )
                return 0
            progress = (index * 2 + tick + 1) / total_ticks
            _write_event(
                output_stream,
                ProgressEvent(
                    job_id=job_id,
                    stage=stage,
                    progress=progress,
                    message=message,
                ),
            )
            time.sleep(delay)
        if stage == "frame_selection":
            _write_event(
                output_stream,
                WarningEvent(
                    job_id=job_id,
                    code="minor_motion",
                    message="Minor curtain movement will be down-weighted.",
                ),
            )
        if outcome == "failure" and stage == "camera_estimation":
            _write_event(
                output_stream,
                ErrorEvent(
                    job_id=job_id,
                    code="fake_camera_estimation_failed",
                    message="The fake camera-estimation stage failed as requested.",
                    recovery_action="Choose Retry or select a different video.",
                    log_reference=f"jobs/{job_id}/logs/fake.log",
                ),
            )
            return 2

    if outcome == "forced":
        while True:
            time.sleep(1.0)

    _write_event(
        output_stream,
        ResultEvent(
            job_id=job_id,
            scene_id=f"scene-{job_id}",
            message="Prepared scene is ready for viewpoint adjustment.",
        ),
    )
    return 0
