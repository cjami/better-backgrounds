"""Feature-first: Integration tests for fake worker supervision."""

import sys
import threading

from better_backgrounds.jobs.events import CancelledEvent, ErrorEvent, JobEvent, ResultEvent
from better_backgrounds.jobs.runner import JobRunner

STDERR_LIMIT = 32


def worker_command(job_id: str, outcome: str) -> list[str]:
    """Build a command targeting the active test interpreter."""
    return [
        sys.executable,
        "-m",
        "better_backgrounds.cli",
        "fake-job",
        "--job-id",
        job_id,
        "--outcome",
        outcome,
        "--delay",
        "0.01",
    ]


def test_success_emits_exactly_one_terminal_event() -> None:
    """Parse progress and preserve the worker's successful result."""
    events: list[JobEvent] = []
    runner = JobRunner(events.append)

    runner.start(worker_command("success", "success"), job_id="success")

    assert runner.wait(5.0)
    terminals = [event for event in events if isinstance(event, ResultEvent | ErrorEvent)]
    assert len(terminals) == 1
    assert isinstance(terminals[0], ResultEvent)


def test_cooperative_cancellation_emits_cancelled() -> None:
    """Send a versioned stdin control message before using force."""
    events: list[JobEvent] = []
    progressed = threading.Event()

    def collect(event: JobEvent) -> None:
        events.append(event)
        progressed.set()

    runner = JobRunner(collect, cancellation_grace_seconds=1.0)
    runner.start(worker_command("cancel", "success"), job_id="cancel")
    assert progressed.wait(2.0)

    assert runner.cancel("cancel")
    assert runner.wait(5.0)
    terminal = [event for event in events if isinstance(event, CancelledEvent | ErrorEvent)]
    assert len(terminal) == 1
    assert isinstance(terminal[0], CancelledEvent)
    assert not terminal[0].forced


def test_forced_cancellation_is_bounded() -> None:
    """Terminate a worker that deliberately ignores cooperative cancellation."""
    events: list[JobEvent] = []
    progressed = threading.Event()

    def collect(event: JobEvent) -> None:
        events.append(event)
        progressed.set()

    runner = JobRunner(collect, cancellation_grace_seconds=0.1)
    runner.start(worker_command("forced", "forced"), job_id="forced")
    assert progressed.wait(2.0)

    assert runner.cancel("forced")
    assert runner.wait(5.0)
    terminal = [event for event in events if isinstance(event, CancelledEvent | ErrorEvent)]
    assert len(terminal) == 1
    assert isinstance(terminal[0], CancelledEvent)
    assert terminal[0].forced


def test_malformed_output_becomes_safe_error() -> None:
    """Convert malformed worker output into one renderer-safe failure."""
    events: list[JobEvent] = []
    runner = JobRunner(events.append)

    runner.start(worker_command("malformed", "malformed"), job_id="malformed")

    assert runner.wait(5.0)
    terminal = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(terminal) == 1
    assert terminal[0].code == "invalid_worker_protocol"


def test_unexpected_exit_becomes_safe_error() -> None:
    """Synthesize one failure when a worker exits without a terminal event."""
    events: list[JobEvent] = []
    runner = JobRunner(events.append)

    runner.start(worker_command("exit", "unexpected-exit"), job_id="exit")

    assert runner.wait(5.0)
    terminal = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(terminal) == 1
    assert terminal[0].code == "unexpected_worker_exit"


def test_stderr_retention_is_bounded() -> None:
    """Drain verbose stderr without allowing unbounded retained logs."""
    events: list[JobEvent] = []
    runner = JobRunner(events.append, max_stderr_bytes=STDERR_LIMIT)
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('x' * 1000); sys.stderr.flush()",
    ]

    runner.start(command, job_id="stderr")

    assert runner.wait(5.0)
    assert len(runner.stderr_text.encode()) <= STDERR_LIMIT
