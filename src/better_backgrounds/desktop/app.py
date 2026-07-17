"""Qt application entry point and packaged-worker dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.main_window import MainWindow, development_worker_command
from better_backgrounds.desktop.theme import STYLESHEET
from better_backgrounds.fake_worker import FakeOutcome, run_fake_job
from better_backgrounds.reconstruction import ReconstructionQuality

if TYPE_CHECKING:
    from collections.abc import Sequence


def packaged_worker_command(job_id: str, outcome: str) -> list[str]:
    """Choose a command that also works when the GUI is frozen."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return [
            str(Path(sys.argv[0]).resolve()),
            "--fake-worker",
            "--job-id",
            job_id,
            "--outcome",
            outcome,
        ]
    return development_worker_command(job_id, outcome)


def packaged_reconstruction_command(
    job_id: str,
    video: Path,
    resume: bool,
    quality: ReconstructionQuality | str,
) -> list[str]:
    """Choose a real worker command that also works in a frozen application."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        command = [
            str(Path(sys.argv[0]).resolve()),
            "--reconstruction-worker",
            "--video",
            str(video),
            "--job-id",
            job_id,
            "--quality",
            ReconstructionQuality(quality).value,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "better_backgrounds.cli",
            "reconstruct",
            str(video),
            "--job-id",
            job_id,
            "--quality",
            ReconstructionQuality(quality).value,
        ]
    if resume:
        command.append("--resume")
    return command


def _run_packaged_worker(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fake-worker", action="store_true")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--outcome",
        choices=("success", "failure", "forced", "malformed", "unexpected-exit"),
        default="success",
    )
    values = parser.parse_args(arguments)
    outcome = cast("FakeOutcome", values.outcome)
    job_id = cast("str", values.job_id)
    return run_fake_job(job_id, outcome=outcome)


def _run_packaged_reconstruction(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reconstruction-worker", action="store_true")
    parser.add_argument("--video", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--quality",
        choices=tuple(item.value for item in ReconstructionQuality),
        default=ReconstructionQuality.BALANCED.value,
    )
    values = parser.parse_args(arguments)
    from better_backgrounds.cli import app  # noqa: PLC0415

    cli_arguments = [
        "reconstruct",
        values.video,
        "--job-id",
        values.job_id,
        "--quality",
        values.quality,
    ]
    if values.resume:
        cli_arguments.append("--resume")
    try:
        app(cli_arguments, standalone_mode=False)
    except SystemExit as error:
        return int(error.code or 0)
    return 0


def _run_packaged_pycolmap(arguments: Sequence[str]) -> int:
    from better_backgrounds.pycolmap_worker import main as pycolmap_main  # noqa: PLC0415

    return pycolmap_main(arguments)


def main() -> int:
    """Run the desktop shell or its packaged fake-worker mode."""
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--pycolmap-worker":
        return _run_packaged_pycolmap(arguments[1:])
    if "--reconstruction-worker" in arguments:
        return _run_packaged_reconstruction(arguments)
    if "--fake-worker" in arguments:
        return _run_packaged_worker(arguments)

    application = cast("QApplication | None", QApplication.instance()) or QApplication(sys.argv)
    application.setApplicationName("Better Backgrounds")
    application.setOrganizationName("Better Backgrounds")
    application.setWindowIcon(application_icon())
    application.setStyleSheet(STYLESHEET)
    window = MainWindow(
        command_factory=packaged_worker_command,
        reconstruction_command_factory=packaged_reconstruction_command,
    )
    window.showMaximized()
    if "--build-smoke-test" in arguments:
        window.room_ready.connect(window.close)
        window.room_ready.connect(application.quit)
        QTimer.singleShot(0, window.start_smoke_build)
        QTimer.singleShot(10_000, lambda: application.exit(2))
    elif "--smoke-test" in arguments:
        QTimer.singleShot(250, window.close)
        QTimer.singleShot(300, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
