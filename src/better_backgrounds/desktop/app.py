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


def main() -> int:
    """Run the desktop shell or its packaged fake-worker mode."""
    arguments = sys.argv[1:]
    if "--fake-worker" in arguments:
        return _run_packaged_worker(arguments)

    application = cast("QApplication | None", QApplication.instance()) or QApplication(sys.argv)
    application.setApplicationName("Better Backgrounds")
    application.setOrganizationName("Better Backgrounds")
    application.setWindowIcon(application_icon())
    application.setStyleSheet(STYLESHEET)
    window = MainWindow(command_factory=packaged_worker_command)
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
