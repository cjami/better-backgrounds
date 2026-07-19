"""Qt application entry point and packaged-worker dispatch."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.icon import application_icon
from better_backgrounds.desktop.startup import StartupCoordinator
from better_backgrounds.desktop.theme import STYLESHEET
from better_backgrounds.jobs.fake_worker import FakeOutcome, run_fake_job

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
    from better_backgrounds.desktop.main_window import development_worker_command  # noqa: PLC0415

    return development_worker_command(job_id, outcome)


def packaged_sharp_command(
    job_id: str,
    image: Path,
    device: str,
    source_kind: str,
) -> list[str]:
    """Choose a SHARP worker command that also works when frozen."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return [
            str(Path(sys.argv[0]).resolve()),
            "--sharp-worker",
            "--image",
            str(image),
            "--job-id",
            job_id,
            "--device",
            device,
            "--source-kind",
            source_kind,
        ]
    return [
        sys.executable,
        "-m",
        "better_backgrounds.cli",
        "sharp-build",
        str(image),
        "--job-id",
        job_id,
        "--device",
        device,
        "--source-kind",
        source_kind,
    ]


def packaged_sharp_prepare_command(job_id: str) -> list[str]:
    """Choose the managed checkpoint worker command for this runtime."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return [
            str(Path(sys.argv[0]).resolve()),
            "--sharp-prepare-worker",
            "--job-id",
            job_id,
            "--accept-model-license",
        ]
    return [
        sys.executable,
        "-m",
        "better_backgrounds.cli",
        "prepare-sharp",
        "--job-id",
        job_id,
        "--accept-model-license",
    ]


def packaged_splat_command(job_id: str, source: Path) -> list[str]:
    """Choose a direct-import worker command that also works when frozen."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return [
            str(Path(sys.argv[0]).resolve()),
            "--splat-worker",
            "--source",
            str(source),
            "--job-id",
            job_id,
        ]
    return [
        sys.executable,
        "-m",
        "better_backgrounds.cli",
        "splat-import",
        str(source),
        "--job-id",
        job_id,
    ]


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


def _run_packaged_sharp(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sharp-worker", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--source-kind", choices=("upload", "camera"), default="upload")
    values = parser.parse_args(arguments)
    from better_backgrounds.cli import app  # noqa: PLC0415

    cli_arguments = [
        "sharp-build",
        values.image,
        "--job-id",
        values.job_id,
        "--device",
        values.device,
        "--source-kind",
        values.source_kind,
    ]
    try:
        app(cli_arguments, standalone_mode=False)
    except SystemExit as error:
        return int(error.code or 0)
    return 0


def _run_packaged_sharp_prepare(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sharp-prepare-worker", action="store_true")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--accept-model-license", action="store_true")
    values = parser.parse_args(arguments)
    from better_backgrounds.cli import app  # noqa: PLC0415

    cli_arguments = ["prepare-sharp", "--job-id", values.job_id]
    if values.accept_model_license:
        cli_arguments.append("--accept-model-license")
    try:
        app(cli_arguments, standalone_mode=False)
    except SystemExit as error:
        return int(error.code or 0)
    return 0


def _run_packaged_splat(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--splat-worker", action="store_true")
    parser.add_argument("--source", required=True)
    parser.add_argument("--job-id", required=True)
    values = parser.parse_args(arguments)
    from better_backgrounds.cli import app  # noqa: PLC0415

    try:
        app(
            ["splat-import", values.source, "--job-id", values.job_id],
            standalone_mode=False,
        )
    except SystemExit as error:
        return int(error.code or 0)
    return 0


def main() -> int:
    """Run the desktop shell or its packaged fake-worker mode."""
    arguments = sys.argv[1:]
    if "--sharp-worker" in arguments:
        return _run_packaged_sharp(arguments)
    if "--sharp-prepare-worker" in arguments:
        return _run_packaged_sharp_prepare(arguments)
    if "--splat-worker" in arguments:
        return _run_packaged_splat(arguments)
    if "--fake-worker" in arguments:
        return _run_packaged_worker(arguments)

    application = cast("QApplication | None", QApplication.instance()) or QApplication(sys.argv)
    application.setApplicationName("Better Backgrounds")
    application.setOrganizationName("Better Backgrounds")
    application.setWindowIcon(application_icon())
    application.setStyleSheet(STYLESHEET)
    startup = StartupCoordinator(application)
    startup.show()
    startup.advance("preferences", "Loading preferences and rooms")
    from better_backgrounds.desktop.main_window import MainWindow  # noqa: PLC0415

    startup.advance("matting", "Preparing background removal")
    window_factory = partial(
        MainWindow,
        command_factory=packaged_worker_command,
        sharp_command_factory=packaged_sharp_command,
        sharp_prepare_command_factory=packaged_sharp_prepare_command,
        splat_command_factory=packaged_splat_command,
    )
    if "--smoke-test" in arguments:
        from PySide6.QtWidgets import QWidget  # noqa: PLC0415

        from better_backgrounds.desktop.camera import InputCameraSource  # noqa: PLC0415
        from better_backgrounds.desktop.live_preview.preview import (  # noqa: PLC0415
            NativeLivePreview,
        )
        from better_backgrounds.desktop.preview import ScenePreview  # noqa: PLC0415

        window = window_factory(
            renderer_factory=ScenePreview,
            live_renderer_factory=lambda: NativeLivePreview(background_factory=QWidget),
            camera_source=InputCameraSource(lambda: ()),
        )
    else:
        window = window_factory()
    window.showMaximized()
    startup.finish(window)
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
