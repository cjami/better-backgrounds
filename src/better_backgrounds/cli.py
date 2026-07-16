"""Developer and worker commands for Better Backgrounds."""

from __future__ import annotations

from typing import Annotated

import typer

from better_backgrounds.fake_worker import FakeOutcome, run_fake_job

app = typer.Typer(
    name="better-backgrounds",
    help="Run Better Backgrounds desktop and diagnostic commands.",
    no_args_is_help=True,
)


@app.command("desktop")
def desktop_command() -> None:
    """Open the Python-owned desktop application."""
    from better_backgrounds.desktop.app import main  # noqa: PLC0415

    raise typer.Exit(main())


@app.command("fake-job", hidden=True)
def fake_job_command(
    job_id: Annotated[str, typer.Option(help="Stable job identifier.")],
    outcome: Annotated[FakeOutcome, typer.Option()] = "success",
    delay: Annotated[float, typer.Option(min=0.0, max=5.0)] = 0.08,
) -> None:
    """Run the deterministic Phase 2 protocol worker."""
    raise typer.Exit(run_fake_job(job_id, outcome=outcome, delay=delay))


if __name__ == "__main__":
    app()
