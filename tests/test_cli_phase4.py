"""Phase 4 command-surface tests."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from better_backgrounds import cli
from better_backgrounds.protocol import ErrorEvent, parse_event_json

if TYPE_CHECKING:
    import pytest

MISSING_TOOLS_EXIT_CODE = 2


def test_doctor_reports_sample_mode_when_native_tools_are_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the fallback usable while honestly reporting reconstruction support."""
    monkeypatch.setattr(
        cli,
        "_application_roots",
        lambda: (tmp_path / "cache", tmp_path / "data"),
    )

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["sample_mode_supported"]
    assert not report["reconstruction_supported"]


def test_reconstruct_missing_tools_emits_one_protocol_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail through the shared NDJSON contract before creating a job."""
    video = tmp_path / "room.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        cli,
        "_application_roots",
        lambda: (tmp_path / "cache", tmp_path / "data"),
    )

    result = CliRunner().invoke(cli.app, ["reconstruct", str(video), "--job-id", "job-1"])

    assert result.exit_code == MISSING_TOOLS_EXIT_CODE
    event = parse_event_json(result.stdout.strip())
    assert isinstance(event, ErrorEvent)
    assert event.code == "tools_unavailable"
    assert not (tmp_path / "data" / "jobs-v1" / "job-1").exists()
