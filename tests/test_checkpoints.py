"""Feature-first: Verify managed checkpoint download, integrity, and publication."""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import pytest

from better_backgrounds.checkpoints import (
    CheckpointIdentity,
    ManagedCheckpointInstaller,
)

if TYPE_CHECKING:
    from pathlib import Path


class CancelledError(RuntimeError):
    """Signal cooperative cancellation for the installer under test."""


def identity(payload: bytes) -> CheckpointIdentity:
    """Pin one compact checkpoint identity for boundary tests."""
    return CheckpointIdentity(
        filename="model.pth",
        url="https://example.invalid/model.pth",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def installer(
    root: Path,
    payload: bytes,
    *,
    served: bytes | None = None,
) -> ManagedCheckpointInstaller:
    """Build an installer whose download boundary returns fixed bytes."""
    return ManagedCheckpointInstaller(
        root,
        identity(payload),
        opener=lambda _url: io.BytesIO(payload if served is None else served),
    )


def test_download_verifies_and_publishes_atomically(tmp_path: Path) -> None:
    """Publish only fully downloaded bytes that match the pinned identity."""
    payload = b"pinned model weights"
    managed = installer(tmp_path, payload)
    progress: list[tuple[int, int]] = []

    result = managed.prepare(progress=lambda done, total: progress.append((done, total)))

    assert result.read_bytes() == payload
    assert managed.is_ready()
    assert progress[-1] == (len(payload), len(payload))
    assert not list(tmp_path.glob("*.part"))


def test_checksum_mismatch_publishes_nothing(tmp_path: Path) -> None:
    """Never retain or publish bytes that fail the pinned SHA-256."""
    managed = installer(tmp_path, b"expected", served=b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        managed.prepare()

    assert not managed.checkpoint_path.exists()
    assert not list(tmp_path.glob("*.part"))
    assert not managed.is_ready()


def test_oversized_response_is_rejected(tmp_path: Path) -> None:
    """Stop a download that exceeds its pinned size instead of filling the disk."""
    managed = installer(tmp_path, b"small", served=b"far too much content")

    with pytest.raises(ValueError, match="pinned size"):
        managed.prepare()

    assert not managed.checkpoint_path.exists()


def test_cancellation_removes_staging(tmp_path: Path) -> None:
    """Leave no partial file behind when preparation is cancelled."""
    managed = installer(tmp_path, b"pinned model weights")
    managed.cancelled_error = CancelledError

    with pytest.raises(CancelledError):
        managed.prepare(is_cancelled=lambda: True)

    assert not managed.checkpoint_path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_repeated_preparation_reuses_the_verified_cache(tmp_path: Path) -> None:
    """Skip the network once a verified checkpoint is already published."""
    payload = b"pinned model weights"
    managed = installer(tmp_path, payload)
    managed.prepare()

    def refuse(_url: str) -> io.BytesIO:
        message = "the prepared checkpoint must not be downloaded again"
        raise AssertionError(message)

    managed._opener = refuse  # noqa: SLF001

    assert managed.prepare() == managed.checkpoint_path


def test_tampered_cache_is_not_reported_ready(tmp_path: Path) -> None:
    """Detect a checkpoint replaced after publication through its integrity marker."""
    payload = b"pinned model weights"
    managed = installer(tmp_path, payload)
    managed.prepare()

    managed.checkpoint_path.write_bytes(b"replaced after publication!")

    assert not managed.is_ready()


def test_validate_rejects_foreign_checkpoints(tmp_path: Path) -> None:
    """Refuse to load pickle-backed weights that are not the pinned bytes."""
    payload = b"pinned model weights"
    managed = installer(tmp_path, payload)
    foreign = tmp_path / "foreign.pth"
    foreign.write_bytes(b"someone else's weights")

    with pytest.raises(ValueError, match="missing or has the wrong size"):
        managed.validate(foreign)
