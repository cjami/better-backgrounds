"""Feature-first: Verify every mandatory model is prepared together and gated once."""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import pytest

from better_backgrounds import model_setup
from better_backgrounds.checkpoints import CheckpointIdentity, ManagedCheckpointInstaller

if TYPE_CHECKING:
    from pathlib import Path

SIZES = {"matanyone": 100, "pih": 200, "sharp": 700}


def build_installer(
    root: Path,
    key: str,
    opened: list[str],
    *,
    requires_license: bool,
) -> ManagedCheckpointInstaller:
    """Build a real installer whose only stub is its download boundary."""
    payload = bytes(SIZES[key])
    installer = ManagedCheckpointInstaller(
        root / key,
        CheckpointIdentity(
            filename=f"{key}.pth",
            url=f"https://example.invalid/{key}.pth",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        opener=lambda _url: (opened.append(key), io.BytesIO(payload))[1],
    )
    installer.label = key
    installer.license_name = f"{key}-license"
    installer.requires_license_acceptance = requires_license
    return installer


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Install the three mandatory models with recorded, in-memory downloads."""
    log: list[str] = []
    installers = [
        (
            "matanyone",
            "MatAnyone 2",
            build_installer(tmp_path, "matanyone", log, requires_license=False),
        ),
        ("pih", "Adobe PIH", build_installer(tmp_path, "pih", log, requires_license=False)),
        ("sharp", "Apple SHARP", build_installer(tmp_path, "sharp", log, requires_license=True)),
    ]
    monkeypatch.setattr(model_setup, "model_installers", lambda: installers)
    return log


def test_every_mandatory_model_is_reported_until_prepared(opened: list[str]) -> None:
    """Treat SHARP, MatAnyone, and PIH as one required set with no optional member."""
    assert not model_setup.all_models_ready()
    assert {status.key for status in model_setup.missing_models()} == {"sharp", "matanyone", "pih"}
    assert model_setup.total_download_size() == 1000

    model_setup.prepare_models(license_accepted=True)

    assert model_setup.all_models_ready()
    assert opened == ["matanyone", "pih", "sharp"]


def test_preparation_requires_the_sharp_model_license(opened: list[str]) -> None:
    """Refuse the license-gated model until its terms are accepted."""
    with pytest.raises(PermissionError, match="license"):
        model_setup.prepare_models(license_accepted=False)

    assert "sharp" not in opened
    assert not model_setup.all_models_ready()


def test_combined_progress_spans_every_pending_model(opened: list[str]) -> None:
    """Report one continuous byte total so a single bar covers the whole setup."""
    del opened
    seen: list[tuple[int, int]] = []

    model_setup.prepare_models(license_accepted=True, progress=lambda d, t: seen.append((d, t)))

    assert [done for done, _total in seen] == [100, 300, 1000]
    assert {total for _done, total in seen} == {1000}


def test_prepared_models_are_not_downloaded_again(opened: list[str]) -> None:
    """Skip the whole setup step once every model is cached."""
    model_setup.prepare_models(license_accepted=True)
    opened.clear()

    model_setup.prepare_models(license_accepted=False)

    assert opened == []
    assert model_setup.all_models_ready()
