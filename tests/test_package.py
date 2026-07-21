"""Smoke tests for the installed package."""

import tomllib
from importlib.metadata import version
from pathlib import Path

from better_backgrounds import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src/better_backgrounds"


def deployment_manifest() -> list[str]:
    """Read the file list that pyside6-deploy resolves when packaging."""
    content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(content)["tool"]["pyside6-project"]["files"]


def first_party_modules() -> set[str]:
    """Return every first-party module that must reach the packaged application."""
    return {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*.py")
        if "_vendor" not in path.parts and "__pycache__" not in path.parts
    }


def test_package_exposes_installed_version() -> None:
    """Expose the same version as the installed distribution metadata."""
    assert __version__ == version("better-backgrounds")


def test_deployment_manifest_resolves_every_declared_file() -> None:
    """Keep packaging from failing on a path that no longer exists."""
    missing = [entry for entry in deployment_manifest() if not (PROJECT_ROOT / entry).exists()]

    assert missing == []


def test_deployment_manifest_covers_every_first_party_module() -> None:
    """Keep a new module from being silently omitted when the desktop app is frozen."""
    unlisted = first_party_modules() - set(deployment_manifest())

    assert unlisted == set()
