"""Smoke tests for the installed package."""

from importlib.metadata import version

from better_backgrounds import __version__


def test_package_exposes_installed_version() -> None:
    """Expose the same version as the installed distribution metadata."""
    assert __version__ == version("better-backgrounds")
