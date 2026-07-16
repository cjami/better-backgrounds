"""Shared application icon resources."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtGui import QIcon


def application_icon() -> QIcon:
    """Load the packaged vector icon used by Qt windows and shell chrome."""
    icon_path = files("better_backgrounds.desktop").joinpath("assets", "app-icon.svg")
    return QIcon(str(icon_path))
