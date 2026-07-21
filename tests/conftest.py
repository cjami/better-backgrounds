"""Shared test-process configuration."""

import os
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

# Read when the platform plugin loads, which is deferred until the first
# application is constructed, so setting it here still beats the fixture below.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> QApplication:
    """Own the single QApplication that Qt allows per process.

    Constructing a widget without one aborts the interpreter, so no test can
    depend on an earlier test in the session having created it first.
    """
    existing = cast("QApplication | None", QApplication.instance())
    return existing or QApplication([])
