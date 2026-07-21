"""Point the deployment spec at the interpreter and icon for the current platform.

`pysidedeploy.spec` is committed with cross-platform defaults. pyside6-deploy needs an
absolute interpreter path that exists on the machine running the build, and macOS
bundles need an `.icns` icon where Windows needs `.ico`.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "pysidedeploy.spec"
ICONS = {"darwin": "app-icon.icns", "win32": "app-icon.ico"}
DEFAULT_ICON = "app-icon.ico"


def main() -> None:
    """Rewrite the spec in place for this platform and report the result."""
    # interpolation=None keeps any literal '%' in Nuitka arguments intact.
    spec = configparser.ConfigParser(interpolation=None)
    # Keys are case-sensitive option names that pyside6-deploy reads verbatim.
    spec.optionxform = str  # type: ignore[method-assign, assignment]
    spec.read(SPEC, encoding="utf-8")

    spec["python"]["python_path"] = sys.executable

    icon = ICONS.get(sys.platform, DEFAULT_ICON)
    spec["app"]["icon"] = f"src/better_backgrounds/desktop/assets/{icon}"

    with SPEC.open("w", encoding="utf-8") as handle:
        spec.write(handle)

    print(f"platform    : {sys.platform}")  # noqa: T201
    print(f"python_path : {spec['python']['python_path']}")  # noqa: T201
    print(f"icon        : {spec['app']['icon']}")  # noqa: T201
    print(f"input_file  : {spec['app']['input_file']}")  # noqa: T201


if __name__ == "__main__":
    main()
