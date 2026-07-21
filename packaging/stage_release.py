"""Locate the packaged desktop build and stage its first-run helper beside it.

`pyside6-deploy` places its standalone output differently per platform and Nuitka
version, so the release workflow discovers the build instead of assuming a path.
Prints the staged directory on success and exits non-zero if nothing was found.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (ROOT / "dist", ROOT / "build")
EXECUTABLES = ("BetterBackgrounds.exe", "BetterBackgrounds")
HELPERS = {
    "win32": ROOT / "packaging/windows/FIRST-RUN.txt",
    "darwin": ROOT / "packaging/macos/Unlock and Open.command",
}


def find_build() -> Path:
    """Return the directory that contains the packaged application."""
    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        candidates.extend(bundle.parent for bundle in root.rglob("*.app") if bundle.is_dir())
        for name in EXECUTABLES:
            candidates.extend(match.parent for match in root.rglob(name) if match.is_file())
    if not candidates:
        searched = ", ".join(str(root) for root in SEARCH_ROOTS)
        message = f"No packaged application found under: {searched}"
        raise SystemExit(message)
    # Prefer the shallowest match so we stage the bundle root, not a nested helper.
    return sorted(set(candidates), key=lambda path: len(path.parts))[0]


def main() -> None:
    """Stage the platform helper next to the packaged application."""
    build = find_build()
    helper = HELPERS.get(sys.platform)
    if helper is not None and helper.is_file():
        destination = build / helper.name
        shutil.copy2(helper, destination)
        if sys.platform == "darwin":
            destination.chmod(0o755)
    print(build)  # noqa: T201


if __name__ == "__main__":
    main()
