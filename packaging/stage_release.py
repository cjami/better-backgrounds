"""Locate the packaged desktop build and stage its first-run helper beside it.

`pyside6-deploy` places its standalone output differently per platform, so the
release workflow discovers the build by Nuitka's directory layout rather than by
executable name. Prints the staged directory, or the trees it searched and a
non-zero exit when nothing matched.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Fixed output location so callers never need to parse this script's stdout.
STAGING = ROOT / "build" / "release"
SEARCH_ROOTS = (ROOT / "dist", ROOT / "build", ROOT / "src")
# Nuitka standalone always emits a `<name>.dist` directory; macOS app-bundle
# mode emits `<name>.app`. Neither depends on the configured executable name.
BUNDLE_SUFFIXES = (".app", ".dist")
HELPERS = {
    "win32": ROOT / "packaging/windows/FIRST-RUN.txt",
    "darwin": ROOT / "packaging/macos/Unlock and Open.command",
}
LISTING_DEPTH = 3
LISTING_LIMIT = 60


def find_build() -> Path:
    """Return the packaged application directory produced by pyside6-deploy."""
    candidates = [
        path
        for root in SEARCH_ROOTS
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_dir()
        and path.suffix in BUNDLE_SUFFIXES
        # Never rediscover a previously staged copy of ourselves.
        and STAGING not in path.parents
    ]
    if not candidates:
        raise SystemExit(_failure_report())
    # Prefer the shallowest match so a nested helper bundle never wins.
    return sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))[0]


def _failure_report() -> str:
    """Describe what actually exists so a failed run is diagnosable."""
    lines = ["No packaged application found (looked for *.app or *.dist directories)."]
    for root in SEARCH_ROOTS:
        lines.append(f"\n{root}:")
        if not root.is_dir():
            lines.append("  <missing>")
            continue
        shown = 0
        for path in sorted(root.rglob("*")):
            if len(path.relative_to(root).parts) > LISTING_DEPTH:
                continue
            lines.append(f"  {path.relative_to(root)}{'/' if path.is_dir() else ''}")
            shown += 1
            if shown >= LISTING_LIMIT:
                lines.append("  ... (truncated)")
                break
        if shown == 0:
            lines.append("  <empty>")
    return "\n".join(lines)


def main() -> None:
    """Assemble a clean release directory holding the app and its first-run helper."""
    build = find_build()
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # Move rather than copy: the bundle is multi-gigabyte and the original is
    # not needed again, so a same-filesystem rename keeps CI fast.
    shutil.move(str(build), str(STAGING / build.name))

    helper = HELPERS.get(sys.platform)
    if helper is not None and helper.is_file():
        destination = STAGING / helper.name
        shutil.copy2(helper, destination)
        if sys.platform == "darwin":
            destination.chmod(0o755)

    staged = sorted(entry.name for entry in STAGING.iterdir())
    if not any(Path(name).suffix in BUNDLE_SUFFIXES for name in staged):
        message = f"Staged {STAGING} without an application bundle: {staged}"
        raise SystemExit(message)
    print(f"Staged into {STAGING}: {', '.join(staged)}")  # noqa: T201


if __name__ == "__main__":
    main()
