"""Fail the release if the produced archive does not actually contain the application.

A packaging or archiving mistake can still yield a small, well-formed zip. Checking
the contents here keeps a broken artifact from reaching a published release.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

MINIMUM_ENTRIES = 20
MINIMUM_BYTES = 20 * 1024 * 1024
EXECUTABLE_STEMS = ("BetterBackgrounds", "QtWebEngineProcess")


def main() -> None:
    """Report the archive contents and exit non-zero when it looks incomplete."""
    archive = Path(sys.argv[1])
    if not archive.is_file():
        message = f"Archive was never created: {archive}"
        raise SystemExit(message)

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        uncompressed = sum(info.file_size for info in bundle.infolist())

    print(f"archive      : {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")  # noqa: T201
    print(f"entries      : {len(names)}")  # noqa: T201
    print(f"uncompressed : {uncompressed / 1024**2:.1f} MiB")  # noqa: T201
    for name in sorted(names)[:15]:
        print(f"  {name}")  # noqa: T201

    problems: list[str] = []
    if len(names) < MINIMUM_ENTRIES:
        problems.append(f"only {len(names)} entries")
    if uncompressed < MINIMUM_BYTES:
        problems.append(f"only {uncompressed / 1024**2:.1f} MiB uncompressed")
    if not any(stem in name for name in names for stem in EXECUTABLE_STEMS):
        problems.append(f"no {' or '.join(EXECUTABLE_STEMS)} entry")
    if problems:
        message = f"Release archive looks incomplete: {'; '.join(problems)}"
        raise SystemExit(message)

    print("Archive contains a complete application build.")  # noqa: T201


if __name__ == "__main__":
    main()
