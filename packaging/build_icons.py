"""Render the packaged SVG app icon into the .ico and .icns files Nuitka needs.

Run from the repository root with `uv run python packaging/build_icons.py`.
Regenerate and commit the results whenever `app-icon.svg` changes.
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

from PySide6.QtCore import QBuffer, QSize
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).resolve().parents[1] / "src/better_backgrounds/desktop/assets"
SOURCE = ASSETS / "app-icon.svg"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICO_PNG_SENTINEL_SIZE = 256
# Apple's PNG-backed icon types, smallest to largest.
ICNS_TYPES = (
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),
)


def render(size: int) -> QImage:
    """Rasterize the vector icon into one square ARGB image."""
    renderer = QSvgRenderer(str(SOURCE))
    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return image


def png_bytes(image: QImage) -> bytes:
    """Encode one rendered image as PNG."""
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def write_ico(path: Path) -> None:
    """Write a multi-resolution Windows icon."""
    entries = [(size, png_bytes(render(size))) for size in ICO_SIZES]
    header = struct.pack("<HHH", 0, 1, len(entries))
    directory = BytesIO()
    payload = BytesIO()
    offset = len(header) + 16 * len(entries)
    for size, data in entries:
        stored = 0 if size >= ICO_PNG_SENTINEL_SIZE else size
        directory.write(struct.pack("<BBBBHHII", stored, stored, 0, 0, 1, 32, len(data), offset))
        payload.write(data)
        offset += len(data)
    path.write_bytes(header + directory.getvalue() + payload.getvalue())


def write_icns(path: Path) -> None:
    """Write a PNG-backed macOS icon."""
    body = BytesIO()
    for icon_type, size in ICNS_TYPES:
        data = png_bytes(render(size))
        body.write(icon_type + struct.pack(">I", len(data) + 8) + data)
    payload = body.getvalue()
    path.write_bytes(b"icns" + struct.pack(">I", len(payload) + 8) + payload)


def main() -> None:
    """Regenerate both packaged icon files."""
    QGuiApplication([])
    write_ico(ASSETS / "app-icon.ico")
    write_icns(ASSETS / "app-icon.icns")


if __name__ == "__main__":
    main()
