"""Build-page source-mode behavior tests."""

from pathlib import Path

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton

from better_backgrounds.desktop.pages import BuildPage
from better_backgrounds.desktop.pages.build import _first_supported_path
from better_backgrounds.reconstruction import SplatDiagnostics, SplatSelection


class _DragStub:
    """Provide the minimal drag-event surface used by drop dispatch."""

    def __init__(self, mime: QMimeData) -> None:
        self._mime = mime

    def mimeData(self) -> QMimeData:  # noqa: N802
        """Return the carried mime payload."""
        return self._mime


def _mime_for(*local_paths: str) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path) for path in local_paths])
    return mime


def test_build_page_separates_file_upload_and_camera_capture() -> None:
    """Place a large camera action beneath the file drop surface."""
    QApplication.instance() or QApplication([])
    page = BuildPage()
    buttons = {button.text(): button for button in page.findChildren(QPushButton)}
    assert "Choose a file" in buttons
    assert "Use webcam" in buttons
    assert "Choose a room photo" not in buttons
    assert "Import Gaussian splat" not in buttons

    drop_card = page.findChild(QFrame, "dropCard")
    camera_action = buttons["Use webcam"]
    upload_page = page._content.widget(0)  # noqa: SLF001
    assert upload_page is not None
    upload_layout = upload_page.layout()
    assert upload_layout is not None
    assert drop_card is not None
    assert camera_action.parentWidget() is upload_page
    assert upload_layout.indexOf(camera_action) > upload_layout.indexOf(drop_card)
    assert camera_action.minimumHeight() == 52
    assert not drop_card.isAncestorOf(camera_action)
    formats = drop_card.findChild(QLabel, "feedMeta")
    assert formats is not None
    assert formats.text() == "JPG  ·  JPEG  ·  PNG  ·  WEBP  ·  PLY  ·  SOG  ·  SSOG  ·  ZIP"
    assert page.acceptDrops()
    page.close()


def test_build_page_accepts_supported_drops_and_rejects_others() -> None:
    """Recognize supported photo and splat drops while ignoring other files."""
    QApplication.instance() or QApplication([])
    photo = "C:/rooms/lounge.JPG"
    splat = "C:/rooms/scene.ply"
    sog = "C:/rooms/gallery.sog"
    other = "C:/rooms/notes.txt"
    assert _first_supported_path(_DragStub(_mime_for(photo))) == photo
    assert _first_supported_path(_DragStub(_mime_for(splat))) == splat
    assert _first_supported_path(_DragStub(_mime_for(sog))) == sog
    assert _first_supported_path(_DragStub(_mime_for(other))) is None
    assert _first_supported_path(_DragStub(QMimeData())) is None
    mixed = _first_supported_path(_DragStub(_mime_for(other, "C:/rooms/b.webp")))
    assert mixed == "C:/rooms/b.webp"


def test_build_page_offers_and_reviews_direct_splat_imports() -> None:
    """Switch review controls from SHARP inference to managed PLY import."""
    QApplication.instance() or QApplication([])
    page = BuildPage()

    page.show_splat_review(
        SplatSelection("meeting_room.ply", Path("meeting_room.ply")),
        SplatDiagnostics(
            gaussian_count=12_345,
            file_size=1024**2,
            layout="compressed",
            framing="Automatic COLMAP framing",
            bounds_minimum=(-1.0, -1.0, 1.0),
            bounds_maximum=(1.0, 1.0, 3.0),
            center_of_mass=(0.0, 0.0, 2.0),
        ),
    )

    assert page._build_action.text() == "Import room"  # noqa: SLF001
    assert page._build_action.isEnabled()  # noqa: SLF001
    assert page._device.isHidden()  # noqa: SLF001
    assert page._diagnostic_values["dimensions"].text() == "12,345"  # noqa: SLF001
    assert page._diagnostic_values["alpha"].text() == "1.0 MiB"  # noqa: SLF001
    page.close()


def test_build_page_reviews_streamed_sog_environment_metadata() -> None:
    """Show LOD and managed-resource diagnostics for an SSOG import."""
    QApplication.instance() or QApplication([])
    page = BuildPage()

    page.show_splat_review(
        SplatSelection("museum.ssog", Path("museum.ssog")),
        SplatDiagnostics(
            gaussian_count=1_250_000,
            total_gaussian_count=2_400_000,
            file_size=64 * 1024**2,
            layout="streamed-sog",
            framing="Automatic streamed bounds",
            bounds_minimum=(-20.0, -3.0, -20.0),
            bounds_maximum=(20.0, 8.0, 20.0),
            center_of_mass=(0.0, 2.0, 0.0),
            lod_levels=4,
            resource_count=97,
        ),
    )

    assert page._diagnostic_values["dimensions"].text() == "1,250,000"  # noqa: SLF001
    assert page._diagnostic_values["format"].text() == "Streamed SOG"  # noqa: SLF001
    assert page._diagnostic_values["orientation"].text() == "4 LOD levels / 97 files"  # noqa: SLF001
    assert page._device.isHidden()  # noqa: SLF001
    page.close()
