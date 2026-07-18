"""Build-page source-mode behavior tests."""

from pathlib import Path

from PySide6.QtWidgets import QApplication, QPushButton

from better_backgrounds.desktop.pages import BuildPage
from better_backgrounds.reconstruction import SplatDiagnostics, SplatSelection


def test_build_page_offers_and_reviews_direct_splat_imports() -> None:
    """Switch review controls from SHARP inference to managed PLY import."""
    QApplication.instance() or QApplication([])
    page = BuildPage()
    buttons = {button.text(): button for button in page.findChildren(QPushButton)}
    assert "Choose a room photo" in buttons
    assert "Import Gaussian splat" in buttons

    page.show_splat_review(
        SplatSelection("meeting_room.ply", Path("meeting_room.ply")),
        SplatDiagnostics(
            gaussian_count=12_345,
            file_size=1024**2,
            layout="compressed",
            framing="Automatic COLMAP framing",
            bounds_minimum=(-1.0, -1.0, 1.0),
            bounds_maximum=(1.0, 1.0, 3.0),
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
            lod_levels=4,
            resource_count=97,
        ),
    )

    assert page._diagnostic_values["dimensions"].text() == "1,250,000"  # noqa: SLF001
    assert page._diagnostic_values["format"].text() == "Streamed SOG"  # noqa: SLF001
    assert page._diagnostic_values["orientation"].text() == "4 LOD levels / 97 files"  # noqa: SLF001
    assert page._device.isHidden()  # noqa: SLF001
    page.close()
