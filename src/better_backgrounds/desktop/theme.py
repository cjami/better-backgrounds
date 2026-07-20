"""Desktop design tokens inspired by the product mock."""

from __future__ import annotations

ACCENT = "#e0a34a"
BACKGROUND = "#0e0f12"
SURFACE = "#15161a"
TEXT = "#e7e8ea"
MUTED = "#8b8e96"
SUCCESS = "#7fce9b"
DANGER = "#e58076"

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QMainWindow {{ background: {BACKGROUND}; }}
QLabel {{ background: transparent; }}
QFrame#header {{
    background: #111216;
    border-bottom: 1px solid #282a30;
}}
QFrame#headerDivider {{ background: #303238; border: none; }}
QFrame#card {{
    background: {SURFACE};
    border: 1px solid #292b31;
    border-radius: 14px;
}}
QWidget#buildPage {{ background: {BACKGROUND}; }}
QFrame#dropCard {{
    background: #171616;
    border: 1px dashed #5b5c61;
    border-radius: 14px;
}}
QFrame#dropCard:hover {{ border-color: {ACCENT}; background: #1b1916; }}
QFrame#dropCard[dragActive="true"] {{ border: 2px solid {ACCENT}; background: #221d13; }}
QLabel#capturePreview {{
    background: #090a0c;
    border: 1px solid #292b31;
    border-radius: 12px;
    color: {MUTED};
}}
QLabel#countdown {{ color: {ACCENT}; font-size: 44px; font-weight: 700; }}
QFrame#inspector {{
    background: #111216;
    border: none;
    border-left: 1px solid #292b31;
}}
QFrame#roomRail {{
    background: #111216;
    border: none;
    border-left: 1px solid #292b31;
}}
QFrame#feedSurface, QWidget#feedOverlay {{ background: transparent; border: none; }}
QLabel#brand {{ font-size: 17px; font-weight: 700; }}
QLabel#title {{ font-size: 28px; font-weight: 700; }}
QLabel#heroTitle {{ font-size: 30px; font-weight: 700; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 2px; }}
QLabel#subtitle, QLabel#muted {{ color: {MUTED}; }}
QLabel#section {{ font-size: 14px; font-weight: 650; }}
QLabel#success {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#danger {{ color: {DANGER}; font-weight: 650; }}
QLabel#stageActive {{ color: {ACCENT}; font-weight: 650; }}
QLabel#stageDone {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#stagePending {{ color: #696c74; }}
QLabel#feedBadge {{
    color: #d9dbe0;
    background: #17181c;
    border: 1px solid #292b31;
    border-radius: 8px;
    padding: 7px 11px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#feedMeta {{
    color: #b5b8c0;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
}}
QLabel#previewNote {{ color: #e3e4e7; font-size: 14px; font-weight: 600; }}
QLabel#roomName {{ font-weight: 650; }}
QLabel#readyBadge {{
    color: #8fd6a9;
    background: #203429;
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#roomThumbnail {{
    background: qradialgradient(cx:0.62, cy:0.18, radius:0.78,
        stop:0 #8b8679, stop:0.34 #3b3d42, stop:1 #171a20);
    border: 1px solid #2c2e34;
    border-radius: 10px;
}}
QLabel#uploadIcon {{
    color: {ACCENT};
    background: #392f20;
    border-radius: 14px;
    padding: 10px 16px;
    font-size: 24px;
    font-weight: 700;
}}
QLabel#controlValue {{ color: #e6e7ea; font-size: 11px; }}
QPushButton {{
    min-height: 34px;
    padding: 2px 15px;
    border-radius: 9px;
    border: 1px solid #34363d;
    background: #1b1c21;
    font-weight: 600;
}}
QPushButton:hover {{ background: #24262c; border-color: #454852; }}
QPushButton:pressed {{ background: #191a1f; }}
QPushButton:disabled {{ color: #62656d; background: #16171b; }}
QPushButton#primary {{
    color: #1a1204;
    background: {ACCENT};
    border-color: {ACCENT};
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: #edb45f; }}
QPushButton#quiet {{ background: transparent; border-color: transparent; color: {MUTED}; }}
QPushButton#danger {{ color: {DANGER}; border-color: #593a3d; background: #24191b; }}
QPushButton#tab {{
    min-height: 28px;
    padding: 0 13px;
    color: #73767e;
    background: transparent;
    border: none;
}}
QPushButton#tab:hover {{ color: {TEXT}; background: #1b1c21; }}
QPushButton#tab[active="true"] {{ color: {ACCENT}; background: #2a2216; }}
QFrame#roomPill {{
    min-height: 30px;
    color: #c9cbd1;
    background: #1b1c21;
    border-radius: 9px;
    border: none;
}}
QPushButton#headerIcon {{
    min-width: 34px;
    max-width: 34px;
    min-height: 30px;
    padding: 0;
    background: #1b1c21;
    border: none;
}}
QPushButton#cameraToggle {{
    min-height: 52px;
    color: #1a1204;
    background: {ACCENT};
    border-color: {ACCENT};
    border-radius: 13px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton#cameraToggle:hover {{ background: #edb45f; }}
QPushButton#cameraToggle[active="true"] {{
    color: #f0a196;
    background: #2b1b1d;
    border-color: #794248;
}}
QPushButton#quietAction {{
    color: {ACCENT};
    background: transparent;
    border-color: #473822;
}}
QPushButton#railAction, QPushButton#sampleAction {{
    color: {ACCENT};
    background: transparent;
    border: none;
    padding: 0 4px;
}}
QPushButton#dropAction {{
    color: {TEXT};
    background: transparent;
    border: none;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#overlayChip, QPushButton#preset {{
    min-height: 28px;
    padding: 0 11px;
    color: #b7bac1;
    background: #18191d;
    border: 1px solid #272930;
}}
QPushButton#overlayChip:checked, QPushButton#preset[active="true"] {{
    color: {ACCENT};
    background: #352b1d;
    border-color: #76582e;
}}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    margin: 3px 0;
    padding: 0;
    border: 1px solid #292b31;
    border-radius: 12px;
    background: #18191d;
}}
QListWidget QWidget {{ background: transparent; }}
QListWidget::item:hover {{ border-color: #454852; background: #1d1f24; }}
QListWidget::item:selected {{
    color: {TEXT};
    border-color: #76582e;
    background: #2a2216;
}}
QProgressBar {{
    height: 9px;
    border: none;
    border-radius: 4px;
    background: #26282e;
    text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: #30323a; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {ACCENT};
}}
QComboBox {{
    min-height: 32px;
    padding: 0 10px;
    border: 1px solid #34363d;
    border-radius: 8px;
    background: #1b1c21;
}}
QTextEdit {{
    background: #101115;
    color: #9da1aa;
    border: 1px solid #272930;
    border-radius: 10px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    padding: 9px;
}}
QToolTip {{ color: {TEXT}; background: #24262c; border: 1px solid #41434b; }}
"""
