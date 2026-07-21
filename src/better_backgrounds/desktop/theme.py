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
QFrame#settingsDivider {{ background: #2a2c32; border: none; }}
QFrame#feedSurface, QWidget#feedOverlay {{ background: transparent; border: none; }}
QLabel#brand {{ font-size: 17px; font-weight: 700; }}
QLabel#title {{ font-size: 28px; font-weight: 700; }}
QLabel#heroTitle {{ font-size: 30px; font-weight: 700; }}
QLabel#dropTitle {{ font-size: 16px; font-weight: 650; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 2px; }}
QLabel#subtitle, QLabel#muted {{ color: {MUTED}; }}
QLabel#section {{ font-size: 14px; font-weight: 650; }}
QLabel#success {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#danger {{ color: {DANGER}; font-weight: 650; }}
QLabel#stageActive {{ color: {ACCENT}; font-weight: 650; }}
QLabel#stageDone {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#stagePending {{ color: #696c74; }}
QLabel#feedMeta {{
    color: #b5b8c0;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
}}
QLabel#previewNote {{ color: #e3e4e7; font-size: 14px; font-weight: 600; }}
QLabel#roomName {{ font-weight: 650; }}
QPushButton#roomDelete {{
    min-height: 0;
    padding: 0;
    color: {MUTED};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    font-size: 13px;
}}
QPushButton#roomDelete:hover {{ color: #f19a91; background: #311f22; border-color: #75464c; }}
QPushButton#roomDelete:pressed {{ color: #cf7168; background: #191113; border-color: #482e32; }}
QLabel#roomThumbnail {{
    background: qradialgradient(cx:0.62, cy:0.18, radius:0.78,
        stop:0 #8b8679, stop:0.34 #3b3d42, stop:1 #171a20);
    border: 1px solid #2c2e34;
    border-radius: 10px;
    color: #72757d;
    font-size: 18px;
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
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton:pressed {{
    color: #d2d4d9;
    background: #121317;
    border-color: #2d2f35;
    padding-top: 4px;
    padding-bottom: 0;
}}
QPushButton:disabled {{ color: #5f626a; background: #16171b; border-color: #26282d; }}
QPushButton#primary {{
    color: #1a1204;
    background: {ACCENT};
    border-color: {ACCENT};
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: #edb45f; }}
QPushButton#primary:focus {{ border-color: #f4cb8e; }}
QPushButton#primary:pressed {{
    color: #110c03;
    background: #bd7f28;
    border-color: #bd7f28;
    padding-top: 4px;
    padding-bottom: 0;
}}
QPushButton#quiet {{ background: transparent; border-color: transparent; color: {MUTED}; }}
QPushButton#quiet:hover {{ color: {TEXT}; background: #1b1c21; }}
QPushButton#quiet:pressed {{ color: #b9bbc1; background: #121317; }}
QPushButton#danger {{ color: {DANGER}; border-color: #593a3d; background: #24191b; }}
QPushButton#danger:hover {{ color: #f19a91; background: #311f22; border-color: #75464c; }}
QPushButton#danger:pressed {{ color: #cf7168; background: #191113; border-color: #482e32; }}
QPushButton#tab {{
    min-width: 88px;
    min-height: 38px;
    padding: 0 20px;
    color: #73767e;
    background: transparent;
    border: none;
    border-radius: 11px;
    font-size: 14px;
}}
QPushButton#tab:hover {{ color: {TEXT}; background: #1b1c21; }}
QPushButton#tab[active="true"] {{ color: {ACCENT}; background: #2a2216; }}
QPushButton#tab:pressed {{ color: #c98c37; background: #17140f; padding-top: 2px; }}
QPushButton#tab:focus {{ border: 1px solid #76582e; }}
QFrame#roomPill {{
    min-height: 30px;
    color: #c9cbd1;
    background: #1b1c21;
    border-radius: 9px;
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
QPushButton#cameraToggle:pressed {{
    color: #160f04;
    background: #bd7f28;
    border-color: #bd7f28;
    padding-top: 4px;
    padding-bottom: 0;
}}
QPushButton#cameraToggle[active="true"]:hover {{ background: #382226; border-color: #92515a; }}
QPushButton#cameraToggle[active="true"]:pressed {{
    color: #d98278;
    background: #1a1113;
    border-color: #583238;
}}
QPushButton#quietAction {{
    color: {ACCENT};
    background: transparent;
    border-color: #473822;
}}
QPushButton#quietAction:hover {{ color: #efb968; background: #211b13; border-color: #624c2a; }}
QPushButton#quietAction:pressed {{ color: #bf8130; background: #14110d; border-color: #382c1c; }}
QPushButton#cameraAction {{
    min-height: 52px;
    color: #d7d9de;
    background: #191a1f;
    border-color: #3a3c44;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton#cameraAction:hover {{ color: {TEXT}; background: #22242a; border-color: #555862; }}
QPushButton#cameraAction:pressed {{
    color: #c5c7cd;
    background: #121317;
    border-color: #30323a;
    padding-top: 4px;
    padding-bottom: 0;
}}
QPushButton#railAction, QPushButton#sampleAction {{
    color: {ACCENT};
    background: transparent;
    border: none;
    padding: 0 4px;
}}
QPushButton#railAction:hover, QPushButton#sampleAction:hover {{
    color: #efb968;
    background: #211b13;
}}
QPushButton#railAction:pressed, QPushButton#sampleAction:pressed {{
    color: #bd7f28;
    background: #14110d;
    padding-top: 2px;
}}
QPushButton#dropAction {{
    min-height: 38px;
    color: #1a1204;
    background: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#dropAction:hover {{ color: #1a1204; background: #edb45f; border-color: #edb45f; }}
QPushButton#dropAction:pressed {{
    color: #110c03;
    background: #bd7f28;
    border-color: #bd7f28;
    padding-top: 4px;
    padding-bottom: 0;
}}
QPushButton#preset {{
    min-height: 28px;
    padding: 0 11px;
    color: #b7bac1;
    background: #18191d;
    border: 1px solid #272930;
}}
QPushButton#preset[active="true"] {{
    color: {ACCENT};
    background: #352b1d;
    border-color: #76582e;
}}
QPushButton#preset:hover {{ border-color: #454852; }}
QPushButton#preset:pressed {{
    color: #989ba3;
    background: #101115;
    border-color: #22242a;
    padding-top: 2px;
}}
QPushButton#preset[active="true"]:pressed {{
    color: #c88b36;
    background: #1c1710;
    border-color: #4d3b22;
}}
QCheckBox {{
    spacing: 10px;
    min-height: 28px;
    color: #c9cbd1;
}}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox:focus {{ color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #4a4d55;
    border-radius: 5px;
    background: #131418;
}}
QCheckBox::indicator:hover {{ border-color: #6d7079; background: #1d1f24; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QCheckBox::indicator:checked:hover {{ background: #edb45f; border-color: #edb45f; }}
QCheckBox::indicator:disabled {{ background: #17181c; border-color: #292b31; }}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget:focus {{ border: 1px solid #554326; border-radius: 12px; }}
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
QSlider::handle:horizontal:hover {{ background: #edb45f; }}
QSlider::handle:horizontal:pressed {{ background: #bd7f28; }}
QSlider:focus {{ outline: none; }}
QComboBox {{
    min-height: 32px;
    padding: 0 10px;
    border: 1px solid #34363d;
    border-radius: 8px;
    background: #1b1c21;
}}
QComboBox:hover {{ border-color: #4c4f58; background: #202126; }}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled {{ color: #62656d; background: #16171b; border-color: #26282d; }}
QComboBox::drop-down {{ width: 30px; border: none; }}
QComboBox QAbstractItemView {{
    color: {TEXT};
    background: #1b1c21;
    border: 1px solid #3a3c43;
    selection-color: {TEXT};
    selection-background-color: #3a2e1d;
    outline: none;
}}
QScrollBar:vertical {{ width: 10px; margin: 2px; background: transparent; }}
QScrollBar::handle:vertical {{ min-height: 34px; background: #383a42; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: #4a4d56; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
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
