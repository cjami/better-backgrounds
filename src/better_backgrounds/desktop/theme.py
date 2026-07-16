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
QFrame#card, QFrame#dropCard, QFrame#inspector {{
    background: {SURFACE};
    border: 1px solid #292b31;
    border-radius: 14px;
}}
QFrame#dropCard:hover {{ border-color: {ACCENT}; }}
QLabel#brand {{ font-size: 17px; font-weight: 700; }}
QLabel#title {{ font-size: 28px; font-weight: 700; }}
QLabel#subtitle, QLabel#muted {{ color: {MUTED}; }}
QLabel#section {{ font-size: 14px; font-weight: 650; }}
QLabel#success {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#danger {{ color: {DANGER}; font-weight: 650; }}
QLabel#stageActive {{ color: {ACCENT}; font-weight: 650; }}
QLabel#stageDone {{ color: {SUCCESS}; font-weight: 650; }}
QLabel#stagePending {{ color: #696c74; }}
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
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    min-height: 44px;
    margin: 3px 0;
    padding: 4px 11px;
    border: 1px solid #292b31;
    border-radius: 10px;
    background: #18191d;
}}
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
