"""Python-owned desktop application for Better Backgrounds."""

from PySide6.QtCore import QByteArray
from PySide6.QtWebEngineCore import QWebEngineUrlScheme

from better_backgrounds.scene import APP_SCHEME, SCENE_SCHEME


def _register_renderer_schemes() -> None:
    """Register controlled package and scene schemes before Qt starts."""
    for name in (APP_SCHEME, SCENE_SCHEME):
        scheme = QWebEngineUrlScheme(QByteArray(name.encode()))
        scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
        scheme.setFlags(
            QWebEngineUrlScheme.Flag.SecureScheme
            | QWebEngineUrlScheme.Flag.CorsEnabled
            | QWebEngineUrlScheme.Flag.FetchApiAllowed,
        )
        QWebEngineUrlScheme.registerScheme(scheme)


_register_renderer_schemes()
