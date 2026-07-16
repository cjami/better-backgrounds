"""Locked-down Qt WebEngine surface for future browser-only rendering."""

from __future__ import annotations

import secrets
from importlib.resources import files
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEngineDownloadRequest,
    QWebEnginePage,
    QWebEnginePermission,
    QWebEngineProfile,
    QWebEngineSettings,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from better_backgrounds.desktop.bridge import RendererBridge

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

TRUSTED_ORIGIN = "https://app.better-backgrounds.invalid"


def navigation_is_allowed(url: QUrl) -> bool:
    """Allow only the bundled renderer's synthetic origin and initial data load."""
    return url.scheme() == "data" or (
        url.scheme() == "https" and url.host() == "app.better-backgrounds.invalid"
    )


class TrustedPage(QWebEnginePage):
    """Reject navigation and all Phase 2 permission requests."""

    def acceptNavigationRequest(
        self,
        url: QUrl | str,
        navigation_type: QWebEnginePage.NavigationType,  # noqa: ARG002
        is_main_frame: bool,  # noqa: FBT001
    ) -> bool:
        """Keep renderer content on its packaged synthetic origin."""
        candidate = QUrl(url) if isinstance(url, str) else url
        return navigation_is_allowed(candidate) and is_main_frame

    def handle_permission(self, permission: QWebEnginePermission) -> None:
        """Deny sensitive browser capabilities until their owning phase."""
        permission.deny()


class SecureRendererView(QWebEngineView):
    """Load local placeholder content with no network or filesystem authority."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an ephemeral profile, narrow bridge, and trusted page."""
        super().__init__(parent)
        profile = QWebEngineProfile(self)
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies,
        )
        profile.downloadRequested.connect(self._reject_download)

        page = TrustedPage(profile, self)
        page.permissionRequested.connect(page.handle_permission)
        settings = page.settings()
        disabled = False
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            disabled,
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            disabled,
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,
            disabled,
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, disabled)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.FullScreenSupportEnabled,
            disabled,
        )

        self._profile = profile
        self._trusted_page = page
        self._bridge = RendererBridge(self)
        self._channel = QWebChannel(self)
        self._channel.registerObject("rendererBridge", self._bridge)
        page.setWebChannel(self._channel)
        self.setPage(page)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAccessibleName("Secure reconstructed scene renderer")
        self._load_placeholder()

    def _load_placeholder(self) -> None:
        template = (
            files("better_backgrounds.desktop")
            .joinpath("assets/viewer.html")
            .read_text(encoding="utf-8")
        )
        nonce = secrets.token_urlsafe(24)
        html = template.replace("{{NONCE}}", nonce)
        self.setHtml(html, QUrl(f"{TRUSTED_ORIGIN}/viewer.html"))

    @staticmethod
    def _reject_download(download: QWebEngineDownloadRequest) -> None:
        download.cancel()


def create_renderer_view() -> QWidget:
    """Create the production embedded renderer surface."""
    return SecureRendererView()
