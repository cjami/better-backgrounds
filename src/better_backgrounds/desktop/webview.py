"""Locked-down Qt WebEngine surface for future browser-only rendering."""

from __future__ import annotations

import secrets
from importlib.resources import files
from mimetypes import guess_type
from typing import TYPE_CHECKING, ClassVar

from PySide6.QtCore import QBuffer, QByteArray, QFile, QIODevice, Qt, QUrl, Signal
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEngineDownloadRequest,
    QWebEnginePage,
    QWebEnginePermission,
    QWebEngineProfile,
    QWebEngineSettings,
    QWebEngineUrlRequestJob,
    QWebEngineUrlSchemeHandler,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from better_backgrounds.desktop.bridge import RendererBridge
from better_backgrounds.scene import APP_SCHEME, ManagedSceneResolver, SceneReference, Viewpoint

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


class ManagedSceneSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serve only verified files named by the sample asset manifest."""

    def __init__(self, resolver: ManagedSceneResolver, parent: QWebEngineProfile) -> None:
        """Keep the resolver and response devices owned by the profile."""
        super().__init__(parent)
        self._resolver = resolver

    def requestStarted(self, request: QWebEngineUrlRequestJob) -> None:
        """Reply to safe GET requests and deny every other request."""
        if request.requestMethod().data() != b"GET":
            request.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        path = self._resolver.resolve(request.requestUrl())
        if path is None:
            request.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        device = QFile(str(path), request)
        if not device.open(QIODevice.OpenModeFlag.ReadOnly):
            request.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
            return
        mime_type = guess_type(path.name)[0] or "application/octet-stream"
        request.reply(QByteArray(mime_type.encode()), device)


class PackagedRendererSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serve the one allowlisted room renderer from package data."""

    RESOURCES: ClassVar[dict[tuple[str, str], tuple[str, str]]] = {
        ("renderer", "/renderer.js"): ("renderer.js", "text/javascript"),
    }

    def requestStarted(self, request: QWebEngineUrlRequestJob) -> None:
        """Reject every package URL except an exact allowlisted asset."""
        url = request.requestUrl()
        resource = self.RESOURCES.get((url.host(), url.path()))
        if (
            request.requestMethod().data() != b"GET"
            or resource is None
            or url.hasQuery()
            or url.hasFragment()
        ):
            request.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        relative_path, mime_type = resource
        content = files("better_backgrounds.desktop").joinpath("assets", relative_path).read_bytes()
        device = QBuffer(request)
        device.setData(QByteArray(content))
        device.open(QIODevice.OpenModeFlag.ReadOnly)
        request.reply(QByteArray(mime_type.encode()), device)


class SecureRendererView(QWebEngineView):
    """Load local placeholder content with no network or filesystem authority."""

    scene_progressed = Signal(int, int)
    scene_failed = Signal(str)
    viewpoint_changed = Signal(object)

    def __init__(
        self,
        resolver: ManagedSceneResolver | None = None,
        parent: QWidget | None = None,
        *,
        snapshot: bool = False,
    ) -> None:
        """Create an ephemeral profile, narrow bridge, and trusted page."""
        super().__init__(parent)
        application = QApplication.instance()
        profile = QWebEngineProfile(application)
        self.destroyed.connect(profile.deleteLater)
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies,
        )
        profile.downloadRequested.connect(self._reject_download)
        self._renderer_scheme_handler = PackagedRendererSchemeHandler(profile)
        profile.installUrlSchemeHandler(APP_SCHEME.encode(), self._renderer_scheme_handler)
        self._scheme_handler: ManagedSceneSchemeHandler | None = None
        if resolver is not None:
            self._scheme_handler = ManagedSceneSchemeHandler(resolver, profile)
            profile.installUrlSchemeHandler(b"bbscene", self._scheme_handler)

        self._snapshot = snapshot
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
        self._renderer_ready = False
        self._scene_clear_pending = False
        self._bridge.ready.connect(self._renderer_became_ready)
        self._bridge.scene_progressed.connect(
            lambda _asset_id, loaded, total: self.scene_progressed.emit(loaded, total),
        )
        self._bridge.scene_failed.connect(
            lambda error: self.scene_failed.emit(error.message),
        )
        self._bridge.viewpoint_received.connect(self.viewpoint_changed)
        self._channel = QWebChannel(self)
        self._channel.registerObject("rendererBridge", self._bridge)
        page.setWebChannel(self._channel)
        self.setPage(page)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAccessibleName("Secure reconstructed scene renderer")
        self._scene: SceneReference | None = None
        self._viewpoint = Viewpoint()
        self._load_placeholder()

    def _renderer_became_ready(self) -> None:
        self._renderer_ready = True
        self._send_pending_scene()

    def _load_placeholder(self) -> None:
        template_name = "background.html" if self._snapshot else "viewer.html"
        template = (
            files("better_backgrounds.desktop")
            .joinpath("assets", template_name)
            .read_text(encoding="utf-8")
        )
        nonce = secrets.token_urlsafe(24)
        html = template.replace("{{NONCE}}", nonce)
        self.setHtml(html, QUrl(f"{TRUSTED_ORIGIN}/viewer.html"))

    def set_scene(self, scene: SceneReference, viewpoint: Viewpoint) -> None:
        """Load one managed scene without recreating the browser surface."""
        if self._scene is not None and self._scene.asset_id == scene.asset_id:
            self.set_viewpoint(viewpoint)
            return
        self._scene = scene
        self._scene_clear_pending = False
        self._viewpoint = viewpoint
        self._send_pending_scene()

    def clear_scene(self) -> None:
        """Remove the current scene when the selected room is unavailable."""
        self._scene = None
        self._scene_clear_pending = True
        self._send_pending_scene()

    def set_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Apply Python-owned controls through the validated bridge."""
        self._viewpoint = viewpoint
        self._bridge.request_viewpoint(viewpoint)

    def reset_viewpoint(self) -> None:
        """Ask the active scene to restore its safe default preset."""
        self._bridge.request_reset()

    def _send_pending_scene(self) -> None:
        if not self._renderer_ready:
            return
        if self._scene is not None:
            self._bridge.request_scene(
                self._scene.asset_id,
                self._scene.managed_url.toString(),
                self._viewpoint,
            )
        elif self._scene_clear_pending:
            self._bridge.request_scene_clear()
            self._scene_clear_pending = False

    @staticmethod
    def _reject_download(download: QWebEngineDownloadRequest) -> None:
        download.cancel()


def create_renderer_view(resolver: ManagedSceneResolver | None = None) -> QWidget:
    """Create the production embedded renderer surface."""
    return SecureRendererView(resolver)


def create_background_renderer_view(resolver: ManagedSceneResolver | None = None) -> QWidget:
    """Create a canvas-only room renderer for atomic native snapshots."""
    return SecureRendererView(resolver, snapshot=True)
