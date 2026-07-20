"""Build-page upload, review, progress, and completion states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from better_backgrounds.desktop.pages.common import card as _card
from better_backgrounds.desktop.pages.common import label as _label

if TYPE_CHECKING:
    from PySide6.QtCore import QMimeData
    from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QImage

    from better_backgrounds.reconstruction import SplatDiagnostics, SplatSelection
    from better_backgrounds.reconstruction.sharp import SceneImageDiagnostics, SceneImageSelection


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SPLAT_EXTENSIONS = frozenset({".ply", ".ssog", ".zip"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | SPLAT_EXTENSIONS


class _MimeEvent(Protocol):
    def mimeData(self) -> QMimeData:  # noqa: N802
        """Return the event's mime payload."""


STAGE_ORDER = (
    ("validation", "Validating image"),
    ("model_preparation", "Preparing model"),
    ("model_loading", "Loading model"),
    ("inference", "Predicting Gaussians"),
    ("ply_validation", "Validating PLY"),
    ("publication", "Publishing scene"),
    ("preview_generation", "Preparing preview"),
)
COMPLETE_PROGRESS = 100


class BuildPage(QWidget):
    """Build from one room image or import one Gaussian scene."""

    file_requested = Signal()
    file_dropped = Signal(str)
    capture_requested = Signal()
    capture_now_requested = Signal()
    capture_cancelled = Signal()
    build_requested = Signal(str)
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create upload, capture, review, and processing states."""
        super().__init__(parent)
        # Paint an opaque backdrop so the stacked live preview never shows
        # through and the Build tab reads as a solid page, not an overlay.
        self.setObjectName("buildPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # noqa: FBT003
        self.setAcceptDrops(True)
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 32, 52, 32)
        self._content = QStackedWidget()
        self._content.setObjectName("buildContent")
        root.addWidget(self._content)
        self._drop_card: QFrame | None = None
        self._create_upload()
        self._create_capture()
        self._create_review()
        self._create_progress()
        self._active_stage_order = STAGE_ORDER
        self._importing = False

    def _create_upload(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addStretch()
        layout.addWidget(
            _label("NEW ROOM", object_name="eyebrow"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(
            _label("Add a room", object_name="heroTitle"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        subtitle = _label(
            "Choose an empty room photo, import an existing 3D room, or use your webcam.",
            object_name="subtitle",
            word_wrap=True,
        )
        subtitle.setMaximumWidth(650)
        subtitle.setMinimumHeight(44)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(subtitle, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10)
        drop = QFrame()
        drop.setObjectName("dropCard")
        drop.setProperty("dragActive", False)  # noqa: FBT003
        drop.setMinimumSize(620, 200)
        drop.setMaximumWidth(760)
        self._drop_card = drop
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(38, 28, 38, 28)
        drop_layout.setSpacing(8)
        drop_layout.addWidget(
            _label("↑", object_name="uploadIcon"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addStretch()
        choose = QPushButton("Choose a file")
        choose.setObjectName("dropAction")
        choose.clicked.connect(self.file_requested)
        actions.addWidget(choose)
        capture = QPushButton("Capture from camera")
        capture.setObjectName("quietAction")
        capture.clicked.connect(self.capture_requested)
        actions.addWidget(capture)
        actions.addStretch()
        drop_layout.addLayout(actions)
        drop_layout.addWidget(
            _label("ROOM PHOTO  ·  3D ROOM", object_name="feedMeta"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(drop, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        self._content.addWidget(page)

    def _create_capture(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        card, card_layout = _card()
        card.setMaximumWidth(780)
        card_layout.addWidget(_label("Capture the empty room", object_name="title"))
        self._capture_instruction = _label(
            "Step out of frame — we'll photograph the room your webcam sees.",
            object_name="subtitle",
            word_wrap=True,
        )
        card_layout.addWidget(self._capture_instruction)
        self._capture_preview = QLabel()
        self._capture_preview.setObjectName("capturePreview")
        self._capture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_preview.setFixedSize(720, 405)
        self._capture_preview.setAccessibleName("Room capture camera preview")
        card_layout.addWidget(self._capture_preview)
        self._countdown = _label("", object_name="countdown")
        self._countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self._countdown)
        footer = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("quiet")
        cancel.clicked.connect(self.capture_cancelled)
        footer.addWidget(cancel)
        footer.addStretch()
        self._capture_now = QPushButton("Capture now")
        self._capture_now.setObjectName("primary")
        self._capture_now.clicked.connect(self.capture_now_requested)
        footer.addWidget(self._capture_now)
        card_layout.addLayout(footer)
        layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        self._content.addWidget(page)

    def _create_review(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        card, card_layout = _card()
        card.setMaximumWidth(780)
        header = QHBoxLayout()
        self._selection_name = _label("", object_name="title")
        header.addWidget(self._selection_name)
        header.addStretch()
        self._readiness = _label("READY TO REVIEW", object_name="success")
        header.addWidget(self._readiness)
        card_layout.addLayout(header)
        self._review_summary = _label(
            "Check the source before building your room.",
            object_name="subtitle",
            word_wrap=True,
        )
        card_layout.addWidget(self._review_summary)
        self._image_preview = QLabel()
        self._image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_preview.setMinimumHeight(180)
        self._image_preview.setMaximumHeight(260)
        self._image_preview.setAccessibleName("Oriented room image preview")
        card_layout.addWidget(self._image_preview)
        diagnostics = (
            ("dimensions", "Oriented dimensions"),
            ("format", "Format"),
            ("orientation", "EXIF orientation"),
            ("focal", "Focal metadata"),
            ("alpha", "Transparency"),
        )
        self._diagnostic_values: dict[str, QLabel] = {}
        self._diagnostic_titles: dict[str, QLabel] = {}
        for key, title in diagnostics:
            row = QHBoxLayout()
            row.addWidget(_label("✓", object_name="success"))
            title_label = _label(title)
            self._diagnostic_titles[key] = title_label
            row.addWidget(title_label)
            row.addStretch()
            value = _label("—", object_name="muted")
            self._diagnostic_values[key] = value
            row.addWidget(value)
            card_layout.addLayout(row)
        footer = QHBoxLayout()
        back = QPushButton("Choose another image")
        back.setObjectName("quiet")
        back.clicked.connect(self.show_upload)
        footer.addWidget(back)
        footer.addStretch()
        self._model_status = _label("Checking room builder…", object_name="muted")
        footer.addWidget(self._model_status)
        footer.addStretch()
        self._device_label = _label("Device", object_name="muted")
        footer.addWidget(self._device_label)
        self._device = QComboBox()
        self._device.setObjectName("sharpDevice")
        self._device.setAccessibleName("Room build device")
        self._device.addItem("Automatic", "auto")
        self._device.addItem("CUDA", "cuda")
        self._device.addItem("Apple MPS", "mps")
        self._device.addItem("CPU", "cpu")
        footer.addWidget(self._device)
        self._build_action = QPushButton("Build room")
        self._build_action.setObjectName("primary")
        self._build_action.clicked.connect(self._emit_build)
        self._build_action.setEnabled(False)
        footer.addWidget(self._build_action)
        card_layout.addLayout(footer)
        layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        self._content.addWidget(page)

    def _create_progress(self) -> None:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setSpacing(22)
        stages_card, stages_layout = _card()
        stages_card.setMaximumWidth(410)
        stages_layout.addWidget(_label("Building your room", object_name="title"))
        stages_layout.addWidget(
            _label("You can use the other tabs while this runs.", object_name="subtitle"),
        )
        self._stage_labels: dict[str, QLabel] = {}
        for key, title in STAGE_ORDER:
            label = _label(f"○  {title}", object_name="stagePending")
            label.setMinimumHeight(34)
            stages_layout.addWidget(label)
            self._stage_labels[key] = label
        stages_layout.addStretch()
        root.addWidget(stages_card, 2)

        detail_card, detail_layout = _card()
        self._status = _label("Preparing the build", object_name="section", word_wrap=True)
        detail_layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setAccessibleName("Room build progress")
        detail_layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setAccessibleName("Build event log")
        detail_layout.addWidget(self._log, 1)
        self._message = _label("", object_name="danger", word_wrap=True)
        self._message.hide()
        detail_layout.addWidget(self._message)
        actions = QHBoxLayout()
        actions.addStretch()
        self._retry = QPushButton("Retry")
        self._retry.setObjectName("primary")
        self._retry.clicked.connect(self.retry_requested)
        self._retry.hide()
        actions.addWidget(self._retry)
        self._new_build = QPushButton("Build another room")
        self._new_build.setObjectName("primary")
        self._new_build.clicked.connect(self.show_upload)
        self._new_build.hide()
        actions.addWidget(self._new_build)
        self._cancel = QPushButton("Cancel build")
        self._cancel.setObjectName("danger")
        self._cancel.clicked.connect(self.cancel_requested)
        actions.addWidget(self._cancel)
        detail_layout.addLayout(actions)
        root.addWidget(detail_card, 3)
        self._content.addWidget(page)

    def show_upload(self) -> None:
        """Show the upload surface."""
        self._content.setCurrentIndex(0)

    def show_capture(self) -> None:
        """Show the live webcam preview and countdown before capture."""
        self._capture_preview.clear()
        self._capture_preview.setText("Requesting camera…")
        self._countdown.setText("")
        self._capture_instruction.setText(
            "Step out of frame — we'll photograph the room your webcam sees.",
        )
        self._capture_now.setEnabled(True)
        self._content.setCurrentIndex(1)

    def set_capture_frame(self, image: QImage) -> None:
        """Paint one live camera frame into the capture preview."""
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        self._capture_preview.setPixmap(
            pixmap.scaled(
                self._capture_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_countdown(self, seconds: int) -> None:
        """Show the remaining seconds before the automatic capture."""
        self._countdown.setText(str(seconds) if seconds > 0 else "")

    def set_capture_error(self, message: str) -> None:
        """Explain why the room could not be captured from the camera."""
        self._capture_instruction.setText(message)
        self._capture_preview.clear()
        self._capture_preview.setText("Camera unavailable")
        self._countdown.setText("")
        self._capture_now.setEnabled(False)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        """Accept a drag only when it carries one supported local file."""
        if _first_supported_path(event) is not None:
            self._set_drag_active(active=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        """Keep accepting a valid drag as the pointer moves across the page."""
        if _first_supported_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        """Clear the drop highlight when the drag leaves the page."""
        self._set_drag_active(active=False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        """Emit the first supported dropped path for controller dispatch."""
        self._set_drag_active(active=False)
        path = _first_supported_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.file_dropped.emit(path)

    def _set_drag_active(self, *, active: bool) -> None:
        card = self._drop_card
        if card is None or bool(card.property("dragActive")) == active:
            return
        card.setProperty("dragActive", active)
        card.style().unpolish(card)
        card.style().polish(card)

    def show_review(
        self,
        selection: SceneImageSelection,
        diagnostics: SceneImageDiagnostics | None = None,
    ) -> None:
        """Show the oriented image, dimensions, and pre-inference warnings."""
        self._set_review_mode(importing=False)
        self._selection_name.setText(selection.display_name)
        self._content.setCurrentIndex(2)
        if selection.source_path is None or diagnostics is None:
            self._image_preview.clear()
            self._readiness.setText("PREPARED SMOKE INPUT")
            self._set_label_style(self._readiness, "success")
            self._review_summary.setText("The deterministic desktop smoke build is ready.")
            for value in self._diagnostic_values.values():
                value.setText("Prepared")
            self._build_action.setEnabled(True)
            return
        with Image.open(selection.source_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            oriented.thumbnail((680, 250), Image.Resampling.LANCZOS)
            oriented = oriented.convert("RGBA")
            pixmap = QPixmap.fromImage(ImageQt(oriented))
        self._image_preview.setPixmap(
            pixmap.scaled(
                QSize(680, 250),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._diagnostic_values["dimensions"].setText(f"{diagnostics.width} x {diagnostics.height}")
        self._diagnostic_values["format"].setText(diagnostics.image_format)
        self._diagnostic_values["orientation"].setText(
            "Applied" if diagnostics.orientation_applied else "Already upright"
        )
        self._diagnostic_values["focal"].setText(
            "Embedded" if diagnostics.has_focal_metadata else "30 mm default"
        )
        self._diagnostic_values["alpha"].setText(
            "Flatten to white" if diagnostics.has_alpha else "Opaque"
        )
        self._readiness.setText("READY")
        self._set_label_style(self._readiness, "success")
        self._review_summary.setText(
            diagnostics.warnings[0] if diagnostics.warnings else "The image is ready to build."
        )
        self._build_action.setEnabled(True)

    def show_splat_review(
        self,
        selection: SplatSelection,
        diagnostics: SplatDiagnostics | None = None,
    ) -> None:
        """Show direct-import scene metadata without decoding a pixel preview."""
        self._set_review_mode(importing=True)
        self._selection_name.setText(selection.display_name)
        self._content.setCurrentIndex(2)
        self._image_preview.clear()
        self._image_preview.setText("3D room ready to import")
        self._readiness.setText("READY" if diagnostics is not None else "ROOM UNAVAILABLE")
        self._set_label_style(
            self._readiness,
            "success" if diagnostics is not None else "danger",
        )
        self._review_summary.setText(
            "The scene will be copied into the local library; the source stays untouched."
        )
        if diagnostics is None:
            for value in self._diagnostic_values.values():
                value.setText("—")
            self._build_action.setEnabled(False)
            return
        size_mib = diagnostics.file_size / 1024**2
        values = {
            "dimensions": f"{diagnostics.gaussian_count:,}",
            "format": diagnostics.encoding,
            "orientation": (
                f"{diagnostics.lod_levels} LOD levels / {diagnostics.resource_count} files"
                if diagnostics.layout == "streamed-sog"
                else diagnostics.layout.title()
            ),
            "focal": diagnostics.framing,
            "alpha": f"{size_mib:.1f} MiB",
        }
        for key, value in values.items():
            self._diagnostic_values[key].setText(value)
        self._build_action.setEnabled(True)

    def set_image_error(self, message: str) -> None:
        """Show an actionable decode or validation failure."""
        self._readiness.setText("ROOM UNAVAILABLE" if self._importing else "IMAGE UNAVAILABLE")
        self._set_label_style(self._readiness, "danger")
        self._review_summary.setText(message)
        self._build_action.setEnabled(False)

    def set_model_ready(self, *, ready: bool) -> None:
        """Describe whether the research checkpoint is cached for offline use."""
        self._model_status.setText("Room builder ready" if ready else "Room builder setup required")

    def reset_progress(self, *, importing: bool = False) -> None:
        """Prepare the progress surface for a new job."""
        self._active_stage_order = (
            (
                ("validation", "Reading scene"),
                ("ply_validation", "Validating splats"),
                ("publication", "Publishing scene"),
            )
            if importing
            else STAGE_ORDER
        )
        self._content.setCurrentIndex(3)
        self._status.setText("Preparing the build")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._log.clear()
        self._message.hide()
        self._retry.hide()
        self._new_build.hide()
        self._cancel.show()
        active_keys = {key for key, _title in self._active_stage_order}
        for key, title in STAGE_ORDER:
            label = self._stage_labels[key]
            label.setVisible(key in active_keys)
            label.setText(f"○  {title}")
            self._set_label_style(label, "stagePending")

    def set_progress(self, stage: str, progress: float | None, message: str) -> None:
        """Render one validated progress or warning event."""
        self._status.setText(message)
        if progress is None:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(round(progress * 100))
        active_index = next(
            (index for index, item in enumerate(self._active_stage_order) if item[0] == stage),
            0,
        )
        for index, (key, title) in enumerate(self._active_stage_order):
            label = self._stage_labels[key]
            if index < active_index:
                label.setText(f"✓  {title}")
                object_name = "stageDone"
            elif index == active_index:
                label.setText(f"●  {title}")
                object_name = "stageActive"
            else:
                label.setText(f"○  {title}")
                object_name = "stagePending"
            self._set_label_style(label, object_name)
        self._log.append(f"> {message}")

    def set_failed(self, message: str, recovery: str | None) -> None:
        """Show a stable user-safe failure and retry action."""
        text = message if recovery is None else f"{message}\n{recovery}"
        self._message.setText(text)
        self._set_label_style(self._message, "danger")
        self._message.show()
        self._retry.show()
        self._cancel.hide()

    def set_completed(self, room_name: str) -> None:
        """Keep a successful build visible without changing tabs."""
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._status.setText("Room ready")
        self._message.setText(f"{room_name} is now selected in Show and Adjust.")
        self._set_label_style(self._message, "success")
        self._message.show()
        self._cancel.hide()
        self._new_build.show()
        self._log.append("> Runtime room is ready")
        for key, title in self._active_stage_order:
            label = self._stage_labels[key]
            label.setText(f"✓  {title}")
            self._set_label_style(label, "stageDone")

    def _emit_build(self) -> None:
        self.build_requested.emit(str(self._device.currentData()))

    def _set_review_mode(self, *, importing: bool) -> None:
        self._importing = importing
        titles = (
            ("Splats", "Encoding", "Layout", "Framing", "File size")
            if importing
            else (
                "Oriented dimensions",
                "Format",
                "EXIF orientation",
                "Focal metadata",
                "Transparency",
            )
        )
        for key, title in zip(self._diagnostic_titles, titles, strict=True):
            self._diagnostic_titles[key].setText(title)
        self._model_status.setVisible(not importing)
        self._device_label.setVisible(not importing)
        self._device.setVisible(not importing)
        self._build_action.setText("Import room" if importing else "Build room")

    @staticmethod
    def _set_label_style(label: QLabel, object_name: str) -> None:
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)


def _first_supported_path(event: _MimeEvent) -> str | None:
    """Return the first dragged local file with a supported extension."""
    mime = event.mimeData()
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        local = url.toLocalFile()
        suffix = local[local.rfind(".") :].lower() if "." in local else ""
        if suffix in SUPPORTED_EXTENSIONS:
            return local
    return None
