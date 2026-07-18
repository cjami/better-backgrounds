"""Build-page upload, review, progress, and completion states."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from better_backgrounds.reconstruction.sharp import SceneImageDiagnostics, SceneImageSelection


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
    """Upload and process one room image within a self-contained tab."""

    image_requested = Signal()
    build_requested = Signal(str)
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create upload, review, and processing states."""
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 32, 52, 32)
        self._content = QStackedWidget()
        self._content.setObjectName("buildContent")
        root.addWidget(self._content)
        self._create_upload()
        self._create_review()
        self._create_progress()

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
            _label("Turn one photo into a room", object_name="heroTitle"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        subtitle = _label(
            "Choose a clear room photo. Apple SHARP creates a metric Gaussian scene "
            "for nearby viewpoint changes, entirely on this device.",
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
        drop.setMinimumSize(620, 180)
        drop.setMaximumWidth(760)
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(38, 28, 38, 28)
        drop_layout.setSpacing(8)
        drop_layout.addWidget(
            _label("↑", object_name="uploadIcon"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        choose = QPushButton("Choose a room photo")
        choose.setObjectName("dropAction")
        choose.clicked.connect(self.image_requested)
        drop_layout.addWidget(choose, alignment=Qt.AlignmentFlag.AlignHCenter)
        drop_layout.addWidget(
            _label("JPEG  /  PNG  /  WEBP  ·  UPLOAD FIRST", object_name="feedMeta"),
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(drop, alignment=Qt.AlignmentFlag.AlignHCenter)
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
            "Review the oriented source before SHARP inference.",
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
        for key, title in diagnostics:
            row = QHBoxLayout()
            row.addWidget(_label("✓", object_name="success"))
            row.addWidget(_label(title))
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
        self._model_status = _label("SHARP model status pending", object_name="muted")
        footer.addWidget(self._model_status)
        footer.addStretch()
        footer.addWidget(_label("Device", object_name="muted"))
        self._device = QComboBox()
        self._device.setObjectName("sharpDevice")
        self._device.setAccessibleName("SHARP inference device")
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

    def show_review(
        self,
        selection: SceneImageSelection,
        diagnostics: SceneImageDiagnostics | None = None,
    ) -> None:
        """Show the oriented image, dimensions, and pre-inference warnings."""
        self._selection_name.setText(selection.display_name)
        self._content.setCurrentIndex(1)
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
            diagnostics.warnings[0]
            if diagnostics.warnings
            else "The image is ready for local SHARP inference."
        )
        self._build_action.setEnabled(True)

    def set_image_error(self, message: str) -> None:
        """Show an actionable decode or validation failure."""
        self._readiness.setText("IMAGE UNAVAILABLE")
        self._set_label_style(self._readiness, "danger")
        self._review_summary.setText(message)
        self._build_action.setEnabled(False)

    def set_model_ready(self, *, ready: bool) -> None:
        """Describe whether the research checkpoint is cached for offline use."""
        self._model_status.setText(
            "SHARP model ready offline" if ready else "SHARP model needs preparation"
        )

    def reset_progress(self) -> None:
        """Prepare the progress surface for a new job."""
        self._content.setCurrentIndex(2)
        self._status.setText("Preparing the build")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._log.clear()
        self._message.hide()
        self._retry.hide()
        self._new_build.hide()
        self._cancel.show()
        for key, title in STAGE_ORDER:
            label = self._stage_labels[key]
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
            (index for index, item in enumerate(STAGE_ORDER) if item[0] == stage),
            0,
        )
        for index, (key, title) in enumerate(STAGE_ORDER):
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
        self._message.setText(f"{room_name} is now selected in Show, Adjust, and Compare.")
        self._set_label_style(self._message, "success")
        self._message.show()
        self._cancel.hide()
        self._new_build.show()
        self._log.append("> Runtime room is ready")
        for key, title in STAGE_ORDER:
            label = self._stage_labels[key]
            label.setText(f"✓  {title}")
            self._set_label_style(label, "stageDone")

    def _emit_build(self) -> None:
        self.build_requested.emit(str(self._device.currentData()))

    @staticmethod
    def _set_label_style(label: QLabel, object_name: str) -> None:
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)
