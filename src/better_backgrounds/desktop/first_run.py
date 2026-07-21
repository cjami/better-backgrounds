"""First-run setup that downloads every mandatory model before the shell opens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from better_backgrounds.model_setup import (
    all_models_ready,
    missing_models,
    prepare_models,
    total_download_size,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

GIB = 1024**3
SHARP_LICENSE_URL = (
    "https://github.com/apple/ml-sharp/blob/1eaa046834b81852261262b41b0919f5c1efdd2e/LICENSE_MODEL"
)


class _PreparationThread(QThread):
    """Download every missing model off the GUI thread."""

    advanced = Signal(int, int)
    failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Start cancellable preparation for the pending models."""
        super().__init__(parent)
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation during the streamed download."""
        self._cancelled = True

    def run(self) -> None:
        """Prepare the models and report one combined progress total."""
        try:
            prepare_models(
                license_accepted=True,
                progress=self.advanced.emit,
                is_cancelled=lambda: self._cancelled,
            )
        except Exception as error:  # noqa: BLE001 - reported verbatim to the user
            if not self._cancelled:
                self.failed.emit(str(error)[:300])


class ModelSetupDialog(QDialog):
    """Ask once for the SHARP model license, then fetch every mandatory model."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Compose the one-screen first-run setup step."""
        super().__init__(parent)
        self.setWindowTitle("Better Backgrounds setup")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._thread: _PreparationThread | None = None

        pending = missing_models()
        total = total_download_size()

        layout = QVBoxLayout(self)
        heading = QLabel("Better Backgrounds needs to download its models before first use.")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        details = QLabel(
            "\n".join(
                f"• {status.label} — {status.size / GIB:.2f} GiB ({status.license_name})"
                for status in pending
            )
            + f"\n\nTotal download: {total / GIB:.2f} GiB. "
            "This happens once; afterwards the application runs offline.",
        )
        details.setWordWrap(True)
        layout.addWidget(details)

        self._license = QCheckBox(
            "I accept Apple's SHARP research-only model license, which permits "
            "non-commercial scientific research and excludes product development.",
        )
        self._license.setToolTip(SHARP_LICENSE_URL)
        layout.addWidget(self._license)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        self._download = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._download.setText("Download models")
        self._download.setEnabled(False)
        self._buttons.accepted.connect(self._start)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._license.toggled.connect(self._download.setEnabled)

    def _start(self) -> None:
        if not self._license.isChecked():
            return
        self._download.setEnabled(False)
        self._license.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Downloading…")
        thread = _PreparationThread(self)
        thread.advanced.connect(self._advance)
        thread.failed.connect(self._fail)
        thread.finished.connect(self._finish)
        self._thread = thread
        thread.start()

    def _advance(self, completed: int, total: int) -> None:
        self._progress.setValue(int(completed * 100 / total) if total else 100)
        self._status.setText(f"Downloading… {completed / GIB:.2f} / {total / GIB:.2f} GiB")

    def _fail(self, message: str) -> None:
        self._status.setText(f"Download failed: {message}")
        self._license.setEnabled(True)
        self._download.setEnabled(True)
        self._download.setText("Retry download")

    def _finish(self) -> None:
        if all_models_ready():
            self.accept()

    def reject(self) -> None:
        """Cancel any in-flight download before closing."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.cancel()
            thread.wait(5_000)
        super().reject()


def ensure_models_ready(parent: QWidget | None = None) -> bool:
    """Return whether every mandatory model is prepared, prompting once if needed."""
    if all_models_ready():
        return True
    dialog = ModelSetupDialog(parent)
    dialog.exec()
    return all_models_ready()


__all__ = ["ModelSetupDialog", "ensure_models_ready"]
