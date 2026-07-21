"""NDJSON worker entry point for direct Gaussian scene imports."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from better_backgrounds.jobs.events import CancelledEvent, ErrorEvent, JobEvent, ResultEvent
from better_backgrounds.reconstruction.splats import (
    SplatImportCancelledError,
    SplatImportConfig,
    SplatImportRequest,
    SplatSceneImporter,
    SplatSelection,
)
from better_backgrounds.scene import SceneCatalogue

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def emit_stdout(event: JobEvent) -> None:
    """Write one validated event to the supervising desktop process."""
    sys.stdout.write(event.model_dump_json() + "\n")
    sys.stdout.flush()


class SplatImportWorker:
    """Own one cancellable Gaussian scene import inside a dedicated process."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        job_id: str,
        source: Path,
        scene_cache_root: Path,
        catalogue_path: Path,
        emit: Callable[[JobEvent], None] = emit_stdout,
        importer: SplatSceneImporter | None = None,
    ) -> None:
        """Keep the worker serializable and independently testable."""
        self.job_id = job_id
        self.source = source
        self.scene_cache_root = scene_cache_root
        self.catalogue_path = catalogue_path
        self.emit = emit
        self.importer = importer or SplatSceneImporter()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        self._cancelled.set()

    def run(self) -> int:
        """Import and catalogue one scene, emitting exactly one terminal event."""
        try:
            reference = self.importer.import_scene(
                SplatImportRequest(
                    job_id=self.job_id,
                    selection=SplatSelection(self.source.name, self.source),
                    config=SplatImportConfig(self.scene_cache_root),
                ),
                self.emit,
                self._cancelled.is_set,
            )
            SceneCatalogue(self.catalogue_path).save(reference)
        except SplatImportCancelledError:
            self.emit(CancelledEvent(job_id=self.job_id, message="Splat import cancelled."))
            return 0
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.emit(
                ErrorEvent(
                    job_id=self.job_id,
                    code="splat_import_failed",
                    message=str(error)[:500] or "The Gaussian scene could not be imported.",
                    recovery_action=(
                        "Choose a compatible Gaussian PLY, standalone SOG bundle, or packaged "
                        "Streamed SOG archive and retry."
                    ),
                ),
            )
            return 1
        self.emit(
            ResultEvent(
                job_id=self.job_id,
                scene_id=reference.asset_id,
                message="Imported room is ready.",
            ),
        )
        return 0
