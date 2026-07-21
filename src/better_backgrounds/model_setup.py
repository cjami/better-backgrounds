"""First-run preparation of the three mandatory Better Backgrounds models.

SHARP, MatAnyone 2, and PIH are all required. None of them is configurable and none
has an alternative backend: the application prepares every checkpoint once into the
managed model cache and then runs entirely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from better_backgrounds.harmonization.checkpoint import PihCheckpointInstaller
from better_backgrounds.matting.runtime import MatAnyoneCheckpointInstaller
from better_backgrounds.reconstruction.sharp.checkpoint import (
    SharpCheckpointInstaller,
    default_sharp_model_root,
)

if TYPE_CHECKING:
    from better_backgrounds.checkpoints import (
        CancellationCheck,
        CheckpointProgress,
        ManagedCheckpointInstaller,
    )

SHARP_KEY = "sharp"
MATANYONE_KEY = "matanyone"
PIH_KEY = "pih"


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Report one mandatory model's identity, size, and readiness."""

    key: str
    label: str
    ready: bool
    size: int
    license_name: str
    requires_license_acceptance: bool


def sharp_installer() -> SharpCheckpointInstaller:
    """Return the managed SHARP installer."""
    return SharpCheckpointInstaller(default_sharp_model_root())


def matanyone_installer() -> MatAnyoneCheckpointInstaller:
    """Return the managed MatAnyone 2 installer."""
    return MatAnyoneCheckpointInstaller()


def pih_installer() -> PihCheckpointInstaller:
    """Return the managed PIH installer."""
    return PihCheckpointInstaller()


def model_installers() -> list[tuple[str, str, ManagedCheckpointInstaller]]:
    """Return every mandatory model in preparation order, smallest first."""
    return [
        (MATANYONE_KEY, "MatAnyone 2", matanyone_installer()),
        (PIH_KEY, "Adobe PIH", pih_installer()),
        (SHARP_KEY, "Apple SHARP", sharp_installer()),
    ]


def model_statuses() -> list[ModelStatus]:
    """Report readiness for every mandatory model without changing the cache."""
    return [
        ModelStatus(
            key=key,
            label=label,
            ready=installer.is_ready(),
            size=installer.identity.size,
            license_name=installer.license_name,
            requires_license_acceptance=installer.requires_license_acceptance,
        )
        for key, label, installer in model_installers()
    ]


def missing_models() -> list[ModelStatus]:
    """Return the mandatory models that still need downloading."""
    return [status for status in model_statuses() if not status.ready]


def all_models_ready() -> bool:
    """Return whether every mandatory model is prepared for offline use."""
    return not missing_models()


def total_download_size() -> int:
    """Return the byte total the first-run setup still needs to download."""
    return sum(status.size for status in missing_models())


def prepare_models(
    *,
    license_accepted: bool,
    progress: CheckpointProgress | None = None,
    is_cancelled: CancellationCheck = lambda: False,
) -> None:
    """Prepare every missing mandatory model, reporting one combined progress total.

    ``progress`` receives bytes completed and the combined byte total across every
    model that still needs downloading, so one bar can cover the whole setup step.
    """
    pending = [
        installer for _key, _label, installer in model_installers() if not installer.is_ready()
    ]
    if not pending:
        return
    total = sum(installer.identity.size for installer in pending)
    completed_before = 0
    for installer in pending:

        def relay(done: int, _model_total: int, *, base: int = completed_before) -> None:
            if progress is not None:
                progress(base + done, total)

        installer.prepare(
            license_accepted=license_accepted,
            progress=relay,
            is_cancelled=is_cancelled,
        )
        completed_before += installer.identity.size
