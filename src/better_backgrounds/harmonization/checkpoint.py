"""Pinned PIH checkpoint identity, discovery, and managed installation."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Literal

from platformdirs import user_cache_path
from pydantic import Field, HttpUrl

from better_backgrounds.checkpoints import (
    CheckpointIdentity,
    ManagedCheckpointInstaller,
    ResourceOpener,
    open_managed_url,
)
from better_backgrounds.scene.models import StrictModel

if TYPE_CHECKING:
    from pathlib import Path

PIH_BUILDER_REVISION = "2823cccf0778c6ea213a3d366f03864ac8ab82e6"
APPLICATION_NAME = "Better Backgrounds"


class PihCancelledError(RuntimeError):
    """Raised when a PIH checkpoint preparation cooperatively stops."""


class PihCheckpointManifest(StrictModel):
    """Pin the official PIH checkpoint bytes and their upstream identity."""

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1, max_length=80)
    filename: str = Field(pattern=r"^[a-zA-Z0-9_.-]+\.pth$")
    url: HttpUrl
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    builder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_name: str = Field(min_length=1, max_length=100)
    license_url: HttpUrl


def load_pih_checkpoint_manifest() -> PihCheckpointManifest:
    """Load the exact checked-in PIH checkpoint identity."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/pih/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = PihCheckpointManifest.model_validate_json(content)
    if manifest.builder_revision != PIH_BUILDER_REVISION:
        msg = "The PIH checkpoint manifest does not match the vendored source revision"
        raise ValueError(msg)
    return manifest


def default_pih_model_root() -> Path:
    """Return the managed PIH cache directory used by the app and its workers."""
    return user_cache_path(APPLICATION_NAME, APPLICATION_NAME) / "models-v1" / "pih"


class PihCheckpointInstaller(ManagedCheckpointInstaller):
    """Download and atomically publish the pinned PIH checkpoint."""

    cancelled_error = PihCancelledError
    label = "PIH"

    def __init__(
        self,
        root: Path | None = None,
        *,
        manifest: PihCheckpointManifest | None = None,
        opener: ResourceOpener = open_managed_url,
    ) -> None:
        """Use the managed model cache and an injectable download boundary."""
        self.manifest = manifest or load_pih_checkpoint_manifest()
        self.license_name = self.manifest.license_name
        super().__init__(
            root if root is not None else default_pih_model_root(),
            CheckpointIdentity(
                filename=self.manifest.filename,
                url=str(self.manifest.url),
                size=self.manifest.size,
                sha256=self.manifest.sha256,
            ),
            opener=opener,
        )
