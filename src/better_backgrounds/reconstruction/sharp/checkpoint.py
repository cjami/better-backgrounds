"""Pinned SHARP checkpoint discovery, validation, and installation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import TYPE_CHECKING

from platformdirs import user_cache_path

from better_backgrounds.checkpoints import (
    CancellationCheck,
    CheckpointIdentity,
    CheckpointProgress,
    ManagedCheckpointInstaller,
    ResourceOpener,
    open_managed_url,
)
from better_backgrounds.reconstruction.sharp.contracts import (
    SharpCancelledError,
    SharpCapabilities,
    SharpCheckpointManifest,
)
from better_backgrounds.reconstruction.sharp.runtime import (
    SharpDeviceRequest,
    ensure_vendored_sharp,
    resolve_sharp_device,
)

if TYPE_CHECKING:
    from pathlib import Path

SHARP_BUILDER_REVISION = "1eaa046834b81852261262b41b0919f5c1efdd2e"
APPLICATION_NAME = "Better Backgrounds"
SHARP_DEPENDENCY_VERSIONS = {
    "plyfile": "1.1.2",
    "scipy": "1.16.2",
    "timm": "1.0.20",
}
__all__ = [
    "SHARP_BUILDER_REVISION",
    "CancellationCheck",
    "CheckpointProgress",
    "ResourceOpener",
    "SharpCheckpointInstaller",
    "load_sharp_checkpoint_manifest",
    "probe_sharp_capabilities",
]


def default_sharp_model_root() -> Path:
    """Return the managed SHARP cache directory used by the app and its workers."""
    return user_cache_path(APPLICATION_NAME, APPLICATION_NAME) / "models-v1" / "sharp"


def load_sharp_checkpoint_manifest() -> SharpCheckpointManifest:
    """Load the exact checked-in SHARP checkpoint identity."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/sharp/manifest-v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = SharpCheckpointManifest.model_validate_json(content)
    if manifest.builder_revision != SHARP_BUILDER_REVISION:
        msg = "The SHARP checkpoint manifest does not match the vendored source revision"
        raise ValueError(msg)
    return manifest


def probe_sharp_capabilities(
    requested: SharpDeviceRequest = "auto",
) -> SharpCapabilities:
    """Verify the pinned inference imports and resolve one usable device."""
    for package, expected in SHARP_DEPENDENCY_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            msg = f"The pinned SHARP dependency {package} is unavailable"
            raise RuntimeError(msg) from error
        if actual != expected:
            msg = f"SHARP requires {package}=={expected}, found {actual}"
            raise RuntimeError(msg)
    ensure_vendored_sharp()
    device = resolve_sharp_device(requested)
    return SharpCapabilities(device_type=device, accelerated=device != "cpu")


class SharpCheckpointInstaller(ManagedCheckpointInstaller):
    """Download and atomically publish the license-gated official checkpoint."""

    cancelled_error = SharpCancelledError
    label = "SHARP"
    requires_license_acceptance = True

    def __init__(
        self,
        root: Path,
        *,
        manifest: SharpCheckpointManifest | None = None,
        opener: ResourceOpener = open_managed_url,
    ) -> None:
        """Use a dedicated model cache and injectable download boundary."""
        self.manifest = manifest or load_sharp_checkpoint_manifest()
        self.license_name = self.manifest.license_name
        super().__init__(
            root,
            CheckpointIdentity(
                filename=self.manifest.filename,
                url=str(self.manifest.url),
                size=self.manifest.size,
                sha256=self.manifest.sha256,
            ),
            opener=opener,
        )
