"""Persistent device-specific MatAnyone calibration profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from pathlib import Path

CALIBRATION_POLICY_VERSION = 1


class CalibrationIdentity(BaseModel):
    """Identify every input that can invalidate a measured inference size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_revision: str
    device_type: str
    device_name: str
    torch_version: str
    accelerator_version: str
    capture_width: int = Field(gt=0)
    capture_height: int = Field(gt=0)
    latency_budget_ms: float = Field(gt=0, allow_inf_nan=False)
    policy_version: Literal[1] = CALIBRATION_POLICY_VERSION


class CalibrationProfile(BaseModel):
    """Store one validated resolution for a concrete runtime identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: CalibrationIdentity
    selected_internal_size: Literal[360, 432, 540]
    measured_p95_ms: float = Field(ge=0, allow_inf_nan=False)


class CalibrationProfiles(BaseModel):
    """Version the complete bounded collection of local profiles."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profiles: list[CalibrationProfile] = Field(default_factory=list, max_length=32)


class CalibrationProfileStore:
    """Load and atomically replace local calibration evidence."""

    def __init__(self, path: Path | None) -> None:
        """Use optional application data without making persistence mandatory."""
        self.path = path

    def find(self, identity: CalibrationIdentity) -> CalibrationProfile | None:
        """Return the exact matching profile, ignoring invalid persistence."""
        profiles = self._load()
        return next(
            (profile for profile in profiles.profiles if profile.identity == identity),
            None,
        )

    def save(self, profile: CalibrationProfile) -> None:
        """Upsert one profile without making calibration persistence mandatory."""
        path = self.path
        if path is None:
            return
        profiles = self._load()
        retained = [item for item in profiles.profiles if item.identity != profile.identity]
        retained.append(profile)
        document = CalibrationProfiles(profiles=retained[-32:])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            return

    def _load(self) -> CalibrationProfiles:
        path = self.path
        if path is None:
            return CalibrationProfiles()
        try:
            return CalibrationProfiles.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError, ValidationError:
            return CalibrationProfiles()
