"""Contracts shared by the global harmonization runtime and compositor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class HarmonizationSettings(BaseModel):
    """Store the single room-scoped global harmonization switch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_harmonization: bool = False

    @property
    def active(self) -> bool:
        """Return whether global harmonization was explicitly requested."""
        return self.global_harmonization


@dataclass(frozen=True, slots=True)
class HarmonizationResult:
    """Return processed pixels and measurable per-frame evidence."""

    image: NDArray[np.uint8] | None
    processing_ms: float
    degraded_components: tuple[str, ...]
    applied: bool
