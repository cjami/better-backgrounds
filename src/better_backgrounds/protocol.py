"""Versioned messages shared by desktop jobs and subprocess workers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = 1


class WireModel(BaseModel):
    """Reject undeclared values at every process boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EventEnvelope(WireModel):
    """Fields common to every job event."""

    schema_version: Literal[1] = SCHEMA_VERSION
    job_id: str = Field(min_length=1, max_length=128)


class ProgressEvent(EventEnvelope):
    """Report defensible progress for one bounded worker stage."""

    type: Literal["progress"] = "progress"
    stage: str = Field(min_length=1, max_length=80)
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str = Field(min_length=1, max_length=500)


class WarningEvent(EventEnvelope):
    """Report a recoverable concern without terminating the job."""

    type: Literal["warning"] = "warning"
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)


class ResultEvent(EventEnvelope):
    """Report the one successful terminal result."""

    type: Literal["result"] = "result"
    scene_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=500)


class ErrorEvent(EventEnvelope):
    """Report the one failed terminal result."""

    type: Literal["error"] = "error"
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    recovery_action: str | None = Field(default=None, max_length=500)
    log_reference: str | None = Field(default=None, max_length=500)


class CancelledEvent(EventEnvelope):
    """Report cooperative or forced cancellation."""

    type: Literal["cancelled"] = "cancelled"
    message: str = Field(min_length=1, max_length=500)
    forced: bool = False


JobEvent = Annotated[
    ProgressEvent | WarningEvent | ResultEvent | ErrorEvent | CancelledEvent,
    Field(discriminator="type"),
]
JOB_EVENT_ADAPTER = TypeAdapter(JobEvent)


class CancelControl(WireModel):
    """Request cooperative cancellation from an active worker."""

    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["cancel"] = "cancel"
    job_id: str = Field(min_length=1, max_length=128)


def parse_event_json(value: str | bytes) -> JobEvent:
    """Validate one complete JSON event."""
    return JOB_EVENT_ADAPTER.validate_json(value)


def event_json_schema() -> dict[str, object]:
    """Return the canonical schema snapshot content."""
    return JOB_EVENT_ADAPTER.json_schema()


def is_terminal(event: JobEvent) -> bool:
    """Return whether an event completes a job."""
    return isinstance(event, ResultEvent | ErrorEvent | CancelledEvent)
