"""Tests for strict versioned worker messages."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from better_backgrounds.protocol import (
    ProgressEvent,
    event_json_schema,
    is_terminal,
    parse_event_json,
)

FIXTURES = Path(__file__).parents[1] / "contracts" / "v1"
VALID_EVENT_COUNT = 5
TERMINAL_EVENT_COUNT = 3


def test_valid_contract_fixtures_are_accepted() -> None:
    """Keep representative messages valid at the Python boundary."""
    values = json.loads((FIXTURES / "valid-events.json").read_text(encoding="utf-8"))

    events = [parse_event_json(json.dumps(value)) for value in values]

    assert len(events) == VALID_EVENT_COUNT
    assert sum(is_terminal(event) for event in events) == TERMINAL_EVENT_COUNT


@pytest.mark.parametrize(
    "value",
    json.loads((FIXTURES / "invalid-events.json").read_text(encoding="utf-8")),
)
def test_invalid_contract_fixtures_are_rejected(value: object) -> None:
    """Reject drift, unknown fields, and invalid bounds."""
    with pytest.raises(ValidationError):
        parse_event_json(json.dumps(value))


def test_progress_is_non_terminal() -> None:
    """Distinguish progress from the exactly-one terminal event."""
    event = ProgressEvent(job_id="job-1", stage="validation", progress=0.5, message="Halfway")

    assert not is_terminal(event)


def test_checked_in_schema_matches_pydantic_models() -> None:
    """Make contract changes explicit in code review."""
    snapshot = json.loads((FIXTURES / "job-event.schema.json").read_text(encoding="utf-8"))

    assert snapshot == event_json_schema()
