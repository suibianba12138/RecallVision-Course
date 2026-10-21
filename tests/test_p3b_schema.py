"""P3B-S1 event-reference schema and status contract tests."""

from __future__ import annotations

import dataclasses

import pytest

from recallvision.query.schema import AnswerStatus, Query, QueryType


def _query_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "question_id": "question-1",
        "video_id": "video-001",
        "question": "Where was the cup before?",
        "query_type": QueryType.WHERE_OBJECT_BEFORE,
        "target_entity_ids": [],
        "target_class": "cup",
    }
    fields.update(overrides)
    return fields


def test_legacy_query_record_loads_with_no_reference_event() -> None:
    query = Query(**_query_fields())

    assert query.reference_event_id is None


def test_explicit_reference_event_id_round_trips_through_dataclass_record() -> None:
    query = Query(**_query_fields(reference_event_id="move-object:test"))
    serialized = dataclasses.asdict(query)
    restored = Query(**serialized)

    assert serialized["reference_event_id"] == "move-object:test"
    assert restored == query


@pytest.mark.parametrize("query_type", [QueryType.WHERE_OBJECT, QueryType.WHEN_OBJECT_MOVED])
def test_existing_query_types_keep_optional_reference_default(query_type: QueryType) -> None:
    query = Query(**_query_fields(query_type=query_type, question="existing query"))

    assert query.reference_event_id is None


@pytest.mark.parametrize("invalid_reference", ["", "   ", 123, False, []])
def test_reference_event_id_rejects_invalid_basic_shape(invalid_reference: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Query(**_query_fields(reference_event_id=invalid_reference))


def test_p3b_status_values_are_stable_and_serializable() -> None:
    assert AnswerStatus.NO_REFERENCE_EVENT.value == "NO_REFERENCE_EVENT"
    assert (
        AnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE.value
        == "AMBIGUOUS_MULTIPLE_EVENT_REFERENCE"
    )
