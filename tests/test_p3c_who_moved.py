"""Focused P3C WHO_MOVED_OBJECT retrieval contract tests."""

from __future__ import annotations

import copy
import inspect

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.query.dispatch import dispatch_query
from recallvision.query.schema import Query, QueryType
from recallvision.query.who_moved import (
    AuthorizedAttributedEvent,
    WhoMovedAnswerStatus,
    answer_who_moved_object,
)


VIDEO_ID = "p1b_pixabay_bottle"
TARGET_ID = "obj:bottle:0001"
PERSON_1 = "obj:person:0001"
PERSON_2 = "obj:person:0002"


def _evidence(frame: int, timestamp: float, *, video_id: str = VIDEO_ID) -> EvidenceRef:
    return EvidenceRef(
        video_id=video_id,
        timestamp=timestamp,
        frame_indices=[frame],
        segment_ref=f"{video_id}#frame={frame}",
    )


def _event(
    event_id: str = "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2",
    *,
    object_id: str = TARGET_ID,
    event_type: EventType = EventType.MOVE_OBJECT,
    person_ids: list[str] | None = None,
    video_id: str = VIDEO_ID,
    start_time: float = 3.069733333333333,
    end_time: float = 3.3033,
) -> EventRecord:
    resolved_person_ids = [PERSON_1] if person_ids is None else list(person_ids)
    return EventRecord(
        event_id=event_id,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        person_ids=resolved_person_ids,
        object_ids=[object_id],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[
            _evidence(92, start_time, video_id=video_id),
            _evidence(99, end_time, video_id=video_id),
        ],
        source_segment_ids=(f"{video_id}#segment=1", f"{video_id}#segment=2"),
    )


def _authorized(
    event: EventRecord,
    *,
    status: str = "AUTHORIZED",
    person_ids: list[str] | None = None,
) -> AuthorizedAttributedEvent:
    return AuthorizedAttributedEvent(
        event=event if person_ids is None else _event(event.event_id, person_ids=person_ids),
        authorization={
            "authorization_status": status,
            "source_event_id": event.event_id,
            "source_event_hash": "source-event-fingerprint",
            "source_event_file_sha256": "source-artifact-sha256",
            "derived_event_hash": "derived-event-sha256",
        },
    )


def _query(
    *,
    target_class: str | None = "bottle",
    target_entity_ids: list[str] | None = None,
    reference_event_id: str | None = None,
    video_id: str = VIDEO_ID,
) -> Query:
    return Query(
        question_id="p3c-question",
        video_id=video_id,
        question="Who moved the bottle?",
        query_type=QueryType.WHO_MOVED_OBJECT,
        target_entity_ids=target_entity_ids or [],
        target_class=target_class,
        reference_event_id=reference_event_id,
    )


def _entities(*extra: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"persistent_id": TARGET_ID, "class_name": "bottle"},
        {"persistent_id": PERSON_1, "class_name": "person"},
        {"persistent_id": PERSON_2, "class_name": "person"},
        *extra,
    ]


def test_valid_unique_authorized_event_returns_success_and_provenance() -> None:
    event = _event()

    result = answer_who_moved_object(_query(), [_authorized(event)], _entities())

    assert result.status is WhoMovedAnswerStatus.SUCCESS
    assert result.person_ids == (PERSON_1,)
    assert result.resolved_event_id == event.event_id
    assert result.frame_interval == (92, 99)
    assert result.time_interval == (3.069733333333333, 3.3033)
    assert result.authorization_status == "AUTHORIZED"
    assert result.source_event_fingerprint == "source-event-fingerprint"
    assert result.source_artifact_sha256 == "source-artifact-sha256"


def test_query_dispatch_routes_who_moved_object_to_p3c_handler() -> None:
    result = dispatch_query(
        _query(),
        attributed_events=[_authorized(_event())],
        persistent_entities=_entities(),
    )

    assert result.status is WhoMovedAnswerStatus.SUCCESS


def test_explicit_reference_resolves_exact_authorized_event() -> None:
    first = _event("move-1")
    second = _event("move-2", person_ids=[PERSON_2])

    result = answer_who_moved_object(
        _query(reference_event_id="move-2"),
        [_authorized(first), _authorized(second)],
        _entities(),
    )

    assert result.status is WhoMovedAnswerStatus.SUCCESS
    assert result.resolved_event_id == "move-2"
    assert result.person_ids == (PERSON_2,)


def test_zero_authorized_events_fails_closed() -> None:
    result = answer_who_moved_object(_query(), [], _entities())

    assert result.status is WhoMovedAnswerStatus.NO_AUTHORIZED_ATTRIBUTION


def test_multiple_authorized_events_without_reference_is_ambiguous() -> None:
    result = answer_who_moved_object(
        _query(), [_authorized(_event("move-1")), _authorized(_event("move-2"))], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE


def test_explicit_reference_disambiguates_multiple_events() -> None:
    result = answer_who_moved_object(
        _query(reference_event_id="move-2"),
        [_authorized(_event("move-1")), _authorized(_event("move-2"))],
        _entities(),
    )

    assert result.status is WhoMovedAnswerStatus.SUCCESS
    assert result.resolved_event_id == "move-2"


def test_explicit_reference_with_wrong_target_fails_without_fallback() -> None:
    event = _event(object_id="obj:chair:0001")

    result = answer_who_moved_object(
        _query(reference_event_id=event.event_id), [_authorized(event)], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.INVALID_REFERENCE
    assert result.resolved_event_id is None


def test_explicit_reference_with_wrong_video_fails_without_fallback() -> None:
    event = _event(video_id="other-video")

    result = answer_who_moved_object(
        _query(reference_event_id=event.event_id), [_authorized(event)], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.INVALID_REFERENCE


def test_non_move_event_is_not_authoritative_for_who_query() -> None:
    result = answer_who_moved_object(
        _query(), [_authorized(_event(event_type=EventType.PICK_UP))], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.NO_AUTHORIZED_ATTRIBUTION


def test_unauthorized_event_with_nonempty_person_ids_is_rejected() -> None:
    result = answer_who_moved_object(
        _query(), [_authorized(_event(), status="BLOCKED_IDENTITY_GATE")], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.NO_AUTHORIZED_ATTRIBUTION


def test_authorized_event_with_empty_person_ids_is_rejected() -> None:
    result = answer_who_moved_object(
        _query(), [_authorized(_event(), person_ids=[])], _entities()
    )

    assert result.status is WhoMovedAnswerStatus.NO_AUTHORIZED_ATTRIBUTION


def test_unresolved_person_id_fails_closed() -> None:
    event = _event(person_ids=["obj:person:9999"])

    result = answer_who_moved_object(_query(), [_authorized(event)], _entities())

    assert result.status is WhoMovedAnswerStatus.UNRESOLVED_PERSON_ID
    assert result.person_ids == ()


def test_single_person_answer_is_template_deterministic() -> None:
    result = answer_who_moved_object(_query(), [_authorized(_event())], _entities())

    assert result.answer == "The bottle was moved by the tracked person obj:person:0001."


def test_multiple_person_answer_preserves_event_order_without_inference() -> None:
    event = _event(person_ids=[PERSON_2, PERSON_1])

    result = answer_who_moved_object(_query(), [_authorized(event)], _entities())

    assert result.status is WhoMovedAnswerStatus.SUCCESS
    assert result.person_ids == (PERSON_2, PERSON_1)
    assert result.answer == (
        "The bottle was moved by the tracked persons obj:person:0002, obj:person:0001."
    )


def test_query_does_not_mutate_attributed_source_event() -> None:
    event = _event()
    before = copy.deepcopy(event.to_record())

    answer_who_moved_object(_query(), [_authorized(event)], _entities())

    assert event.to_record() == before


def test_query_result_is_deterministic_across_repeated_runs() -> None:
    event = _event()
    first = answer_who_moved_object(_query(), [_authorized(event)], _entities()).to_record()
    second = answer_who_moved_object(_query(), [_authorized(event)], _entities()).to_record()

    assert first == second


def test_query_api_has_no_query_time_perception_or_reasoning_inputs() -> None:
    signature = inspect.signature(answer_who_moved_object)

    assert tuple(signature.parameters) == (
        "query",
        "attributed_events",
        "persistent_entities",
    )
    assert "brief_reason" not in inspect.getsource(answer_who_moved_object)
