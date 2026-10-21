"""Tests for event-relative WHERE_OBJECT_BEFORE retrieval."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline
from recallvision.query.object_before import answer_where_object_before
from recallvision.query.schema import AnswerStatus, Query, QueryType


VIDEO_ID = "fixture_video_001"
OTHER_VIDEO_ID = "other_video_002"
PERSISTENT_ID = "obj:bottle:0001"
OTHER_PERSISTENT_ID = "obj:bottle:0002"


def _evidence(frame: int, timestamp: float, *, video_id: str = VIDEO_ID) -> EvidenceRef:
    return EvidenceRef(
        video_id=video_id,
        timestamp=timestamp,
        frame_indices=[frame],
        segment_ref=f"{video_id}#frame={frame}",
    )


def _segment(
    index: int,
    *,
    zone: str,
    persistent_id: str = PERSISTENT_ID,
    video_id: str = VIDEO_ID,
) -> ObjectStateRecord:
    start_frame = 10 + index * 10
    end_frame = start_frame + 2
    start_time = float(index)
    end_time = start_time + 0.2
    return ObjectStateRecord(
        persistent_id=persistent_id,
        class_name="bottle",
        video_id=video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
        zone=zone,
        observation_count=3,
        first_evidence=_evidence(start_frame, start_time, video_id=video_id),
        last_evidence=_evidence(end_frame, end_time, video_id=video_id),
        source_track_ids=("track-1",),
        source_frame_indices=(start_frame, start_frame + 1, end_frame),
        last_confidence=0.9,
    )


def _timeline(
    *,
    persistent_id: str = PERSISTENT_ID,
    video_id: str = VIDEO_ID,
    zones: tuple[str, ...] = ("left_region", "center_region"),
) -> ObjectStateTimeline:
    return ObjectStateTimeline(
        persistent_id=persistent_id,
        class_name="bottle",
        video_id=video_id,
        segments=tuple(
            _segment(index, zone=zone, persistent_id=persistent_id, video_id=video_id)
            for index, zone in enumerate(zones, start=1)
        ),
    )


def _segment_id(*, index: int, persistent_id: str = PERSISTENT_ID, video_id: str = VIDEO_ID) -> str:
    return f"{video_id}#{persistent_id}#segment={index}"


def _move_event(
    *,
    event_id: str = "move-1",
    persistent_id: str = PERSISTENT_ID,
    video_id: str = VIDEO_ID,
    from_segment: int = 1,
    to_segment: int = 2,
    event_type: EventType = EventType.MOVE_OBJECT,
    source_segment_ids: tuple[str, ...] | None = None,
) -> EventRecord:
    source_ids = (
        source_segment_ids
        if source_segment_ids is not None
        else (
            _segment_id(index=from_segment, persistent_id=persistent_id, video_id=video_id),
            _segment_id(index=to_segment, persistent_id=persistent_id, video_id=video_id),
        )
    )
    return EventRecord(
        event_id=event_id,
        event_type=event_type,
        start_time=1.2,
        end_time=2.0,
        object_ids=[persistent_id],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[_evidence(12, 1.2, video_id=video_id), _evidence(22, 2.0, video_id=video_id)],
        source_segment_ids=source_ids,
    )


def _query(
    *,
    target_class: str | None = "bottle",
    target_entity_ids: list[str] | None = None,
    video_id: str = VIDEO_ID,
    reference_event_id: str | None = None,
) -> Query:
    return Query(
        question_id="question-1",
        video_id=video_id,
        question="Where was the bottle before?",
        query_type=QueryType.WHERE_OBJECT_BEFORE,
        target_entity_ids=target_entity_ids or [],
        target_class=target_class,
        reference_event_id=reference_event_id,
    )


def test_explicit_reference_returns_from_segment_state_and_provenance() -> None:
    result = answer_where_object_before(
        _query(reference_event_id="move-1"),
        [_timeline()],
        [_move_event()],
    )

    assert result.status is AnswerStatus.OK
    assert result.target_entity_id == PERSISTENT_ID
    assert result.reference_event_id == "move-1"
    assert result.from_source_segment_id == _segment_id(index=1)
    assert result.zone == "left_region"
    assert result.timestamp == 1.2
    assert result.frame_index == 22
    assert result.evidence[0].frame_indices == [22]
    assert result.answer == (
        "Before that supported movement event, the bottle was last observed in the left region."
    )
    record = result.to_record()
    assert record["query_type"] == "WHERE_OBJECT_BEFORE"
    assert record["last_observed_frame"] == 22
    assert record["from_source_segment_id"] == _segment_id(index=1)


def test_unique_move_event_is_used_when_reference_is_omitted() -> None:
    result = answer_where_object_before(_query(), [_timeline()], [_move_event()])

    assert result.status is AnswerStatus.OK
    assert result.reference_event_id == "move-1"
    assert result.zone == "left_region"


def test_zero_move_events_returns_no_reference_event_without_nonmovement_claim() -> None:
    result = answer_where_object_before(_query(), [_timeline()], [])

    assert result.status is AnswerStatus.NO_REFERENCE_EVENT
    assert result.reference_event_id is None
    assert "did not move" not in result.answer
    assert "No supported movement event" in result.answer


def test_multiple_move_events_without_reference_are_ambiguous() -> None:
    result = answer_where_object_before(
        _query(),
        [_timeline(zones=("left_region", "center_region", "right_region"))],
        [
            _move_event(event_id="move-1", from_segment=1, to_segment=2),
            _move_event(event_id="move-2", from_segment=2, to_segment=3),
        ],
    )

    assert result.status is AnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE
    assert result.reference_event_id is None
    assert "ambiguous" in result.answer.casefold()


def test_explicit_reference_selects_one_event_when_multiple_match() -> None:
    result = answer_where_object_before(
        _query(reference_event_id="move-2"),
        [_timeline(zones=("left_region", "center_region", "right_region"))],
        [
            _move_event(event_id="move-1", from_segment=1, to_segment=2),
            _move_event(event_id="move-2", from_segment=2, to_segment=3),
        ],
    )

    assert result.status is AnswerStatus.OK
    assert result.reference_event_id == "move-2"
    assert result.from_source_segment_id == _segment_id(index=2)
    assert result.zone == "center_region"


def test_mixed_event_evidence_video_is_rejected() -> None:
    event = _move_event()
    malformed = replace(
        event,
        evidence=[event.evidence[0], _evidence(22, 2.0, video_id=OTHER_VIDEO_ID)],
    )

    result = answer_where_object_before(
        _query(reference_event_id="move-1"),
        [_timeline()],
        [malformed],
    )

    assert result.status is AnswerStatus.UNSUPPORTED_QUERY
    assert "INVALID_REFERENCE" in result.answer


def test_from_segment_evidence_must_match_its_timeline() -> None:
    timeline = _timeline()
    malformed_first = replace(
        timeline.segments[0],
        last_evidence=_evidence(22, 1.2, video_id=OTHER_VIDEO_ID),
    )
    malformed_timeline = replace(timeline, segments=(malformed_first, timeline.segments[1]))

    result = answer_where_object_before(
        _query(reference_event_id="move-1"),
        [malformed_timeline],
        [_move_event()],
    )

    assert result.status is AnswerStatus.UNSUPPORTED_QUERY
    assert "INVALID_PROVENANCE" in result.answer


def test_missing_from_segment_evidence_fails_closed() -> None:
    timeline = _timeline()
    malformed_first = replace(timeline.segments[0], last_evidence=None)
    malformed_timeline = replace(timeline, segments=(malformed_first, timeline.segments[1]))

    result = answer_where_object_before(
        _query(reference_event_id="move-1"),
        [malformed_timeline],
        [_move_event()],
    )

    assert result.status is AnswerStatus.UNSUPPORTED_QUERY
    assert "INVALID_PROVENANCE" in result.answer


@pytest.mark.parametrize(
    ("label", "events", "reference_event_id"),
    [
        ("missing event", [_move_event()], "does-not-exist"),
        (
            "wrong event type",
            [replace(_move_event(), event_type=EventType.ZONE_CHANGE)],
            "move-1",
        ),
        (
            "other object",
            [_move_event(persistent_id=OTHER_PERSISTENT_ID)],
            "move-1",
        ),
        (
            "other video",
            [_move_event(video_id=OTHER_VIDEO_ID)],
            "move-1",
        ),
        (
            "missing source provenance",
            [_move_event(source_segment_ids=())],
            "move-1",
        ),
        (
            "from segment missing",
            [_move_event(source_segment_ids=("fixture_video_001#obj:bottle:0001#segment=9", _segment_id(index=2)))],
            "move-1",
        ),
        (
            "from segment other object",
            [
                _move_event(
                    source_segment_ids=(
                        _segment_id(index=1, persistent_id=OTHER_PERSISTENT_ID),
                        _segment_id(index=2),
                    )
                )
            ],
            "move-1",
        ),
    ],
)
def test_explicit_invalid_reference_fails_without_implicit_fallback(
    label: str,
    events: list[EventRecord],
    reference_event_id: str,
) -> None:
    result = answer_where_object_before(
        _query(reference_event_id=reference_event_id),
        [_timeline()],
        events,
    )

    assert result.status is AnswerStatus.UNSUPPORTED_QUERY, label
    assert result.zone is None
    assert "INVALID_REFERENCE" in result.answer or "INVALID_PROVENANCE" in result.answer


def test_target_resolution_honors_explicit_persistent_id_and_rejects_ambiguity() -> None:
    result = answer_where_object_before(
        _query(target_class=None, target_entity_ids=[PERSISTENT_ID]),
        [_timeline(), _timeline(persistent_id=OTHER_PERSISTENT_ID)],
        [_move_event()],
    )
    assert result.status is AnswerStatus.OK
    assert result.target_entity_id == PERSISTENT_ID

    ambiguous = answer_where_object_before(
        _query(),
        [_timeline(), _timeline(persistent_id=OTHER_PERSISTENT_ID)],
        [_move_event()],
    )
    assert ambiguous.status is AnswerStatus.AMBIGUOUS_TARGET
    assert "AMBIGUOUS_TARGET" in ambiguous.answer


def test_repeated_query_is_deterministic() -> None:
    query = _query()
    timelines = [_timeline()]
    events = [_move_event()]

    first = answer_where_object_before(query, timelines, events)
    second = answer_where_object_before(query, timelines, events)

    assert first == second
    assert first.to_record() == second.to_record()


def test_handler_import_does_not_load_p2d_movement_evaluator() -> None:
    code = """
import sys
import recallvision.query.object_before
assert "recallvision.events.movement_evidence" not in sys.modules
print("P3B_QUERY_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "P3B_QUERY_IMPORT_PASS"
