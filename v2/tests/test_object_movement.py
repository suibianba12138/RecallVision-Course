"""Tests for deterministic WHEN_OBJECT_MOVED retrieval over event memory."""

from __future__ import annotations

import dataclasses

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline
from recallvision.query.object_movement import (
    MovementAnswerStatus,
    answer_when_object_moved,
)
from recallvision.query.schema import Query, QueryType


VIDEO_ID = "fixture_video_001"
PERSISTENT_ID = "obj:bottle:0001"


def _ref(frame: int, timestamp: float, *, video_id: str = VIDEO_ID) -> EvidenceRef:
    return EvidenceRef(
        video_id=video_id,
        timestamp=timestamp,
        frame_indices=[frame],
        segment_ref=f"{video_id}#frame={frame}",
    )


def _timeline(
    *,
    persistent_id: str = PERSISTENT_ID,
    class_name: str = "bottle",
    video_id: str = VIDEO_ID,
) -> ObjectStateTimeline:
    segment = ObjectStateRecord(
        persistent_id=persistent_id,
        class_name=class_name,
        video_id=video_id,
        start_frame=1,
        end_frame=200,
        start_time=0.0,
        end_time=8.0,
        zone="center_region",
        observation_count=3,
        first_evidence=_ref(1, 0.0, video_id=video_id),
        last_evidence=_ref(200, 8.0, video_id=video_id),
        source_track_ids=("track-1",),
        source_frame_indices=(1, 100, 200),
        last_confidence=0.9,
    )
    return ObjectStateTimeline(
        persistent_id=persistent_id,
        class_name=class_name,
        video_id=video_id,
        segments=(segment,),
    )


def _query(
    *,
    target_class: str | None = "bottle",
    target_entity_ids: list[str] | None = None,
    video_id: str = VIDEO_ID,
) -> Query:
    return Query(
        question_id="question-1",
        video_id=video_id,
        question="When was the bottle moved?",
        query_type=QueryType.WHEN_OBJECT_MOVED,
        target_entity_ids=target_entity_ids or [],
        target_class=target_class,
    )


def _move_event(
    *,
    event_id: str = "move-1",
    persistent_id: str = PERSISTENT_ID,
    start_time: float = 3.069733333333333,
    end_time: float = 3.3033,
    frames: tuple[int, ...] = (92, 99),
    video_id: str = VIDEO_ID,
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        event_type=EventType.MOVE_OBJECT,
        start_time=start_time,
        end_time=end_time,
        person_ids=[],
        object_ids=[persistent_id],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[
            _ref(frame, start_time if index == 0 else end_time, video_id=video_id)
            for index, frame in enumerate(frames)
        ],
        source_segment_ids=("fixture_video_001#segment=1", "fixture_video_001#segment=2"),
    )


def _zone_change_event(*, persistent_id: str = PERSISTENT_ID) -> EventRecord:
    event = _move_event(event_id="zone-only", persistent_id=persistent_id)
    return EventRecord(
        event_id=event.event_id,
        event_type=EventType.ZONE_CHANGE,
        start_time=event.start_time,
        end_time=event.end_time,
        object_ids=event.object_ids,
        evidence=event.evidence,
        source_segment_ids=event.source_segment_ids,
        zone_before=event.zone_before,
        zone_after=event.zone_after,
    )


def test_one_move_event_preserves_interval_and_provenance() -> None:
    result = answer_when_object_moved(_query(), [_timeline()], [_move_event()])

    assert result.status is MovementAnswerStatus.MOVEMENT_FOUND
    assert result.persistent_id == PERSISTENT_ID
    assert [event.event_id for event in result.events] == ["move-1"]
    assert result.time_intervals == ((3.069733333333333, 3.3033),)
    assert result.frame_intervals == ((92, 99),)
    assert "sometime between approximately 3.07 s and 3.30 s" in result.answer
    record = result.to_record()
    assert record["events"][0]["event_id"] == "move-1"
    assert record["events"][0]["evidence"][0]["frame_indices"] == [92]
    assert record["events"][0]["source_segment_ids"]


def test_zero_move_events_returns_no_supported_event_without_nonmovement_claim() -> None:
    result = answer_when_object_moved(_query(), [_timeline()], [])

    assert result.status is MovementAnswerStatus.NO_SUPPORTED_EVENT
    assert result.events == ()
    assert "did not move" not in result.answer
    assert "No supported movement event" in result.answer


def test_multiple_move_events_are_sorted_and_retained() -> None:
    later = _move_event(event_id="move-2", start_time=6.0, end_time=6.2, frames=(180, 190))
    earlier = _move_event(event_id="move-1", start_time=2.0, end_time=2.4, frames=(50, 60))

    result = answer_when_object_moved(_query(), [_timeline()], [later, earlier])

    assert [event.event_id for event in result.events] == ["move-1", "move-2"]
    assert result.time_intervals == ((2.0, 2.4), (6.0, 6.2))
    assert result.frame_intervals == ((50, 60), (180, 190))
    assert "2 supported movement intervals" in result.answer


def test_event_for_other_persistent_object_is_not_returned() -> None:
    result = answer_when_object_moved(
        _query(), [_timeline()], [_move_event(persistent_id="obj:bottle:0002")]
    )

    assert result.status is MovementAnswerStatus.NO_SUPPORTED_EVENT
    assert result.events == ()


def test_zone_change_alone_is_not_a_supported_movement_event() -> None:
    result = answer_when_object_moved(_query(), [_timeline()], [_zone_change_event()])

    assert result.status is MovementAnswerStatus.NO_SUPPORTED_EVENT
    assert "did not move" not in result.answer


def test_uncertain_event_preserves_bounds_and_does_not_use_midpoint() -> None:
    result = answer_when_object_moved(
        _query(), [_timeline()], [_move_event(start_time=2.0, end_time=4.0, frames=(20, 40))]
    )

    assert result.time_intervals == ((2.0, 4.0),)
    assert "between approximately 2.00 s and 4.00 s" in result.answer
    assert "3.00 s" not in result.answer


def test_exact_time_event_is_preserved_as_equal_interval() -> None:
    result = answer_when_object_moved(
        _query(), [_timeline()], [_move_event(start_time=4.0, end_time=4.0, frames=(120, 120))]
    )

    assert result.status is MovementAnswerStatus.MOVEMENT_FOUND
    assert result.time_intervals == ((4.0, 4.0),)
    assert result.frame_intervals == ((120, 120),)


def test_confirmation_window_evidence_does_not_extend_event_interval() -> None:
    result = answer_when_object_moved(
        _query(),
        [_timeline()],
        [_move_event(frames=(92, 99, 113))],
    )

    assert result.time_intervals == ((3.069733333333333, 3.3033),)
    assert result.frame_intervals == ((92, 99),)
    assert "3.77" not in result.answer
    assert result.to_record()["events"][0]["evidence"][-1]["frame_indices"] == [113]


def test_explicit_persistent_id_resolves_without_hard_coded_class_target() -> None:
    result = answer_when_object_moved(
        _query(target_class=None, target_entity_ids=[PERSISTENT_ID]),
        [_timeline()],
        [_move_event()],
    )

    assert result.status is MovementAnswerStatus.MOVEMENT_FOUND
    assert result.persistent_id == PERSISTENT_ID
    assert result.class_name == "bottle"


def test_ambiguous_class_target_fails_clearly() -> None:
    result = answer_when_object_moved(
        _query(),
        [_timeline(), _timeline(persistent_id="obj:bottle:0002")],
        [_move_event()],
    )

    assert result.status is MovementAnswerStatus.AMBIGUOUS_TARGET
    assert result.events == ()


def test_missing_class_target_is_distinct_from_zero_supported_events() -> None:
    result = answer_when_object_moved(
        _query(target_class="cup"), [_timeline()], [_move_event()]
    )

    assert result.status is MovementAnswerStatus.TARGET_NOT_FOUND
    assert result.status is not MovementAnswerStatus.NO_SUPPORTED_EVENT


def test_repeated_query_is_deterministic() -> None:
    events = [
        _move_event(event_id="move-2", start_time=6.0, end_time=6.2, frames=(180, 190)),
        _move_event(event_id="move-1", start_time=2.0, end_time=2.4, frames=(50, 60)),
    ]
    first = answer_when_object_moved(_query(), [_timeline()], events)
    second = answer_when_object_moved(_query(), [_timeline()], events)

    assert first == second
    assert first.to_record() == second.to_record()


def test_event_from_other_video_is_not_returned() -> None:
    result = answer_when_object_moved(
        _query(), [_timeline()], [_move_event(video_id="other-video")]
    )

    assert result.status is MovementAnswerStatus.NO_SUPPORTED_EVENT


def test_who_moved_object_remains_unsupported() -> None:
    result = answer_when_object_moved(
        dataclasses.replace(_query(), query_type=QueryType.WHO_MOVED_OBJECT),
        [_timeline()],
        [_move_event()],
    )

    assert result.status is MovementAnswerStatus.UNSUPPORTED_QUERY
    assert "WHO_MOVED_OBJECT" in result.answer
