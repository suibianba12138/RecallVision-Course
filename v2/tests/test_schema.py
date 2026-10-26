"""Contract tests for the lightweight P0 schemas."""

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import (
    BoundingBox,
    EvidenceRef,
    ObjectMemory,
    Observation,
    ObjectState,
)
from recallvision.query.schema import Answer, Query, QueryType


def test_memory_records_can_be_constructed() -> None:
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    observation = Observation(
        video_id="video-001",
        frame_index=12,
        timestamp=0.5,
        track_id="track-3",
        persistent_id="object-7",
        entity_type="object",
        class_name="cup",
        bbox=bbox,
        confidence=0.95,
        zone="table",
    )
    state = ObjectState(timestamp=0.5, zone="table", state="stationary")
    memory = ObjectMemory(
        object_id="object-7",
        class_name="cup",
        observations=[observation],
        trajectory=[bbox],
        zone_history=[(0.5, "table")],
        state_history=[state],
    )

    assert memory.observations[0].persistent_id == "object-7"
    assert memory.trajectory[0].x1 == 10.0


def test_event_record_uses_stable_event_type_and_evidence() -> None:
    evidence = EvidenceRef(
        video_id="video-001",
        timestamp=3.0,
        frame_indices=[90],
    )
    event = EventRecord(
        event_id="event-1",
        event_type=EventType.MOVE_OBJECT,
        start_time=2.5,
        end_time=3.5,
        person_ids=["person-1"],
        object_ids=["object-7"],
        zone_before="shelf",
        zone_after="table",
        evidence=[evidence],
    )

    assert event.event_type is EventType.MOVE_OBJECT
    assert event.evidence[0].frame_indices == [90]


def test_query_and_answer_can_share_evidence() -> None:
    evidence = EvidenceRef(
        video_id="video-001",
        start_time=8.0,
        end_time=9.0,
        segment_ref="video-001#t=8,9",
    )
    query = Query(
        question_id="question-1",
        video_id="video-001",
        question="Where is the cup?",
        query_type=QueryType.WHERE_OBJECT,
        target_entity_ids=["object-7"],
    )
    answer = Answer(
        question_id=query.question_id,
        answer="On the table.",
        evidence=[evidence],
    )

    assert query.query_type.value == "WHERE_OBJECT"
    assert answer.evidence[0].segment_ref == "video-001#t=8,9"


def test_event_and_query_enum_values_are_frozen() -> None:
    assert [event_type.value for event_type in EventType] == [
        "PERSON_ENTER",
        "PERSON_EXIT",
        "OBJECT_APPEAR",
        "OBJECT_DISAPPEAR",
        "ZONE_CHANGE",
        "MOVE_OBJECT",
        "PICK_UP",
        "PUT_DOWN",
    ]
    assert [query_type.value for query_type in QueryType] == [
        "WHERE_OBJECT",
        "WHERE_OBJECT_BEFORE",
        "WHEN_OBJECT_MOVED",
        "WHO_MOVED_OBJECT",
    ]
