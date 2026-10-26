"""Tests for deterministic WHERE_OBJECT retrieval."""

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.query.object_location import answer_where_object
from recallvision.query.schema import AnswerStatus, Query, QueryType


def make_observation(
    *,
    class_name: str = "bottle",
    persistent_id: str | None = "bottle-1",
    timestamp: float = 1.0,
    frame_index: int = 30,
    zone: str | None = "left_region",
    confidence: float | None = 0.95,
    entity_type: str = "object",
) -> Observation:
    return Observation(
        video_id="fixture_video_001",
        frame_index=frame_index,
        timestamp=timestamp,
        track_id=f"track-{persistent_id}",
        persistent_id=persistent_id,
        entity_type=entity_type,
        class_name=class_name,
        bbox=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
        confidence=confidence,
        zone=zone,
    )


def make_query(
    *,
    query_type: QueryType = QueryType.WHERE_OBJECT,
    target_class: str | None = "bottle",
) -> Query:
    return Query(
        question_id="question-1",
        video_id="fixture_video_001",
        question="Where is the bottle?",
        query_type=query_type,
        target_class=target_class,
    )


def test_where_object_selects_latest_observation_and_evidence() -> None:
    answer = answer_where_object(
        make_query(),
        [
            make_observation(timestamp=1.0, frame_index=30, zone="left_region"),
            make_observation(timestamp=4.0, frame_index=120, zone="center_region"),
            make_observation(timestamp=8.0, frame_index=240, zone="right_region"),
            make_observation(
                class_name="person",
                persistent_id="person-1",
                entity_type="person",
                timestamp=9.0,
                frame_index=270,
                zone="center_region",
            ),
        ],
    )

    assert answer.status is AnswerStatus.OK
    assert answer.answer == "The bottle is in the right region."
    assert answer.target_entity_id == "bottle-1"
    assert answer.target_class == "bottle"
    assert answer.zone == "right_region"
    assert answer.timestamp == 8.0
    assert answer.frame_index == 240
    assert answer.evidence[0].video_id == "fixture_video_001"
    assert answer.evidence[0].timestamp == 8.0
    assert answer.evidence[0].frame_indices == [240]


def test_missing_target_returns_explicit_status() -> None:
    answer = answer_where_object(
        make_query(target_class="cup"),
        [make_observation()],
    )

    assert answer.status is AnswerStatus.TARGET_NOT_FOUND
    assert answer.answer
    assert answer.evidence == []


def test_raw_observation_path_uses_global_target_aliases() -> None:
    answer = answer_where_object(
        make_query(target_class="television"),
        [make_observation(class_name="tv", persistent_id=None)],
    )

    assert answer.status is AnswerStatus.OK
    assert answer.target_resolution_status == "LEGACY_RAW_OBSERVATION"
    assert answer.zone == "left_region"


def test_unsupported_query_returns_explicit_status() -> None:
    answer = answer_where_object(
        make_query(query_type=QueryType.WHEN_OBJECT_MOVED),
        [make_observation()],
    )

    assert answer.status is AnswerStatus.UNSUPPORTED_QUERY
    assert "WHEN_OBJECT_MOVED" in answer.answer


def test_same_timestamp_prefers_larger_frame_index() -> None:
    answer = answer_where_object(
        make_query(),
        [
            make_observation(timestamp=8.0, frame_index=240, zone="center_region"),
            make_observation(timestamp=8.0, frame_index=241, zone="right_region"),
        ],
    )

    assert answer.frame_index == 241
    assert answer.zone == "right_region"


def test_confidence_threshold_keeps_earlier_reliable_observation() -> None:
    answer = answer_where_object(
        make_query(),
        [
            make_observation(timestamp=8.0, frame_index=240, zone="center_region", confidence=0.9),
            make_observation(timestamp=9.0, frame_index=270, zone="right_region", confidence=0.2),
        ],
        min_confidence=0.5,
    )

    assert answer.status is AnswerStatus.OK
    assert answer.timestamp == 8.0
    assert answer.zone == "center_region"


def test_observation_without_zone_is_not_a_reliable_location() -> None:
    answer = answer_where_object(
        make_query(),
        [make_observation(zone=None)],
    )

    assert answer.status is AnswerStatus.TARGET_NOT_FOUND
