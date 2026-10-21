"""P2B observed-state timeline and last-seen query contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.query.schema import Query, QueryType
from recallvision.query.object_state_location import (
    StateAnswerStatus,
    answer_where_object_state,
)
from recallvision.memory.state_timeline import (
    StateTimelineConfig,
    build_state_timeline,
    build_state_timelines,
    select_last_seen_state,
)


def make_observation(
    *,
    persistent_id: str = "obj:bottle:0001",
    class_name: str = "bottle",
    timestamp: float = 1.0,
    frame_index: int = 30,
    zone: str | None = "left_region",
    confidence: float | None = 0.95,
    track_id: str = "track-1",
    source_track_ids: tuple[str, ...] = (),
    video_id: str = "fixture_video_001",
) -> Observation:
    return Observation(
        video_id=video_id,
        frame_index=frame_index,
        timestamp=timestamp,
        track_id=track_id,
        persistent_id=persistent_id,
        entity_type="object",
        class_name=class_name,
        bbox=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
        confidence=confidence,
        zone=zone,
        source_track_ids=source_track_ids,
    )


def make_query(*, target_class: str | None = "bottle", target_ids: list[str] | None = None) -> Query:
    return Query(
        question_id="question-1",
        video_id="fixture_video_001",
        question="Where is the bottle?",
        query_type=QueryType.WHERE_OBJECT,
        target_entity_ids=target_ids or [],
        target_class=target_class,
    )


def test_unsorted_observations_are_ordered_by_time_and_frame() -> None:
    timeline = build_state_timeline(
        [
            make_observation(timestamp=2.0, frame_index=60),
            make_observation(timestamp=1.0, frame_index=30),
        ],
        StateTimelineConfig(max_observation_gap_seconds=2.0),
    )

    segment = timeline.segments[0]
    assert (segment.start_frame, segment.end_frame) == (30, 60)
    assert segment.observation_count == 2


def test_same_zone_observations_are_compressed_into_one_segment() -> None:
    timeline = build_state_timeline(
        [make_observation(timestamp=1.0), make_observation(timestamp=1.2, frame_index=36)],
        StateTimelineConfig(max_observation_gap_seconds=0.5),
    )

    assert len(timeline.segments) == 1
    assert timeline.segments[0].zone == "left_region"


def test_zone_change_splits_without_creating_an_event_record() -> None:
    timeline = build_state_timeline(
        [
            make_observation(timestamp=1.0, zone="left_region"),
            make_observation(timestamp=1.2, frame_index=36, zone="center_region"),
        ],
        StateTimelineConfig(max_observation_gap_seconds=0.5),
    )

    assert [segment.zone for segment in timeline.segments] == ["left_region", "center_region"]
    assert all("event_type" not in segment.to_record() for segment in timeline.segments)


def test_temporal_gap_splits_even_when_zone_is_unchanged() -> None:
    timeline = build_state_timeline(
        [
            make_observation(timestamp=1.0),
            make_observation(timestamp=1.51, frame_index=45),
        ],
        StateTimelineConfig(max_observation_gap_seconds=0.5),
    )

    assert len(timeline.segments) == 2


def test_multiple_persistent_entities_are_kept_isolated() -> None:
    timelines = build_state_timelines(
        [
            make_observation(persistent_id="obj:bottle:0001", timestamp=1.0),
            make_observation(persistent_id="obj:bottle:0002", timestamp=1.1, track_id="track-2"),
        ],
        StateTimelineConfig(),
    )

    assert [timeline.persistent_id for timeline in timelines] == [
        "obj:bottle:0001",
        "obj:bottle:0002",
    ]


def test_source_track_and_frame_provenance_is_preserved() -> None:
    timeline = build_state_timeline(
        [
            make_observation(
                timestamp=1.0,
                frame_index=30,
                track_id="16",
                source_track_ids=("16", "17"),
            ),
            make_observation(
                timestamp=1.2,
                frame_index=36,
                track_id="23",
                source_track_ids=("23",),
            ),
        ],
        StateTimelineConfig(),
    )

    segment = timeline.segments[0]
    assert segment.source_track_ids == ("16", "17", "23")
    assert segment.source_frame_indices == (30, 36)
    assert segment.first_evidence.frame_indices == [30]
    assert segment.last_evidence.frame_indices == [36]


def test_last_seen_state_selects_latest_observation_and_computes_staleness() -> None:
    timeline = build_state_timeline(
        [make_observation(timestamp=7.0, frame_index=210, zone="center_region", confidence=0.8)],
        StateTimelineConfig(),
    )

    state = select_last_seen_state(timeline, video_end_timestamp=10.0)
    assert state.zone == "center_region"
    assert state.timestamp == 7.0
    assert state.frame_index == 210
    assert state.confidence == 0.8
    assert state.staleness_seconds == 3.0


def test_negative_staleness_fails_explicitly() -> None:
    timeline = build_state_timeline([make_observation(timestamp=7.0)], StateTimelineConfig())

    with pytest.raises(ValueError, match="video_end_timestamp"):
        select_last_seen_state(timeline, video_end_timestamp=6.99)


def test_state_query_has_deterministic_last_seen_wording_and_metadata() -> None:
    timeline = build_state_timeline(
        [make_observation(timestamp=5.071733333333333, frame_index=152, zone="center_region")],
        StateTimelineConfig(),
    )

    result = answer_where_object_state(
        make_query(),
        [timeline],
        video_end_timestamp=7.9079,
    )
    assert result.status is StateAnswerStatus.TARGET_LAST_SEEN
    assert result.answer == "The bottle was last observed in the center region at 5.07 s."
    assert result.persistent_id == "obj:bottle:0001"
    assert result.video_end_timestamp == 7.9079
    assert result.staleness_seconds == pytest.approx(7.9079 - 5.071733333333333)
    assert result.evidence_ref is not None
    assert result.evidence_ref.frame_indices == [152]


def test_state_query_distinguishes_not_found_and_ambiguous_target() -> None:
    bottle_a = build_state_timeline([make_observation()], StateTimelineConfig())
    bottle_b = build_state_timeline(
        [make_observation(persistent_id="obj:bottle:0002", track_id="track-2")],
        StateTimelineConfig(),
    )

    missing = answer_where_object_state(
        make_query(target_class="cup"), [bottle_a], video_end_timestamp=2.0
    )
    ambiguous = answer_where_object_state(
        make_query(), [bottle_a, bottle_b], video_end_timestamp=2.0
    )
    assert missing.status is StateAnswerStatus.TARGET_NOT_FOUND
    assert ambiguous.status is StateAnswerStatus.AMBIGUOUS_TARGET


def test_explicit_persistent_id_resolves_multiple_same_class_entities() -> None:
    bottle_a = build_state_timeline([make_observation()], StateTimelineConfig())
    bottle_b = build_state_timeline(
        [make_observation(persistent_id="obj:bottle:0002", track_id="track-2", zone="right_region")],
        StateTimelineConfig(),
    )

    result = answer_where_object_state(
        make_query(target_ids=["obj:bottle:0002"]),
        [bottle_a, bottle_b],
        video_end_timestamp=2.0,
    )
    assert result.status is StateAnswerStatus.TARGET_LAST_SEEN
    assert result.persistent_id == "obj:bottle:0002"
    assert result.zone == "right_region"


def test_old_p1_where_object_path_remains_importable() -> None:
    from recallvision.query.object_location import answer_where_object

    assert callable(answer_where_object)


def test_p2b_query_core_does_not_import_optional_perception_dependencies() -> None:
    code = """
import sys
import recallvision.memory.state_timeline
import recallvision.query.object_state_location
forbidden = {"ultralytics", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("P2B_CORE_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "P2B_CORE_IMPORT_PASS"
