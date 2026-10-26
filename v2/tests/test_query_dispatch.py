"""End-to-end dispatch coverage for the four V1 query types."""

from __future__ import annotations

from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline
from recallvision.query.dispatch import dispatch_query
from recallvision.query.object_state_location import StateAnswerStatus
from recallvision.query.parser import parse_query
from recallvision.query.schema import Query, QueryType


def test_where_object_dispatch_uses_precomputed_state_timeline() -> None:
    timeline = ObjectStateTimeline(
        persistent_id="obj:bottle:0001",
        class_name="bottle",
        video_id="fixture_video_001",
        segments=(
            ObjectStateRecord(
                persistent_id="obj:bottle:0001",
                class_name="bottle",
                video_id="fixture_video_001",
                start_frame=99,
                end_frame=152,
                start_time=3.3033,
                end_time=5.071733333333333,
                zone="center_region",
                observation_count=2,
                first_evidence=EvidenceRef(
                    video_id="fixture_video_001",
                    timestamp=3.3033,
                    frame_indices=[99],
                    segment_ref="fixture_video_001#frame=99",
                ),
                last_evidence=EvidenceRef(
                    video_id="fixture_video_001",
                    timestamp=5.071733333333333,
                    frame_indices=[152],
                    segment_ref="fixture_video_001#frame=152",
                ),
                source_track_ids=("track-1",),
                source_frame_indices=(99, 152),
                last_confidence=0.43,
            ),
        ),
    )
    query = Query(
        question_id="dispatch-where",
        video_id="fixture_video_001",
        question="Where is the bottle?",
        query_type=QueryType.WHERE_OBJECT,
        target_entity_ids=["obj:bottle:0001"],
        target_class="bottle",
    )

    result = dispatch_query(
        query,
        timelines=[timeline],
        video_end_timestamp=7.9079,
    )

    assert result.status is StateAnswerStatus.TARGET_LAST_SEEN
    assert result.zone == "center_region"
    assert result.frame_index == 152
    assert result.evidence_ref is not None
    assert result.evidence_ref.frame_indices == [152]


def test_where_object_dispatch_resolves_global_television_alias() -> None:
    timeline = ObjectStateTimeline(
        persistent_id="obj:tv:0001",
        class_name="tv",
        video_id="fixture_video_001",
        segments=(
            ObjectStateRecord(
                persistent_id="obj:tv:0001",
                class_name="tv",
                video_id="fixture_video_001",
                start_frame=99,
                end_frame=152,
                start_time=3.3033,
                end_time=5.071733333333333,
                zone="left_region",
                observation_count=2,
                first_evidence=EvidenceRef(
                    video_id="fixture_video_001",
                    timestamp=3.3033,
                    frame_indices=[99],
                    segment_ref="fixture_video_001#frame=99",
                ),
                last_evidence=EvidenceRef(
                    video_id="fixture_video_001",
                    timestamp=5.071733333333333,
                    frame_indices=[152],
                    segment_ref="fixture_video_001#frame=152",
                ),
                source_track_ids=("track-tv-1",),
                source_frame_indices=(99, 152),
                last_confidence=0.43,
            ),
        ),
    )
    query = parse_query(
        "Where is the television?",
        video_id="fixture_video_001",
        question_id="dispatch-tv",
    )

    result = dispatch_query(
        query,
        timelines=[timeline],
        video_end_timestamp=7.9079,
    )

    assert result.status is StateAnswerStatus.TARGET_LAST_SEEN
    assert result.persistent_id == "obj:tv:0001"
