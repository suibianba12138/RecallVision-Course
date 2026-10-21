"""Unit tests for the P2A tracklet audit and association contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.tracklets import build_tracklet_summaries


def make_observation(
    *,
    class_name: str = "bottle",
    track_id: str = "1",
    frame_index: int = 0,
    timestamp: float | None = None,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 60.0),
    zone: str = "left_region",
) -> Observation:
    return Observation(
        video_id="video-1",
        frame_index=frame_index,
        timestamp=frame_index / 10.0 if timestamp is None else timestamp,
        track_id=track_id,
        persistent_id=f"track:{track_id}",
        entity_type="object",
        class_name=class_name,
        bbox=BoundingBox(*bbox),
        confidence=0.8,
        zone=zone,
    )


def test_tracklet_summary_aggregates_sorted_observations() -> None:
    observations = [
        make_observation(track_id="2", frame_index=12, bbox=(20, 20, 40, 60)),
        make_observation(track_id="1", frame_index=20, bbox=(30, 20, 50, 60)),
        make_observation(track_id="2", frame_index=10, bbox=(10, 20, 30, 60)),
        make_observation(track_id="2", frame_index=11, bbox=(15, 20, 35, 60)),
    ]

    summaries = build_tracklet_summaries(observations, frame_width=100, frame_height=100)

    tracklet = next(summary for summary in summaries if summary.track_id == "2")
    assert tracklet.start_frame == 10
    assert tracklet.end_frame == 12
    assert tracklet.observation_count == 3
    assert tracklet.first_bbox == BoundingBox(10, 20, 30, 60)
    assert tracklet.last_bbox == BoundingBox(20, 20, 40, 60)
    assert tracklet.median_bbox == BoundingBox(15, 20, 35, 60)
    assert tracklet.first_zone == "left_region"
    assert len(tracklet.representative_frames) == 3
    assert tracklet.first_center_normalized == (0.2, 0.4)
    assert math.isclose(tracklet.median_area, 800.0)


def test_tracklet_key_includes_video_class_and_track_id() -> None:
    observations = [
        make_observation(class_name="bottle", track_id="1", frame_index=0),
        Observation(
            video_id="video-2",
            frame_index=0,
            timestamp=0.0,
            track_id="1",
            persistent_id=None,
            entity_type="object",
            class_name="bottle",
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.5,
            zone="center_region",
        ),
        make_observation(class_name="book", track_id="1", frame_index=0),
    ]

    summaries = build_tracklet_summaries(observations, frame_width=100, frame_height=100)

    assert len(summaries) == 3
    assert {summary.key for summary in summaries} == {
        ("video-1", "bottle", "1"),
        ("video-1", "book", "1"),
        ("video-2", "bottle", "1"),
    }


def test_association_features_apply_class_temporal_spatial_and_scale_gates() -> None:
    from recallvision.memory.association import (
        AssociationConfig,
        compute_association_features,
    )

    summaries = build_tracklet_summaries(
        [
            make_observation(track_id="1", frame_index=0, bbox=(10, 10, 30, 30)),
            make_observation(track_id="2", frame_index=10, bbox=(15, 12, 35, 32)),
        ],
        frame_width=100,
        frame_height=100,
    )
    source, target = summaries

    features = compute_association_features(
        source,
        target,
        appearance_similarity=0.8,
        config=AssociationConfig(),
    )

    assert features.class_consistency is True
    assert features.temporal_gap == pytest.approx(1.0)
    assert features.center_distance == pytest.approx(math.sqrt(0.05**2 + 0.02**2))
    assert features.area_ratio == pytest.approx(1.0)
    assert features.spatial_similarity == pytest.approx(math.exp(-features.center_distance / 0.25))
    assert features.scale_similarity == pytest.approx(1.0)
    assert features.hard_gate_pass is True
    assert features.score == pytest.approx(
        0.5 * 0.8 + 0.3 * features.spatial_similarity + 0.2
    )


def test_class_gate_and_temporal_overlap_reject_invalid_candidates() -> None:
    from recallvision.memory.association import AssociationConfig, compute_association_features

    source = build_tracklet_summaries(
        [
            make_observation(class_name="bottle", track_id="1", frame_index=10),
            make_observation(class_name="bottle", track_id="1", frame_index=11),
        ],
        frame_width=100,
        frame_height=100,
    )[0]
    different_class = build_tracklet_summaries(
        [make_observation(class_name="book", track_id="2", frame_index=20)],
        frame_width=100,
        frame_height=100,
    )[0]
    different = compute_association_features(
        source,
        different_class,
        appearance_similarity=1.0,
        config=AssociationConfig(),
    )
    assert different.class_consistency is False
    assert different.hard_gate_pass is False
    assert "class" in different.rejection_reason

    overlapping = build_tracklet_summaries(
        [
            make_observation(track_id="3", frame_index=11),
            make_observation(track_id="3", frame_index=12),
        ],
        frame_width=100,
        frame_height=100,
    )[0]
    overlap_features = compute_association_features(
        source,
        overlapping,
        appearance_similarity=1.0,
        config=AssociationConfig(max_temporal_overlap_frames=0),
    )
    assert overlap_features.temporal_overlap_frames == 1
    assert overlap_features.hard_gate_pass is False
    assert "overlap" in overlap_features.rejection_reason

    too_late = build_tracklet_summaries(
        [make_observation(track_id="4", frame_index=30)],
        frame_width=100,
        frame_height=100,
    )[0]
    late_features = compute_association_features(
        source,
        too_late,
        appearance_similarity=1.0,
        config=AssociationConfig(),
    )
    assert late_features.hard_gate_pass is False
    assert late_features.rejection_reason == "temporal_gap"


def test_appearance_histogram_is_normalized_and_similar_images_match() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from recallvision.memory.appearance import (
        AppearanceExtractionError,
        extract_hsv_histogram,
        histogram_intersection_similarity,
    )

    red = np.zeros((40, 40, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)
    near_red = red.copy()
    near_red[0:4, 0:4] = (0, 10, 245)

    descriptor = extract_hsv_histogram(red, (0, 0, 40, 40))
    near_descriptor = extract_hsv_histogram(near_red, (0, 0, 40, 40))

    assert len(descriptor) == 32 * 32
    assert sum(descriptor) == pytest.approx(1.0)
    assert histogram_intersection_similarity(descriptor, near_descriptor) > 0.9
    with pytest.raises(AppearanceExtractionError):
        extract_hsv_histogram(red, (40, 40, 40, 40))


def test_association_selection_is_deterministic_and_respects_one_to_one() -> None:
    from recallvision.memory.association import (
        AssociationCandidate,
        AssociationConfig,
        select_association_edges,
    )

    def candidate(source: str, target: str, score: float) -> AssociationCandidate:
        return AssociationCandidate(
            source_key=("video-1", "bottle", source),
            target_key=("video-1", "bottle", target),
            source_track_id=source,
            target_track_id=target,
            class_name="bottle",
            temporal_gap=0.5,
            center_distance=0.1,
            area_ratio=1.0,
            appearance_similarity=0.8,
            spatial_similarity=0.9,
            scale_similarity=1.0,
            class_consistency=True,
            temporal_overlap_frames=0,
            score=score,
            hard_gate_pass=True,
        )

    candidates = [candidate("2", "4", 0.8), candidate("1", "4", 0.8), candidate("1", "3", 0.7)]
    first = select_association_edges(candidates, config=AssociationConfig(association_threshold=0.65))
    second = select_association_edges(candidates, config=AssociationConfig(association_threshold=0.65))

    assert [(edge.source_track_id, edge.target_track_id) for edge in first.accepted_edges] == [("1", "4")]
    assert first.accepted_edges == second.accepted_edges
    assert any(
        edge.source_track_id == "2"
        and edge.target_track_id == "4"
        and edge.rejection_reason == "one_predecessor"
        for edge in first.all_edges
    )


def test_chain_and_unmatched_tracklet_get_deterministic_persistent_ids() -> None:
    from recallvision.memory.association import AssociationCandidate
    from recallvision.memory.persistent_objects import build_persistent_objects

    observations = [
        make_observation(track_id="1", frame_index=0),
        make_observation(track_id="2", frame_index=10),
        make_observation(track_id="3", frame_index=20),
    ]
    summaries = build_tracklet_summaries(observations, frame_width=100, frame_height=100)

    def edge(source: str, target: str) -> AssociationCandidate:
        return AssociationCandidate(
            source_key=("video-1", "bottle", source),
            target_key=("video-1", "bottle", target),
            source_track_id=source,
            target_track_id=target,
            class_name="bottle",
            temporal_gap=0.1,
            center_distance=0.1,
            area_ratio=1.0,
            appearance_similarity=1.0,
            spatial_similarity=1.0,
            scale_similarity=1.0,
            class_consistency=True,
            temporal_overlap_frames=0,
            score=1.0,
            hard_gate_pass=True,
            accepted=True,
        )

    result = build_persistent_objects(summaries, observations, [edge("1", "2")])
    assert len(result.objects) == 2
    assert result.mapping[("video-1", "bottle", "1")] == "obj:bottle:0001"
    assert result.mapping[("video-1", "bottle", "2")] == "obj:bottle:0001"
    assert result.mapping[("video-1", "bottle", "3")] == "obj:bottle:0002"
    assert result.objects[0].track_ids == ("1", "2")
    assert result.objects[0].observation_count == 2


def test_persistent_object_builder_rejects_association_cycle() -> None:
    from recallvision.memory.association import AssociationCandidate
    from recallvision.memory.persistent_objects import build_persistent_objects

    observations = [
        make_observation(track_id="1", frame_index=0),
        make_observation(track_id="2", frame_index=10),
    ]
    summaries = build_tracklet_summaries(observations, frame_width=100, frame_height=100)

    def edge(source: str, target: str) -> AssociationCandidate:
        return AssociationCandidate(
            source_key=("video-1", "bottle", source),
            target_key=("video-1", "bottle", target),
            source_track_id=source,
            target_track_id=target,
            class_name="bottle",
            temporal_gap=0.1,
            center_distance=0.1,
            area_ratio=1.0,
            appearance_similarity=1.0,
            spatial_similarity=1.0,
            scale_similarity=1.0,
            class_consistency=True,
            temporal_overlap_frames=0,
            score=1.0,
            hard_gate_pass=True,
            accepted=True,
        )

    with pytest.raises(ValueError, match="cycle|forward"):
        build_persistent_objects(
            summaries,
            observations,
            [edge("1", "2"), edge("2", "1")],
        )


def test_enriched_memory_changes_only_persistent_id_and_query_still_works() -> None:
    from recallvision.memory.persistent_objects import enrich_observations
    from recallvision.query.object_location import answer_where_object
    from recallvision.query.schema import Query, QueryType

    observations = [make_observation(track_id="1", frame_index=5)]
    enriched = enrich_observations(observations, {("video-1", "bottle", "1"): "obj:bottle:0001"})

    assert enriched[0].persistent_id == "obj:bottle:0001"
    assert enriched[0].__class__ is observations[0].__class__
    assert enriched[0].__dict__ | {"persistent_id": observations[0].persistent_id} == observations[0].__dict__
    answer = answer_where_object(
        Query(
            question_id="q",
            video_id="video-1",
            question="Where is the bottle?",
            query_type=QueryType.WHERE_OBJECT,
            target_class="bottle",
        ),
        enriched,
    )
    assert answer.target_entity_id == "obj:bottle:0001"


def test_repeat_association_mapping_is_byte_identical() -> None:
    from recallvision.memory.association import associate_tracklets

    observations = [
        make_observation(track_id="1", frame_index=0),
        make_observation(track_id="2", frame_index=20),
    ]
    summaries = build_tracklet_summaries(observations, frame_width=100, frame_height=100)
    descriptors = {summary.key: (1.0,) for summary in summaries}
    first = associate_tracklets(summaries, descriptors)
    second = associate_tracklets(summaries, descriptors)

    assert first.to_records() == second.to_records()
