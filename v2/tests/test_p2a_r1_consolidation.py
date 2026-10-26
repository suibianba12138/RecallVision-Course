"""Tests for duplicate-aware P2A-R1 entity consolidation."""

from __future__ import annotations

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.tracklets import build_tracklet_summaries


def obs(
    track: str,
    frame: int,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
    timestamp: float | None = None,
) -> Observation:
    return Observation(
        video_id="video-1",
        frame_index=frame,
        timestamp=frame / 10.0 if timestamp is None else timestamp,
        track_id=track,
        persistent_id=None,
        entity_type="object",
        class_name="bottle",
        bbox=BoundingBox(*bbox),
        confidence=confidence,
        zone="center_region",
    )


def summaries_for(observations: list[Observation]):
    return build_tracklet_summaries(observations, frame_width=100, frame_height=100)


def descriptors_for(summaries):
    return {summary.key: (1.0, 0.0, 0.0, 0.0) for summary in summaries}


def test_overlapping_same_object_tracklets_are_duplicate_candidates() -> None:
    from recallvision.memory.consolidation import EntityConsolidationConfig
    from recallvision.memory.relations import find_duplicate_candidates

    observations = [
        obs("1", 0, (10, 10, 30, 50)),
        obs("1", 1, (10, 10, 30, 50)),
        obs("2", 0, (11, 11, 31, 51)),
        obs("2", 1, (11, 11, 31, 51)),
    ]
    summaries = summaries_for(observations)
    candidates = find_duplicate_candidates(
        summaries,
        observations,
        descriptors_for(summaries),
        frame_width=100,
        frame_height=100,
        config=EntityConsolidationConfig(),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.shared_frame_count == 2
    assert candidate.overlap_ratio == pytest.approx(1.0)
    assert candidate.median_iou > 0.8
    assert candidate.accepted is True
    assert candidate.relation_type.value == "DUPLICATE"


def test_partial_box_uses_ios_when_iou_is_low() -> None:
    from recallvision.memory.consolidation import EntityConsolidationConfig
    from recallvision.memory.relations import find_duplicate_candidates

    observations = [
        obs("1", 0, (10, 10, 50, 50)),
        obs("1", 1, (10, 10, 50, 50)),
        obs("2", 0, (35, 20, 55, 40)),
        obs("2", 1, (35, 20, 55, 40)),
    ]
    summaries = summaries_for(observations)
    candidates = find_duplicate_candidates(
        summaries,
        observations,
        descriptors_for(summaries),
        frame_width=100,
        frame_height=100,
        config=EntityConsolidationConfig(),
    )

    assert candidates[0].median_iou < 0.2
    assert candidates[0].median_ios == pytest.approx(0.75)
    assert candidates[0].accepted is True


def test_distinct_same_class_tracklets_are_conflict_not_duplicate() -> None:
    from recallvision.memory.consolidation import (
        EntityConsolidationConfig,
        build_entity_fragments,
        detect_spatial_conflicts,
    )
    from recallvision.memory.relations import (
        build_duplicate_groups,
        find_duplicate_candidates,
    )

    observations = [
        obs("1", 0, (5, 5, 20, 20)),
        obs("1", 1, (5, 5, 20, 20)),
        obs("2", 0, (75, 75, 95, 95)),
        obs("2", 1, (75, 75, 95, 95)),
    ]
    summaries = summaries_for(observations)
    config = EntityConsolidationConfig()
    candidates = find_duplicate_candidates(
        summaries,
        observations,
        descriptors_for(summaries),
        frame_width=100,
        frame_height=100,
        config=config,
    )
    assert candidates[0].accepted is False
    groups = build_duplicate_groups(summaries, candidates)
    fragments = build_entity_fragments(
        summaries,
        groups,
        observations,
        {("video-1", "bottle", frame): (1.0, 0.0) for frame in (0, 1)},
        frame_width=100,
        frame_height=100,
        window_size=3,
    )
    conflicts = detect_spatial_conflicts(
        fragments,
        config=config,
        frame_width=100,
        frame_height=100,
    )
    assert len(conflicts) == 1
    assert conflicts[0].relation_type.value == "CONFLICT"


def test_duplicate_union_find_and_ids_are_deterministic() -> None:
    from recallvision.memory.consolidation import EntityConsolidationConfig
    from recallvision.memory.relations import build_duplicate_groups, find_duplicate_candidates

    observations = [
        obs("1", 0, (10, 10, 30, 30)), obs("1", 1, (10, 10, 30, 30)),
        obs("2", 0, (11, 11, 31, 31)), obs("2", 1, (11, 11, 31, 31)),
        obs("3", 0, (12, 12, 32, 32)), obs("3", 1, (12, 12, 32, 32)),
    ]
    summaries = summaries_for(observations)
    candidates = find_duplicate_candidates(
        summaries, observations, descriptors_for(summaries), frame_width=100, frame_height=100,
        config=EntityConsolidationConfig(),
    )
    first = build_duplicate_groups(summaries, candidates)
    second = build_duplicate_groups(list(reversed(summaries)), list(reversed(candidates)))
    assert [group.to_record() for group in first] == [group.to_record() for group in second]
    assert first[0].track_ids == ("1", "2", "3")


def test_same_frame_canonical_selection_confidence_area_then_track_id() -> None:
    from recallvision.memory.relations import DuplicateGroup, canonicalize_observations

    group = DuplicateGroup(
        group_id="dup:bottle:0001",
        video_id="video-1",
        class_name="bottle",
        tracklet_keys=(
            ("video-1", "bottle", "1"),
            ("video-1", "bottle", "2"),
            ("video-1", "bottle", "3"),
        ),
        track_ids=("1", "2", "3"),
        start_frame=0,
        end_frame=0,
        accepted_duplicate_edges=("1<->2", "2<->3"),
    )
    observations = [
        obs("1", 0, (10, 10, 20, 20), confidence=0.9),
        obs("2", 0, (10, 10, 30, 30), confidence=0.9),
        obs("3", 0, (10, 10, 40, 40), confidence=0.9),
    ]
    canonical = canonicalize_observations(observations, [group])
    assert len(canonical) == 1
    assert canonical[0].observation.track_id == "3"
    assert canonical[0].source_track_ids == ("1", "2", "3")


def test_boundary_window_uses_median_of_first_and_last_three() -> None:
    from recallvision.memory.consolidation import build_boundary_summary
    from recallvision.memory.relations import canonicalize_observations

    observations = [obs("1", i, (10 + i, 10, 20 + i, 20)) for i in range(5)]
    canonical = canonicalize_observations(observations, [])
    appearances = {("video-1", "bottle", i): (float(i), 0.0) for i in range(5)}
    summary = build_boundary_summary(
        "frag:0001", canonical, appearances, frame_width=100, frame_height=100, window_size=3
    )
    assert summary.start_frame == 0
    assert summary.end_frame == 4
    assert summary.start_center == pytest.approx((0.16, 0.15))
    assert summary.end_center == pytest.approx((0.18, 0.15))
    assert summary.start_area == pytest.approx(100.0)
    assert summary.end_appearance == (3.0, 0.0)


def test_continuation_after_duplicate_collapse_and_short_gap() -> None:
    from recallvision.memory.consolidation import (
        EntityConsolidationConfig,
        build_entity_fragments,
        find_continuation_candidates,
        select_continuation_edges,
    )
    from recallvision.memory.relations import build_duplicate_groups, find_duplicate_candidates, canonicalize_observations

    observations = [
        obs("1", 0, (10, 10, 30, 40)), obs("1", 1, (10, 10, 30, 40)),
        obs("2", 0, (11, 10, 31, 40)), obs("2", 1, (11, 10, 31, 40)),
        obs("3", 3, (12, 10, 32, 40)), obs("3", 4, (12, 10, 32, 40)),
    ]
    config = EntityConsolidationConfig()
    summaries = summaries_for(observations)
    duplicates = find_duplicate_candidates(summaries, observations, descriptors_for(summaries), frame_width=100, frame_height=100, config=config)
    groups = build_duplicate_groups(summaries, duplicates)
    canonical = canonicalize_observations(observations, groups)
    appearances = {(o.observation.video_id, o.observation.class_name, o.observation.frame_index): (1.0, 0.0) for o in canonical}
    fragments = build_entity_fragments(summaries, groups, observations, appearances, frame_width=100, frame_height=100, window_size=3)
    candidates = find_continuation_candidates(fragments, config=config)
    selection = select_continuation_edges(candidates, conflicts=[] , config=config)
    assert len(selection.accepted_edges) == 1
    assert selection.accepted_edges[0].source_fragment_id != selection.accepted_edges[0].target_fragment_id


def test_spatial_conflict_blocks_transitive_merge() -> None:
    from recallvision.memory.consolidation import (
        BoundarySummary,
        EntityConsolidationConfig,
        EntityConflict,
        EntityFragment,
        ContinuationCandidate,
        select_continuation_edges,
    )

    def fragment(fragment_id: str, start: int) -> EntityFragment:
        boundary = BoundarySummary(
            fragment_id=fragment_id, video_id="video-1", class_name="bottle", track_ids=(fragment_id,),
            start_frame=start, end_frame=start, start_time=start / 10, end_time=start / 10,
            observation_count=1, canonical_observation_count=1,
            start_center=(0.1, 0.1), end_center=(0.1, 0.1), start_area=100, end_area=100,
            start_appearance=(1.0,), end_appearance=(1.0,),
        )
        return EntityFragment(fragment_id=fragment_id, group_id=fragment_id, boundary=boundary)

    def edge(source: str, target: str) -> ContinuationCandidate:
        return ContinuationCandidate(
            source_fragment_id=source, target_fragment_id=target, video_id="video-1", class_name="bottle",
            temporal_gap=0.1, center_distance=0.01, area_ratio=1.0, appearance_similarity=1.0,
            spatial_similarity=1.0, scale_similarity=1.0, score=1.0, hard_gate_pass=True,
        )

    fragments = [fragment("A", 0), fragment("B", 1), fragment("C", 2)]
    conflict = EntityConflict(
        fragment_a="A", fragment_b="C", video_id="video-1", class_name="bottle",
        frame_indices=(1,), center_distance=0.5, median_iou=0.0, reason="SIMULTANEOUS_SPATIAL_CONFLICT",
    )
    result = select_continuation_edges([edge("A", "B"), edge("B", "C")], conflicts=[conflict], config=EntityConsolidationConfig())
    assert [(e.source_fragment_id, e.target_fragment_id) for e in result.accepted_edges] == [("A", "B")]
    assert any(e.rejection_reason == "simultaneous_spatial_conflict" for e in result.all_edges)


def test_two_independent_bottles_remain_two_entities_and_duplicate_sequence_is_one() -> None:
    from recallvision.memory.consolidation import (
        EntityConsolidationConfig,
        build_persistent_entities,
        build_entity_fragments,
        find_continuation_candidates,
        select_continuation_edges,
    )
    from recallvision.memory.relations import build_duplicate_groups, canonicalize_observations, find_duplicate_candidates

    observations = [
        obs("1", 0, (5, 5, 20, 20)), obs("1", 1, (5, 5, 20, 20)),
        obs("2", 0, (7, 7, 22, 22)), obs("2", 1, (7, 7, 22, 22)),
        obs("3", 3, (8, 8, 23, 23)), obs("3", 4, (8, 8, 23, 23)),
        obs("4", 0, (75, 75, 90, 90)), obs("4", 1, (75, 75, 90, 90)),
    ]
    config = EntityConsolidationConfig()
    summaries = summaries_for(observations)
    descriptors = descriptors_for(summaries)
    duplicates = find_duplicate_candidates(summaries, observations, descriptors, frame_width=100, frame_height=100, config=config)
    groups = build_duplicate_groups(summaries, duplicates)
    canonical = canonicalize_observations(observations, groups)
    appearances = {(c.observation.video_id, c.observation.class_name, c.observation.frame_index): (1.0, 0.0) for c in canonical}
    fragments = build_entity_fragments(summaries, groups, observations, appearances, frame_width=100, frame_height=100, window_size=3)
    continuation = find_continuation_candidates(fragments, config=config)
    selected = select_continuation_edges(continuation, conflicts=[], config=config)
    build = build_persistent_entities(fragments, observations, canonical, selected.accepted_edges, duplicates)
    assert len(build.entities) == 2
    assert any(set(entity.track_ids) == {"1", "2", "3"} for entity in build.entities)


def test_consolidated_memory_removes_same_frame_duplicates_and_query_works() -> None:
    from recallvision.memory.persistent_objects import enrich_observations
    from recallvision.memory.relations import DuplicateGroup, canonicalize_observations
    from recallvision.query.object_location import answer_where_object
    from recallvision.query.schema import Query, QueryType

    group = DuplicateGroup(
        group_id="dup:bottle:0001", video_id="video-1", class_name="bottle",
        tracklet_keys=(("video-1", "bottle", "1"), ("video-1", "bottle", "2")),
        track_ids=("1", "2"), start_frame=0, end_frame=0, accepted_duplicate_edges=("1<->2",),
    )
    observations = [obs("1", 0, (10, 10, 30, 30)), obs("2", 0, (11, 11, 31, 31)), obs("2", 2, (12, 12, 32, 32))]
    canonical = canonicalize_observations(observations, [group])
    assert len(observations) == 3
    assert len(canonical) == 2
    enriched = enrich_observations(observations, {("video-1", "bottle", "1"): "obj:bottle:0001", ("video-1", "bottle", "2"): "obj:bottle:0001"})
    answer = answer_where_object(
        Query(question_id="q", video_id="video-1", question="Where is the bottle?", query_type=QueryType.WHERE_OBJECT, target_class="bottle"),
        [item.observation for item in canonical],
    )
    assert answer.status.value == "OK"
    assert answer.frame_index == 2
    assert enriched[0].frame_index == 0


def test_r1_mapping_is_deterministic_for_repeated_build() -> None:
    from recallvision.memory.consolidation import build_persistent_entities, EntityConsolidationConfig, build_entity_fragments, find_continuation_candidates, select_continuation_edges
    from recallvision.memory.relations import build_duplicate_groups, canonicalize_observations, find_duplicate_candidates

    observations = [obs("1", 0, (10, 10, 30, 30)), obs("2", 0, (11, 11, 31, 31)), obs("3", 2, (12, 12, 32, 32))]
    config = EntityConsolidationConfig()
    summaries = summaries_for(observations)
    duplicates = find_duplicate_candidates(summaries, observations, descriptors_for(summaries), frame_width=100, frame_height=100, config=config)
    groups = build_duplicate_groups(summaries, duplicates)
    canonical = canonicalize_observations(observations, groups)
    appearances = {(c.observation.video_id, c.observation.class_name, c.observation.frame_index): (1.0, 0.0) for c in canonical}
    fragments = build_entity_fragments(summaries, groups, observations, appearances, frame_width=100, frame_height=100, window_size=3)
    selected = select_continuation_edges(find_continuation_candidates(fragments, config=config), conflicts=[], config=config)
    first = build_persistent_entities(fragments, observations, canonical, selected.accepted_edges, duplicates)
    second = build_persistent_entities(list(reversed(fragments)), list(reversed(observations)), list(reversed(canonical)), list(reversed(selected.accepted_edges)), list(reversed(duplicates)))
    assert first.to_records() == second.to_records()
