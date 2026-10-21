"""P2D evidence-bounded, agent-neutral MOVE_OBJECT event contracts."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from recallvision.events.movement_evidence import (
    MovementEvidenceConfig,
    MovementEvidenceEvaluator,
    MovementObservation,
)
from recallvision.events.schema import EventType
from recallvision.events.zone_transitions import ZoneTransitionEventBuilder
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline


VIDEO_ID = "fixture_video_001"
PERSISTENT_ID = "obj:bottle:0001"


def _evidence(timestamp: float, frame_index: int, label: str) -> EvidenceRef:
    return EvidenceRef(
        video_id=VIDEO_ID,
        timestamp=timestamp,
        frame_indices=[frame_index],
        segment_ref=f"{VIDEO_ID}#frame={frame_index}#{label}",
    )


def _segment(
    *,
    index: int,
    zone: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    observation_count: int,
) -> ObjectStateRecord:
    return ObjectStateRecord(
        persistent_id=PERSISTENT_ID,
        class_name="bottle",
        video_id=VIDEO_ID,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
        zone=zone,
        observation_count=observation_count,
        first_evidence=_evidence(start_time, start_frame, f"segment={index}:first"),
        last_evidence=_evidence(end_time, end_frame, f"segment={index}:last"),
        source_track_ids=("track-before",) if index == 1 else ("track-after",),
        source_frame_indices=tuple(range(start_frame, end_frame + 1)),
        last_confidence=0.9,
    )


def _timeline(
    *,
    before: tuple[tuple[float, float], ...],
    after: tuple[tuple[float, float], ...],
    transition_gap: float = 0.2,
    before_zone: str = "left_region",
    after_zone: str = "center_region",
) -> tuple[ObjectStateTimeline, list[MovementObservation]]:
    before_start = 10
    before_frames = tuple(before_start + 10 * index for index in range(len(before)))
    before_times = tuple(1.0 + 0.1 * index for index in range(len(before)))
    before_end = before_frames[-1]
    before_end_time = before_times[-1]
    after_start = before_end + 10
    after_frames = tuple(after_start + 10 * index for index in range(len(after)))
    after_start_time = before_end_time + transition_gap
    after_times = tuple(after_start_time + 0.1 * index for index in range(len(after)))
    after_end = after_frames[-1]
    after_end_time = after_times[-1]
    timeline = ObjectStateTimeline(
        persistent_id=PERSISTENT_ID,
        class_name="bottle",
        video_id=VIDEO_ID,
        segments=(
            _segment(
                index=1,
                zone=before_zone,
                start_frame=before_frames[0],
                end_frame=before_end,
                start_time=before_times[0],
                end_time=before_end_time,
                observation_count=len(before),
            ),
            _segment(
                index=2,
                zone=after_zone,
                start_frame=after_frames[0],
                end_frame=after_end,
                start_time=after_times[0],
                end_time=after_end_time,
                observation_count=len(after),
            ),
        ),
    )
    observations: list[MovementObservation] = []
    for frame, timestamp, center in zip(before_frames, before_times, before):
        observations.append(
            MovementObservation(
                persistent_id=PERSISTENT_ID,
                frame_index=frame,
                timestamp=timestamp,
                zone=before_zone,
                center_normalized=center,
                source_track_ids=("track-before",),
                bbox=(100.0, 100.0, 140.0, 200.0),
                confidence=0.9,
            )
        )
    for frame, timestamp, center in zip(after_frames, after_times, after):
        observations.append(
            MovementObservation(
                persistent_id=PERSISTENT_ID,
                frame_index=frame,
                timestamp=timestamp,
                zone=after_zone,
                center_normalized=center,
                source_track_ids=("track-after",),
                bbox=(100.0, 100.0, 140.0, 200.0),
                confidence=0.9,
            )
        )
    return timeline, observations


def _trigger(timeline: ObjectStateTimeline):
    return ZoneTransitionEventBuilder().build(timeline)[0]


def _evaluate(
    *,
    before: tuple[tuple[float, float], ...],
    after: tuple[tuple[float, float], ...],
    transition_gap: float = 0.2,
    before_zone: str = "left_region",
    after_zone: str = "center_region",
    config: MovementEvidenceConfig | None = None,
):
    timeline, observations = _timeline(
        before=before,
        after=after,
        transition_gap=transition_gap,
        before_zone=before_zone,
        after_zone=after_zone,
    )
    result = MovementEvidenceEvaluator(config or MovementEvidenceConfig()).evaluate(
        timeline,
        observations,
        _trigger(timeline) if before_zone != after_zone else None,
    )
    return result, timeline, observations


def _extra_observations(
    *,
    zone: str,
    centers: tuple[tuple[float, float], ...],
    start_frame: int,
    start_time: float,
    track_id: str,
) -> list[MovementObservation]:
    return [
        MovementObservation(
            persistent_id=PERSISTENT_ID,
            frame_index=start_frame + index,
            timestamp=start_time + 0.1 * index,
            zone=zone,
            center_normalized=center,
            source_track_ids=(track_id,),
            bbox=(100.0, 100.0, 140.0, 200.0),
            confidence=0.9,
        )
        for index, center in enumerate(centers)
    ]


def _with_segment_extras(
    timeline: ObjectStateTimeline,
    observations: list[MovementObservation],
    *,
    segment_index: int,
    extras: list[MovementObservation],
) -> tuple[ObjectStateTimeline, list[MovementObservation]]:
    segment = timeline.segments[segment_index]
    segment_observations = [
        item
        for item in (*observations, *extras)
        if item.persistent_id == segment.persistent_id and item.zone == segment.zone
    ]
    start_observation = min(segment_observations, key=lambda item: (item.frame_index, item.timestamp))
    end_observation = max(segment_observations, key=lambda item: (item.frame_index, item.timestamp))
    updated_segment = dataclasses.replace(
        segment,
        start_frame=start_observation.frame_index,
        end_frame=end_observation.frame_index,
        start_time=start_observation.timestamp,
        end_time=end_observation.timestamp,
        observation_count=len(segment_observations),
        first_evidence=(
            _evidence(start_observation.timestamp, start_observation.frame_index, "extended:first")
            if start_observation.frame_index != segment.start_frame
            else segment.first_evidence
        ),
        last_evidence=(
            _evidence(end_observation.timestamp, end_observation.frame_index, "extended:last")
            if end_observation.frame_index != segment.end_frame
            else segment.last_evidence
        ),
        source_frame_indices=tuple(sorted({item.frame_index for item in segment_observations})),
    )
    segments = list(timeline.segments)
    segments[segment_index] = updated_segment
    return dataclasses.replace(timeline, segments=tuple(segments)), [*observations, *extras]


def test_clear_zone_change_emits_one_agent_neutral_move_object_event() -> None:
    result, _, _ = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )

    assert result.accepted is True
    assert result.reason == "ACCEPTED"
    assert result.event is not None
    assert result.event.event_type is EventType.MOVE_OBJECT
    assert result.event.object_ids == [PERSISTENT_ID]
    assert result.event.person_ids == []
    assert result.evidence["trigger_event_id"].startswith("zone-change:")
    assert result.evidence["gates"]["temporal"]["passed"] is True
    assert result.evidence["gates"]["identity"]["passed"] is True


def test_boundary_local_support_is_explicit_and_does_not_extend_event_end() -> None:
    result, timeline, _ = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )

    assert result.accepted is True
    assert result.event is not None
    assert result.event.end_time == timeline.segments[1].start_time
    assert result.evidence["evidence_window_policy"] == "BOUNDARY_LOCAL_TIME_WINDOW"
    assert result.evidence["support_horizon_seconds"] == 0.5
    assert result.evidence["left_support_frames"]
    assert result.evidence["right_support_frames"]
    assert result.evidence["left_support_time_range"][1] == timeline.segments[0].end_time
    assert result.evidence["right_support_time_range"][0] == timeline.segments[1].start_time
    assert result.evidence["left_cluster_center"] == result.evidence["before_cluster"]["center_normalized"]
    assert result.evidence["right_cluster_center"] == result.evidence["after_cluster"]["center_normalized"]


def test_far_future_motion_outside_support_is_invariant() -> None:
    baseline, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    extras = _extra_observations(
        zone="center_region",
        centers=((0.05, 0.05), (0.90, 0.90), (0.10, 0.90), (0.90, 0.10)),
        start_frame=1000,
        start_time=4.0,
        track_id="track-after",
    )
    extended_timeline, extended_observations = _with_segment_extras(
        timeline, observations, segment_index=1, extras=extras
    )
    extended = MovementEvidenceEvaluator().evaluate(
        extended_timeline, extended_observations, _trigger(extended_timeline)
    )

    assert (extended.accepted, extended.reason) == (baseline.accepted, baseline.reason)
    for key in ("local_variation", "before_cluster", "after_cluster", "gates"):
        assert extended.evidence[key] == baseline.evidence[key]
    assert extended.evidence["right_support_frames"] == baseline.evidence["right_support_frames"]
    assert all(frame < 1000 for frame in extended.evidence["right_support_frames"])


def test_far_past_motion_outside_support_is_invariant() -> None:
    baseline, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    extras = _extra_observations(
        zone="left_region",
        centers=((0.90, 0.90), (0.05, 0.05), (0.90, 0.10), (0.10, 0.90)),
        start_frame=1,
        start_time=0.0,
        track_id="track-before",
    )
    extended_timeline, extended_observations = _with_segment_extras(
        timeline, observations, segment_index=0, extras=extras
    )
    extended = MovementEvidenceEvaluator().evaluate(
        extended_timeline, extended_observations, _trigger(extended_timeline)
    )

    assert (extended.accepted, extended.reason) == (baseline.accepted, baseline.reason)
    for key in ("local_variation", "before_cluster", "after_cluster", "gates"):
        assert extended.evidence[key] == baseline.evidence[key]
    assert extended.evidence["left_support_frames"] == baseline.evidence["left_support_frames"]
    assert all(frame >= 10 for frame in extended.evidence["left_support_frames"])


def test_late_motion_cannot_create_move_object_when_boundary_is_not_material() -> None:
    result, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.101, 0.400), (0.102, 0.400), (0.103, 0.400)),
    )
    extras = _extra_observations(
        zone="center_region",
        centers=((0.90, 0.90), (0.05, 0.05), (0.90, 0.10)),
        start_frame=1000,
        start_time=4.0,
        track_id="track-after",
    )
    extended_timeline, extended_observations = _with_segment_extras(
        timeline, observations, segment_index=1, extras=extras
    )
    extended = MovementEvidenceEvaluator().evaluate(
        extended_timeline, extended_observations, _trigger(extended_timeline)
    )

    assert result.accepted is False
    assert extended.accepted is False
    assert extended.event is None
    assert extended.reason == "DISPLACEMENT_NOT_MATERIAL"
    assert all(frame < 1000 for frame in extended.evidence["right_support_frames"])


def test_local_move_survives_later_broad_wander() -> None:
    timeline, observations = _timeline(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    extras = _extra_observations(
        zone="center_region",
        centers=((0.90, 0.90), (0.05, 0.05), (0.90, 0.10), (0.10, 0.90)),
        start_frame=1000,
        start_time=4.0,
        track_id="track-after",
    )
    extended_timeline, extended_observations = _with_segment_extras(
        timeline, observations, segment_index=1, extras=extras
    )
    result = MovementEvidenceEvaluator().evaluate(
        extended_timeline, extended_observations, _trigger(extended_timeline)
    )

    assert result.accepted is True
    assert result.event is not None
    assert result.evidence["right_support_frames"] == [40, 50, 60]
    assert result.evidence["after_cluster"]["count"] == 3


def test_unrelated_same_class_or_other_entity_in_candidate_range_is_not_identity_conflict() -> None:
    result, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    unrelated = MovementObservation(
        persistent_id="obj:bottle:0002",
        frame_index=20,
        timestamp=1.1,
        zone="left_region",
        center_normalized=(0.8, 0.4),
        source_track_ids=("other-track",),
    )
    result = MovementEvidenceEvaluator().evaluate(
        timeline,
        [*observations, unrelated],
        _trigger(timeline),
    )

    assert result.accepted is True
    assert "obj:bottle:0002" in result.evidence["gates"]["identity"]["observed_persistent_ids"]


def test_zone_change_with_overlapping_clusters_is_rejected() -> None:
    result, _, _ = _evaluate(
        before=((0.332, 0.400), (0.328, 0.400), (0.324, 0.400)),
        after=((0.338, 0.400), (0.334, 0.400), (0.330, 0.400)),
    )

    assert result.accepted is False
    assert result.event is None
    assert result.reason == "CLUSTER_ENVELOPES_OVERLAP"
    assert result.evidence["gates"]["displacement"]["passed"] is True
    assert result.evidence["gates"]["cluster_separation"]["passed"] is False


def test_same_zone_large_displacement_requires_zone_change_trigger() -> None:
    result, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.500, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
        before_zone="left_region",
        after_zone="left_region",
    )

    assert ZoneTransitionEventBuilder().build(timeline) == []
    assert result.accepted is False
    assert result.reason == "ZONE_CHANGE_TRIGGER_REQUIRED"
    assert observations[-1].zone == "left_region"


def test_insufficient_local_baseline_fails_closed() -> None:
    result, _, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    unique = [dataclasses.replace(item, source_track_ids=(f"unique-{index}",)) for index, item in enumerate(observations)]
    timeline, _ = _timeline(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    result = MovementEvidenceEvaluator().evaluate(timeline, unique, _trigger(timeline))

    assert result.accepted is False
    assert result.reason == "INSUFFICIENT_LOCAL_VARIATION"
    assert result.evidence["gates"]["sufficient_evidence"]["passed"] is False
    assert result.evidence["local_variation"]["count"] == 0


def test_insufficient_segment_support_fails_closed() -> None:
    result, _, _ = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400)),
    )

    assert result.accepted is False
    assert result.reason == "INSUFFICIENT_SEGMENT_SUPPORT"
    assert result.evidence["gates"]["sufficient_evidence"]["passed"] is False


def test_insufficient_boundary_local_support_has_no_full_segment_fallback() -> None:
    timeline, observations = _timeline(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    observations[0] = dataclasses.replace(observations[0], timestamp=0.0)
    result = MovementEvidenceEvaluator().evaluate(
        timeline, observations, _trigger(timeline)
    )

    assert result.accepted is False
    assert result.reason == "INSUFFICIENT_SEGMENT_SUPPORT"
    assert result.evidence["segment_support"]["before_count"] == 3
    assert result.evidence["segment_support"]["left_support_count"] == 2
    assert result.evidence["left_cluster_center"] is None


def test_identity_mismatch_fails_closed() -> None:
    timeline, observations = _timeline(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    mismatched = [dataclasses.replace(item, persistent_id="obj:bottle:other") for item in observations]
    result = MovementEvidenceEvaluator().evaluate(timeline, mismatched, _trigger(timeline))

    assert result.accepted is False
    assert result.reason == "IDENTITY_MISMATCH"
    assert result.event is None


def test_raw_track_fragment_change_does_not_reject_valid_persistent_entity() -> None:
    result, _, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )

    assert {item.source_track_ids[0] for item in observations} == {"track-before", "track-after"}
    assert result.accepted is True
    assert result.evidence["gates"]["identity"]["raw_track_fragment_change_allowed"] is True


def test_transition_gap_over_p2b_limit_fails_closed() -> None:
    result, _, _ = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
        transition_gap=0.51,
    )

    assert result.accepted is False
    assert result.reason == "TEMPORAL_GAP_TOO_LARGE"
    assert result.evidence["gates"]["temporal"]["max_gap_seconds"] == 0.5


def test_repeated_evaluation_is_deterministic() -> None:
    result, timeline, observations = _evaluate(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    repeat = MovementEvidenceEvaluator().evaluate(timeline, observations, _trigger(timeline))

    assert result.evidence == repeat.evidence
    assert result.event is not None and repeat.event is not None
    assert result.event.to_record() == repeat.event.to_record()


def test_evidence_core_does_not_import_perception_dependencies() -> None:
    code = """
import sys
import recallvision.events.movement_evidence
forbidden = {"ultralytics", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("P2D_EVENT_CORE_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "P2D_EVENT_CORE_IMPORT_PASS"


@pytest.mark.skip(
    reason="requires generated local P1B/P2A/P2B/P2C outputs excluded from the public candidate"
)
def test_p2d_cli_emits_inspectable_event_and_diagnostics(tmp_path: Path) -> None:
    from recallvision.cli.p2d import run

    root = Path(__file__).parents[1]
    result = run(
        timeline_path=root / "outputs/p2b/pixabay_bottle/object_state_timelines.json",
        zone_events_path=root / "outputs/p2c/pixabay_bottle/zone_change_events.json",
        consolidated_memory=root / "outputs/p2a_r1/pixabay_bottle/observations_consolidated.jsonl",
        persistent_entities=root / "outputs/p2a_r1/pixabay_bottle/persistent_entities.json",
        entity_summary=root / "outputs/p2a_r1/pixabay_bottle/entity_consolidation_summary.json",
        config_path=root / "configs/p2d_move_object_events.yaml",
        output_dir=tmp_path,
    )

    assert result["p2d_status"] == "PASS"
    assert result["move_object_event_count"] == 1
    assert result["evidence_window_policy"] == "BOUNDARY_LOCAL_TIME_WINDOW"
    assert result["support_horizon_seconds"] == 0.5
    assert result["support_horizon_source"] == (
        "config.max_transition_gap_seconds (P2B max_observation_gap_seconds)"
    )
    assert result["full_source_segments_used_for_cluster_gate"] is False
    events = __import__("json").loads((tmp_path / "move_object_events.json").read_text())
    evidence = __import__("json").loads((tmp_path / "move_object_evidence.json").read_text())
    assert len(events) == 1
    assert events[0]["event_type"] == "MOVE_OBJECT"
    assert events[0]["person_ids"] == []
    assert evidence[0]["status"] == "ACCEPTED"
    assert evidence[0]["trigger_event_id"].startswith("zone-change:")
    assert evidence[0]["bbox_diagnostics"]["before"]["available"] is True


@pytest.mark.parametrize("bad_event_type", [EventType.MOVE_OBJECT, EventType.PERSON_ENTER])
def test_non_zone_change_trigger_is_rejected(bad_event_type: EventType) -> None:
    timeline, _ = _timeline(
        before=((0.100, 0.400), (0.101, 0.400), (0.102, 0.400)),
        after=((0.500, 0.400), (0.501, 0.400), (0.502, 0.400)),
    )
    trigger = _trigger(timeline)
    trigger = dataclasses.replace(trigger, event_type=bad_event_type)
    result = MovementEvidenceEvaluator().evaluate(timeline, [], trigger)

    assert result.accepted is False
    assert result.reason == "ZONE_CHANGE_TRIGGER_REQUIRED"
