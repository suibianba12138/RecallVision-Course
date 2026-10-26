"""Robust motion candidate generation and shared evidence gate contracts."""

from __future__ import annotations

from recallvision.events.motion_candidates import (
    MotionCandidateConfig,
    RobustMotionEventDetector,
)
from recallvision.events.movement_evidence import MovementObservation
from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef


VIDEO = "sha256:fixture-motion"
ENTITY = "obj:cup:0001"


def _obs(
    frame: int,
    timestamp: float,
    center: tuple[float, float],
    *,
    track: str = "1",
    persistent_id: str = ENTITY,
    bbox_size: tuple[float, float] = (0.04, 0.08),
    zone: str = "center_region",
):
    return MovementObservation(
        persistent_id=persistent_id,
        frame_index=frame,
        timestamp=timestamp,
        zone=zone,
        center_normalized=center,
        source_track_ids=(track,),
        track_id=track,
        bbox=(100.0, 100.0, 140.0, 180.0),
        confidence=0.9,
        bbox_size_normalized=bbox_size,
    )


def _two_position_observations(*, after_time: float = 0.3):
    return [
        _obs(0, 0.0, (0.200, 0.400)),
        _obs(1, 0.1, (0.202, 0.399)),
        _obs(2, 0.2, (0.198, 0.401)),
        _obs(3, after_time, (0.350, 0.400)),
        _obs(4, after_time + 0.1, (0.352, 0.399)),
        _obs(5, after_time + 0.2, (0.348, 0.401)),
    ]


def _static_scene_anchors(observations):
    return [
        _obs(
            item.frame_index,
            item.timestamp,
            center,
            persistent_id=f"obj:anchor:{index}",
            track=f"anchor-track-{index}",
        )
        for index, center in enumerate(((0.70, 0.20), (0.72, 0.50), (0.68, 0.80)), start=1)
        for item in observations
    ]


def _zone_trigger() -> EventRecord:
    return EventRecord(
        event_id="zone-change:fixture:0001",
        event_type=EventType.ZONE_CHANGE,
        start_time=0.2,
        end_time=0.3,
        object_ids=[ENTITY],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[
            EvidenceRef(video_id=VIDEO, timestamp=0.2, frame_indices=[2]),
            EvidenceRef(video_id=VIDEO, timestamp=0.3, frame_indices=[3]),
        ],
        source_segment_ids=("before", "after"),
    )


def test_same_zone_stable_windows_can_support_move_object() -> None:
    observations = _two_position_observations()
    result = RobustMotionEventDetector().evaluate(
        observations + _static_scene_anchors(observations),
        video_id=VIDEO,
    )

    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["event_type"] == "MOVE_OBJECT"
    assert event["zone_before"] == event["zone_after"] == "center_region"
    assert event["trigger_provenance"] == ["CONTINUOUS_DISPLACEMENT"]
    assert event["position_before"] != event["position_after"]
    assert event["observation_references"]
    assert event["displacement"]["scene_compensated"] is False
    assert event["scene_transform_source"] == "COMMON_TRACKS"


def test_gap_over_policy_limit_is_candidate_but_not_accepted_or_interpolated() -> None:
    result = RobustMotionEventDetector().evaluate(
        _two_position_observations(after_time=1.0),
        video_id=VIDEO,
    )

    assert result["events"] == []
    candidates = [item for item in result["diagnostics"] if item["candidate_type"] == "CONTINUOUS_DISPLACEMENT"]
    assert candidates
    assert candidates[0]["status"] == "INSUFFICIENT_EVIDENCE"
    assert candidates[0]["reason"] == "TEMPORAL_GAP_TOO_LARGE"
    assert candidates[0]["gap_seconds"] == 0.8
    assert set(candidates[0]["observation_frames"]) == {0, 1, 2, 3, 4, 5}


def test_zone_support_windows_cannot_hide_an_internal_detection_gap() -> None:
    observations = [
        _obs(0, 0.0, (0.20, 0.40), zone="left_region"),
        _obs(6, 0.6, (0.20, 0.40), zone="left_region"),
        _obs(7, 0.7, (0.40, 0.40), zone="center_region"),
        _obs(8, 0.8, (0.40, 0.40), zone="center_region"),
    ]
    event = EventRecord(
        event_id="zone-change:internal-gap",
        event_type=EventType.ZONE_CHANGE,
        start_time=0.6,
        end_time=0.7,
        object_ids=[ENTITY],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[
            EvidenceRef(video_id=VIDEO, timestamp=0.6, frame_indices=[6]),
            EvidenceRef(video_id=VIDEO, timestamp=0.7, frame_indices=[7]),
        ],
    )
    observations += _static_scene_anchors(observations)

    result = RobustMotionEventDetector().evaluate(
        observations,
        zone_events=[event],
        video_id=VIDEO,
    )

    assert result["events"] == []
    diagnostic = next(item for item in result["diagnostics"] if item["candidate_type"] == "ZONE_CHANGE")
    assert diagnostic["reason"] == "TEMPORAL_GAP_TOO_LARGE"
    assert diagnostic["gap_seconds"] == 0.6


def test_zone_candidate_uses_same_robust_gate_for_duplicate_track_boxes() -> None:
    observations = []
    positions = {
        0: ((0.370, 0.18), (0.371, 0.18)),
        1: ((0.369, 0.18), (0.370, 0.18)),
        2: ((0.329, 0.18), (0.399, 0.18), (0.370, 0.18)),
        3: ((0.369, 0.18),),
        4: ((0.370, 0.18),),
        5: ((0.371, 0.18),),
    }
    for frame, centers in positions.items():
        for duplicate_index, center in enumerate(centers):
            observations.append(
                _obs(frame, frame * 0.1, center, track=str(duplicate_index + 1))
            )

    result = RobustMotionEventDetector().evaluate(
        observations,
        zone_events=[_zone_trigger()],
        video_id=VIDEO,
    )

    assert result["events"] == []
    zone_diagnostics = [item for item in result["diagnostics"] if item["candidate_type"] == "ZONE_CHANGE"]
    assert zone_diagnostics
    assert zone_diagnostics[0]["status"] == "INSUFFICIENT_EVIDENCE"
    assert zone_diagnostics[0]["reason"] == "AMBIGUOUS_DUPLICATE_GEOMETRY"
    assert zone_diagnostics[0]["duplicate_frame_count"] >= 1


def test_zone_and_continuous_triggers_for_one_motion_are_deduplicated() -> None:
    observations = _two_position_observations()
    result = RobustMotionEventDetector().evaluate(
        observations + _static_scene_anchors(observations),
        zone_events=[_zone_trigger()],
        video_id=VIDEO,
    )

    assert len(result["events"]) == 1
    event = result["events"][0]
    assert set(event["trigger_provenance"]) == {"ZONE_CHANGE", "CONTINUOUS_DISPLACEMENT"}
    assert result["summary"]["accepted_event_count"] == 1
    assert result["summary"]["duplicate_event_count"] == 1
    duplicate_diagnostics = [item for item in result["diagnostics"] if item["reason"] == "DUPLICATE_EVENT"]
    assert len(duplicate_diagnostics) == 1
    assert duplicate_diagnostics[0]["status"] == "REJECTED"
    assert duplicate_diagnostics[0]["evidence_status"] == "REJECTED"


def test_shot_boundary_is_a_conservative_insufficient_evidence_guard() -> None:
    result = RobustMotionEventDetector().evaluate(
        _two_position_observations(),
        video_id=VIDEO,
        shot_boundary_frames={3},
    )

    assert result["events"] == []
    assert any(
        item["status"] == "INSUFFICIENT_EVIDENCE"
        and item["reason"] == "SHOT_BOUNDARY_CROSSED"
        for item in result["diagnostics"]
    )


def test_identity_provenance_is_recorded_without_promoting_review_status() -> None:
    observations = _two_position_observations()
    result = RobustMotionEventDetector().evaluate(
        observations + _static_scene_anchors(observations),
        video_id=VIDEO,
        identity_provenance={
            ENTITY: {
                "identity_provenance": "HUMAN_CONFIRMED",
                "source_identity_review_status": "PENDING_HUMAN_REVIEW",
                "overlay_revision": 1,
            }
        },
    )

    assert len(result["events"]) == 1
    provenance = result["events"][0]["identity_provenance"]
    assert provenance["identity_provenance"] == "HUMAN_CONFIRMED"
    assert provenance["source_identity_review_status"] == "PENDING_HUMAN_REVIEW"


def test_zone_boundary_uses_two_observed_positions_to_retain_c1_style_positive() -> None:
    observations = [
        _obs(90, 3.00, (0.225, 0.36)),
        _obs(92, 3.07, (0.257, 0.36)),
        _obs(99, 3.30, (0.361, 0.35)),
        _obs(100, 3.34, (0.374, 0.35)),
    ]
    event = EventRecord(
        event_id="zone-change:c1-style",
        event_type=EventType.ZONE_CHANGE,
        start_time=3.07,
        end_time=3.30,
        object_ids=[ENTITY],
        zone_before="left_region",
        zone_after="center_region",
        evidence=[
            EvidenceRef(video_id=VIDEO, timestamp=3.07, frame_indices=[92]),
            EvidenceRef(video_id=VIDEO, timestamp=3.30, frame_indices=[99]),
        ],
    )

    unanchored = RobustMotionEventDetector().evaluate(
        observations,
        video_id=VIDEO,
        zone_events=[event],
    )

    assert unanchored["events"] == []
    assert any(item["reason"] == "SCENE_CONTINUITY_UNVERIFIED" for item in unanchored["diagnostics"])

    result = RobustMotionEventDetector().evaluate(
        observations + _static_scene_anchors(observations),
        video_id=VIDEO,
        zone_events=[event],
    )

    assert len(result["events"]) == 1
    move = result["events"][0]
    assert move["start_evidence"]["frame_indices"] == [90, 92]
    assert move["end_evidence"]["frame_indices"] == [99, 100]
    assert "ZONE_CHANGE" in move["trigger_provenance"]


def test_same_frame_conflicting_track_boxes_are_not_median_collapsed_into_motion() -> None:
    observations = []
    for frame, timestamp, centers in (
        (0, 0.0, ((0.20, 0.40), (0.38, 0.40))),
        (1, 0.1, ((0.20, 0.40), (0.38, 0.40))),
        (2, 0.2, ((0.20, 0.40), (0.38, 0.40))),
        (3, 0.3, ((0.35, 0.40), (0.53, 0.40))),
        (4, 0.4, ((0.35, 0.40), (0.53, 0.40))),
        (5, 0.5, ((0.35, 0.40), (0.53, 0.40))),
    ):
        observations.extend(
            _obs(frame, timestamp, center, track=f"duplicate-{idx}")
            for idx, center in enumerate(centers)
        )

    result = RobustMotionEventDetector().evaluate(observations, video_id=VIDEO)

    assert result["events"] == []
    assert any(item["reason"] == "AMBIGUOUS_DUPLICATE_GEOMETRY" for item in result["diagnostics"])


def test_unlocalized_full_width_boxes_cannot_support_object_motion() -> None:
    observations = [
        _obs(i, i * 0.1, center, bbox_size=(0.93, 0.52))
        for i, center in enumerate(((0.45, 0.72), (0.45, 0.72), (0.45, 0.72), (0.51, 0.75), (0.51, 0.75), (0.51, 0.75)))
    ]
    observations += _static_scene_anchors(observations)

    result = RobustMotionEventDetector().evaluate(observations, video_id=VIDEO)

    assert result["events"] == []
    assert any(item["reason"] == "BBOX_LOCALIZATION_TOO_BROAD" for item in result["diagnostics"])


def test_bbox_scale_jump_cannot_turn_center_shift_into_object_motion() -> None:
    observations = [
        _obs(i, i * 0.1, center, bbox_size=(0.22, 0.90))
        for i, center in enumerate(((0.50, 0.50),) * 3)
    ]
    observations += [
        _obs(i, i * 0.1, center, bbox_size=(0.22, 0.56))
        for i, center in zip((3, 4, 5), ((0.50, 0.33),) * 3)
    ]
    observations += _static_scene_anchors(observations)

    result = RobustMotionEventDetector().evaluate(observations, video_id=VIDEO)

    assert result["events"] == []
    assert any(item["reason"] == "BBOX_SCALE_CHANGE_TOO_LARGE" for item in result["diagnostics"])


def test_person_track_is_not_reported_as_move_object() -> None:
    person_id = "obj:person:0001"
    observations = [
        _obs(i, i * 0.1, center, persistent_id=person_id)
        for i, center in enumerate(((0.20, 0.40), (0.20, 0.40), (0.20, 0.40), (0.40, 0.40), (0.40, 0.40), (0.40, 0.40)))
    ]

    result = RobustMotionEventDetector().evaluate(observations, video_id=VIDEO)

    assert result["events"] == []
    assert any(item["reason"] == "NON_OBJECT_ENTITY_CLASS" for item in result["diagnostics"])


def test_unanchored_same_zone_motion_remains_insufficient() -> None:
    result = RobustMotionEventDetector().evaluate(_two_position_observations(), video_id=VIDEO)

    assert result["events"] == []
    assert any(item["reason"] == "SCENE_CONTINUITY_UNVERIFIED" for item in result["diagnostics"])


def test_coherent_camera_translation_is_removed_before_acceptance() -> None:
    target = [
        _obs(i, i * 0.1, center)
        for i, center in enumerate(((0.20, 0.40), (0.20, 0.40), (0.20, 0.40), (0.28, 0.34), (0.28, 0.34), (0.28, 0.34)))
    ]
    anchors = []
    shifts = (0.08, -0.06)
    for idx, center in enumerate(((0.70, 0.20), (0.72, 0.50), (0.68, 0.80)), start=1):
        for i in range(6):
            point = (center[0], center[1]) if i < 3 else (center[0] + shifts[0], center[1] + shifts[1])
            anchors.append(_obs(i, i * 0.1, point, persistent_id=f"obj:anchor:{idx}", track=f"anchor-{idx}"))

    result = RobustMotionEventDetector().evaluate(target + anchors, video_id=VIDEO)

    assert result["events"] == []
    diagnostic = next(item for item in result["diagnostics"] if item["effective_entity_id"] == ENTITY)
    assert diagnostic["reason"] == "SCENE_SHIFT_EXPLAINS_DISPLACEMENT"
    assert diagnostic["gates"]["scene_compensation"]["anchor_count"] == 3


def test_video_camera_transform_is_compensated_without_suppressing_relative_motion(monkeypatch) -> None:
    from recallvision.events import motion_candidates

    target = [
        _obs(i, i * 0.1, center)
        for i, center in enumerate(((0.20, 0.40),) * 3 + ((0.38, 0.34),) * 3)
    ]
    monkeypatch.setattr(
        motion_candidates,
        "_estimate_video_motion",
        lambda *_args, **_kwargs: {
            "status": "ESTIMATED",
            "transform_normalized": [[1.0, 0.0, 0.08], [0.0, 1.0, -0.06]],
            "inlier_count": 80,
            "inlier_ratio": 0.8,
        },
    )

    result = RobustMotionEventDetector().evaluate(
        target,
        video_id=VIDEO,
        video_path="fixture.mp4",
    )

    assert len(result["events"]) == 1
    assert result["events"][0]["displacement"]["normalized_l2"] > 0.03
    assert result["events"][0]["scene_continuity_status"] == "VIDEO_FEATURE_MOTION_COMPENSATED"
    assert result["events"][0]["displacement"]["scene_compensated"] is True
    assert result["events"][0]["scene_transform_source"] == "VIDEO_FEATURES"


def test_video_translation_explaining_target_shift_rejects_move(monkeypatch) -> None:
    from recallvision.events import motion_candidates

    target = [
        _obs(i, i * 0.1, center)
        for i, center in enumerate(((0.20, 0.40),) * 3 + ((0.28, 0.34),) * 3)
    ]
    monkeypatch.setattr(
        motion_candidates,
        "_estimate_video_motion",
        lambda *_args, **_kwargs: {
            "status": "ESTIMATED",
            "transform_normalized": [[1.0, 0.0, 0.08], [0.0, 1.0, -0.06]],
            "inlier_count": 80,
            "inlier_ratio": 0.8,
        },
    )

    result = RobustMotionEventDetector().evaluate(
        target,
        video_id=VIDEO,
        video_path="fixture.mp4",
    )

    assert result["events"] == []
    assert any(item["reason"] == "SCENE_SHIFT_EXPLAINS_DISPLACEMENT" for item in result["diagnostics"])


def test_video_feature_estimation_failure_is_fail_closed(monkeypatch) -> None:
    from recallvision.events import motion_candidates

    target = _two_position_observations()
    monkeypatch.setattr(
        motion_candidates,
        "_estimate_video_motion",
        lambda *_args, **_kwargs: {"status": "UNVERIFIED", "reason": "INSUFFICIENT_FEATURE_MATCHES"},
    )

    result = RobustMotionEventDetector().evaluate(
        target,
        video_id=VIDEO,
        video_path="fixture.mp4",
    )

    assert result["events"] == []
    assert any(item["reason"] == "SCENE_CONTINUITY_UNVERIFIED" for item in result["diagnostics"])


def test_adjacent_stable_transitions_coalesce_into_one_motion_episode() -> None:
    target = [
        _obs(i, i * 0.1, center)
        for i, center in enumerate(
            ((0.20, 0.40),) * 3 + ((0.30, 0.40),) * 3 + ((0.40, 0.40),) * 3
        )
    ]
    result = RobustMotionEventDetector().evaluate(
        target + _static_scene_anchors(target),
        video_id=VIDEO,
    )

    assert len(result["events"]) == 1
    assert result["summary"]["coalesced_event_count"] == 1
    assert result["events"][0]["start_evidence"]["center_normalized"] == [0.2, 0.4]
    assert result["events"][0]["end_evidence"]["center_normalized"] == [0.4, 0.4]
