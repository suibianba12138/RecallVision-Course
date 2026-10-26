"""Dependency-free tests for AGENT-E1 pose/wrist evidence."""

from __future__ import annotations

import subprocess
import sys

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.pose.association import associate_frame_poses
from recallvision.pose.evidence import (
    EvidenceConfig,
    build_interaction_evidence,
    select_event_local_frames,
)
from recallvision.pose.geometry import (
    measure_wrist_proximity,
    point_to_bbox_distance,
    select_closest_wrist,
)
from recallvision.pose.schema import (
    KeypointObservation,
    PersonPoseObservation,
    PoseDetection,
)


VIDEO_ID = "video-e1"
OBJECT_ID = "obj:bottle:0001"
PERSON_ID = "obj:person:0001"


def _observation(
    *,
    frame_index: int,
    persistent_id: str,
    entity_type: str,
    bbox: tuple[float, float, float, float],
    timestamp: float | None = None,
) -> Observation:
    return Observation(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 10 if timestamp is None else timestamp,
        track_id=f"track-{persistent_id}-{frame_index}",
        persistent_id=persistent_id,
        entity_type=entity_type,
        class_name="person" if entity_type == "person" else "bottle",
        bbox=BoundingBox(*bbox),
        confidence=0.9,
        zone="center_region",
    )


def _pose_detection(
    *,
    frame_index: int,
    bbox: tuple[float, float, float, float],
    left_wrist: tuple[float, float] = (20.0, 20.0),
    right_wrist: tuple[float, float] = (30.0, 20.0),
    confidence: float = 0.9,
) -> PoseDetection:
    return PoseDetection(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 10,
        detection_index=0,
        pose_bbox=BoundingBox(*bbox),
        keypoints=(
            KeypointObservation("left_wrist", left_wrist[0], left_wrist[1], confidence),
            KeypointObservation("right_wrist", right_wrist[0], right_wrist[1], confidence),
        ),
        pose_confidence=0.9,
    )


def _person_pose(
    *,
    frame_index: int,
    person_id: str = PERSON_ID,
    left_wrist: tuple[float, float] = (20.0, 20.0),
    right_wrist: tuple[float, float] = (30.0, 20.0),
) -> PersonPoseObservation:
    return PersonPoseObservation(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 10,
        persistent_person_id=person_id,
        source_person_observation_ref=f"{VIDEO_ID}#frame={frame_index}#{person_id}",
        pose_backend="test-backend",
        pose_bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
        keypoints=(
            KeypointObservation("left_wrist", left_wrist[0], left_wrist[1], 0.9),
            KeypointObservation("right_wrist", right_wrist[0], right_wrist[1], 0.9),
        ),
        association_diagnostics={"status": "MATCHED"},
    )


def _event() -> dict[str, object]:
    return {
        "event_id": "move-object:video-e1:obj:bottle:0001:1->2",
        "event_type": "MOVE_OBJECT",
        "start_time": 1.0,
        "end_time": 1.2,
        "object_ids": [OBJECT_ID],
        "person_ids": [],
        "evidence": [{"video_id": VIDEO_ID, "frame_indices": [10, 12]}],
    }


def _event_diagnostic() -> dict[str, object]:
    return {
        "event_id": "move-object:video-e1:obj:bottle:0001:1->2",
        "left_support_frames": [8, 10],
        "right_support_frames": [12, 14],
        "left_support_timestamps": [0.8, 1.0],
        "right_support_timestamps": [1.2, 1.4],
        "support_horizon_seconds": 0.5,
    }


def test_point_to_bbox_distance_is_normalized_in_image_coordinates() -> None:
    bbox = BoundingBox(10.0, 10.0, 20.0, 20.0)

    assert point_to_bbox_distance((15.0, 15.0), bbox, frame_width=100, frame_height=100) == 0.0
    assert point_to_bbox_distance((30.0, 15.0), bbox, frame_width=100, frame_height=100) == 0.1


def test_inside_wrist_is_overlap_candidate_not_contact() -> None:
    measurement = measure_wrist_proximity(
        KeypointObservation("left_wrist", 15.0, 15.0, 0.9),
        BoundingBox(10.0, 10.0, 20.0, 20.0),
        frame_width=100,
        frame_height=100,
        minimum_confidence=0.5,
    )

    assert measurement is not None
    assert measurement.distance_to_bbox == 0.0
    assert measurement.inside_object_bbox is True
    assert measurement.semantic_label == "WRIST_OBJECT_OVERLAP_CANDIDATE"
    assert measurement.semantic_label != "HAND_OBJECT_CONTACT"


def test_two_wrists_are_preserved_and_closest_wrist_is_selected() -> None:
    object_bbox = BoundingBox(40.0, 40.0, 50.0, 50.0)
    left = measure_wrist_proximity(
        KeypointObservation("left_wrist", 0.0, 0.0, 0.9), object_bbox,
        frame_width=100, frame_height=100, minimum_confidence=0.5,
    )
    right = measure_wrist_proximity(
        KeypointObservation("right_wrist", 45.0, 45.0, 0.9), object_bbox,
        frame_width=100, frame_height=100, minimum_confidence=0.5,
    )

    assert left is not None and right is not None
    assert {left.wrist_name, right.wrist_name} == {"left_wrist", "right_wrist"}
    assert select_closest_wrist((left, right)).wrist_name == "right_wrist"


def test_low_confidence_and_missing_wrists_fail_closed() -> None:
    low = measure_wrist_proximity(
        KeypointObservation("left_wrist", 15.0, 15.0, 0.49),
        BoundingBox(10.0, 10.0, 20.0, 20.0),
        frame_width=100, frame_height=100, minimum_confidence=0.5,
    )
    missing = measure_wrist_proximity(
        KeypointObservation("right_wrist", 15.0, 15.0, None),
        BoundingBox(10.0, 10.0, 20.0, 20.0),
        frame_width=100, frame_height=100, minimum_confidence=0.5,
    )

    assert low is None
    assert missing is None
    assert select_closest_wrist(()) is None


def test_pose_person_association_accepts_one_unambiguous_same_frame_match() -> None:
    pose = _pose_detection(frame_index=10, bbox=(0.0, 0.0, 100.0, 100.0))
    person = _observation(
        frame_index=10, persistent_id=PERSON_ID, entity_type="person", bbox=(5.0, 5.0, 95.0, 95.0)
    )

    result = associate_frame_poses(
        (pose,), (person,), frame_width=100, frame_height=100, pose_backend="test-backend"
    )

    assert result.status == "MATCHED"
    assert [item.persistent_person_id for item in result.observations] == [PERSON_ID]
    assert result.diagnostics[0]["status"] == "MATCHED"


def test_pose_person_association_fails_closed_on_ambiguity() -> None:
    pose = _pose_detection(frame_index=10, bbox=(0.0, 0.0, 100.0, 100.0))
    persons = (
        _observation(frame_index=10, persistent_id="obj:person:0001", entity_type="person", bbox=(0.0, 0.0, 80.0, 100.0)),
        _observation(frame_index=10, persistent_id="obj:person:0002", entity_type="person", bbox=(20.0, 0.0, 100.0, 100.0)),
    )

    result = associate_frame_poses(
        (pose,), persons, frame_width=100, frame_height=100, pose_backend="test-backend"
    )

    assert result.status == "AMBIGUOUS"
    assert result.observations == ()
    assert result.diagnostics[0]["status"] == "AMBIGUOUS"


def test_event_local_frame_selection_excludes_far_observations() -> None:
    observations = (
        _observation(frame_index=5, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=0.5),
        _observation(frame_index=8, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=0.8),
        _observation(frame_index=10, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=1.0),
        _observation(frame_index=14, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=1.4),
        _observation(frame_index=20, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=2.0),
    )

    window = select_event_local_frames(
        _event(), _event_diagnostic(), observations, horizon_seconds=0.5
    )

    assert window.event_frames == (8, 10, 12, 14)
    assert window.pre_control_frames == (5, 8)
    assert window.post_control_frames == ()
    assert 20 not in window.all_frames


def test_negative_controls_remain_separate_from_event_summary() -> None:
    object_observations = (
        _observation(frame_index=1, persistent_id=OBJECT_ID, entity_type="object", bbox=(80, 80, 90, 90), timestamp=0.1),
        _observation(frame_index=2, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=0.2),
    )
    person_observations = (
        _observation(frame_index=1, persistent_id=PERSON_ID, entity_type="person", bbox=(0, 0, 100, 100), timestamp=0.1),
        _observation(frame_index=2, persistent_id=PERSON_ID, entity_type="person", bbox=(0, 0, 100, 100), timestamp=0.2),
    )
    event = {**_event(), "start_time": 0.2, "end_time": 0.2, "evidence": [{"video_id": VIDEO_ID, "frame_indices": [2]}]}
    window = select_event_local_frames(
        event,
        {"event_id": event["event_id"], "left_support_frames": [2], "right_support_frames": [2]},
        (*object_observations, *person_observations),
        horizon_seconds=0.5,
    )
    pose_observations = (
        _person_pose(frame_index=1, left_wrist=(0.0, 0.0), right_wrist=(0.0, 0.0)),
        _person_pose(frame_index=2, left_wrist=(15.0, 15.0), right_wrist=(0.0, 0.0)),
    )

    report = build_interaction_evidence(
        event=event,
        event_window=window,
        observations=(*object_observations, *person_observations),
        pose_observations=pose_observations,
        frame_width=100,
        frame_height=100,
        config=EvidenceConfig(minimum_keypoint_confidence=0.5),
    )

    person_report = report["persons"][PERSON_ID]
    assert person_report["event_summary"]["valid_wrist_frames"] == 1
    assert person_report["negative_control_summary"]["valid_wrist_frames"] == 1
    assert person_report["event_summary"]["minimum_distance"] < person_report["negative_control_summary"]["minimum_distance"]


def test_event_evidence_is_deterministic_for_identical_structured_inputs() -> None:
    object_observation = _observation(
        frame_index=2, persistent_id=OBJECT_ID, entity_type="object", bbox=(10, 10, 20, 20), timestamp=0.2
    )
    person_observation = _observation(
        frame_index=2, persistent_id=PERSON_ID, entity_type="person", bbox=(0, 0, 100, 100), timestamp=0.2
    )
    event = {**_event(), "start_time": 0.2, "end_time": 0.2, "evidence": [{"video_id": VIDEO_ID, "frame_indices": [2]}]}
    window = select_event_local_frames(
        event, {"event_id": event["event_id"], "left_support_frames": [2], "right_support_frames": [2]},
        (object_observation, person_observation), horizon_seconds=0.5,
    )
    inputs = dict(
        event=event,
        event_window=window,
        observations=(object_observation, person_observation),
        pose_observations=(_person_pose(frame_index=2, left_wrist=(15.0, 15.0)),),
        frame_width=100,
        frame_height=100,
        config=EvidenceConfig(),
    )

    assert build_interaction_evidence(**inputs) == build_interaction_evidence(**inputs)


def test_local_pose_backend_rejects_missing_weights_without_loading_optional_stack(tmp_path) -> None:
    from recallvision.pose.ultralytics_backend import UltralyticsPoseBackend

    with pytest.raises(FileNotFoundError):
        UltralyticsPoseBackend(model_path=tmp_path / "missing-pose.pt")


def test_e1_cli_import_does_not_load_optional_perception_packages() -> None:
    code = """
import sys
import recallvision.cli.e1
forbidden = {"ultralytics", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("E1_CLI_IMPORT_PASS")
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "E1_CLI_IMPORT_PASS"
