"""Focused tests for persistent-person ROI pose evidence."""

from __future__ import annotations

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.pose.association import associate_frame_poses
from recallvision.pose.evidence import EvidenceConfig
from recallvision.pose.roi import (
    compute_person_roi,
    remap_pose_detection,
    select_person_roi_observations,
)
from recallvision.pose.schema import KeypointObservation, PoseDetection


VIDEO_ID = "video-roi"


def _person(frame_index: int, persistent_id: str = "obj:person:0001") -> Observation:
    return Observation(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 10,
        track_id=f"track-{frame_index}",
        persistent_id=persistent_id,
        entity_type="person",
        class_name="person",
        bbox=BoundingBox(10.25, 20.75, 50.9, 80.1),
        confidence=0.9,
    )


def _pose(frame_index: int, *, detection_index: int = 0) -> PoseDetection:
    return PoseDetection(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 10,
        detection_index=detection_index,
        pose_bbox=BoundingBox(0.0, 0.0, 40.0, 60.0),
        keypoints=(KeypointObservation("left_wrist", 1.5, 2.5, 0.9),),
        pose_confidence=0.9,
    )


def test_zero_margin_roi_uses_canonical_person_bbox() -> None:
    roi = compute_person_roi(_person(1).bbox, frame_width=100, frame_height=100, margin=0.0)

    assert roi.margin == 0.0
    assert roi.canonical_bbox == _person(1).bbox
    assert roi.crop_bbox == (10, 20, 51, 81)
    assert roi.crop_width == 41
    assert roi.crop_height == 61
    assert roi.clipped is False


def test_roi_context_is_clipped_to_image_bounds() -> None:
    roi = compute_person_roi(BoundingBox(0.2, 0.2, 10.7, 10.8), 100, 100, margin=1.0)

    assert roi.crop_bbox == (0, 0, 22, 22)
    assert roi.clipped is True


def test_roi_keypoints_remap_to_global_coordinates_and_preserve_provenance() -> None:
    roi = compute_person_roi(BoundingBox(10.25, 20.75, 50.9, 80.1), 100, 100, margin=0.0)

    remapped = remap_pose_detection(_pose(1), roi, persistent_person_id="obj:person:0001")

    assert remapped.keypoint("left_wrist") == KeypointObservation("left_wrist", 11.5, 22.5, 0.9)
    assert remapped.pose_bbox == BoundingBox(10.0, 20.0, 50.0, 80.0)
    assert remapped.inference_policy == "PERSISTENT_PERSON_ROI"
    assert remapped.roi_provenance["crop_bbox"] == [10, 20, 51, 81]
    assert remapped.roi_provenance["local_keypoints"][0]["x"] == 1.5


def test_roi_association_accepts_one_compatible_pose_after_remapping() -> None:
    person = _person(1)
    roi = compute_person_roi(person.bbox, 100, 100, margin=0.0)
    result = associate_frame_poses(
        (remap_pose_detection(_pose(1), roi, persistent_person_id=person.persistent_id),),
        (person,),
        frame_width=100,
        frame_height=100,
        pose_backend="test-backend",
    )

    assert result.status == "MATCHED"
    assert len(result.observations) == 1
    assert result.observations[0].keypoint("left_wrist").x == 11.5
    assert result.observations[0].inference_policy == "PERSISTENT_PERSON_ROI"


def test_zero_roi_pose_detections_fail_closed() -> None:
    result = associate_frame_poses(
        (),
        (_person(1),),
        frame_width=100,
        frame_height=100,
        pose_backend="test-backend",
    )

    assert result.status == "NO_MATCH"
    assert result.observations == ()


def test_multiple_roi_poses_matching_one_person_fail_closed() -> None:
    person = _person(1)
    roi = compute_person_roi(person.bbox, 100, 100, margin=0.0)
    poses = (
        remap_pose_detection(_pose(1, detection_index=0), roi, persistent_person_id=person.persistent_id),
        remap_pose_detection(_pose(1, detection_index=1), roi, persistent_person_id=person.persistent_id),
    )

    result = associate_frame_poses(
        poses,
        (person,),
        frame_width=100,
        frame_height=100,
        pose_backend="test-backend",
    )

    assert result.status == "AMBIGUOUS"
    assert result.observations == ()


def test_nonzero_production_roi_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="roi_margin must be 0.0"):
        EvidenceConfig(roi_margin=0.15)


def test_event_local_roi_selection_excludes_unselected_frames() -> None:
    selected = select_person_roi_observations(
        (_person(1), _person(2), _person(3)), selected_frames=(1, 3)
    )

    assert [item.frame_index for item in selected] == [1, 3]


def test_remapped_geometry_matches_equivalent_global_keypoints() -> None:
    from recallvision.pose.geometry import measure_wrist_proximity

    person = _person(1)
    roi = compute_person_roi(person.bbox, 100, 100, margin=0.0)
    remapped = remap_pose_detection(_pose(1), roi, persistent_person_id=person.persistent_id)
    object_bbox = BoundingBox(11.0, 22.0, 15.0, 26.0)

    roi_measurement = measure_wrist_proximity(
        remapped.keypoint("left_wrist"), object_bbox, frame_width=100, frame_height=100, minimum_confidence=0.5
    )
    global_measurement = measure_wrist_proximity(
        KeypointObservation("left_wrist", 11.5, 22.5, 0.9),
        object_bbox,
        frame_width=100,
        frame_height=100,
        minimum_confidence=0.5,
    )

    assert roi_measurement == global_measurement
