"""Tests for mapping an external detector result into Observation."""

from recallvision.memory.schema import BoundingBox
from recallvision.perception.ultralytics_backend import observation_from_detection


def test_detection_mapping_preserves_contract_and_provisional_identity() -> None:
    observation = observation_from_detection(
        video_id="video-1",
        frame_index=30,
        fps=30.0,
        frame_width=900,
        bbox=[0.0, 10.0, 300.0, 110.0],
        class_name="bottle",
        confidence=0.88,
        track_id=7,
        detection_index=0,
    )

    assert observation.video_id == "video-1"
    assert observation.frame_index == 30
    assert observation.timestamp == 1.0
    assert observation.track_id == "7"
    assert observation.persistent_id == "track:7"
    assert observation.entity_type == "object"
    assert observation.class_name == "bottle"
    assert observation.bbox == BoundingBox(x1=0.0, y1=10.0, x2=300.0, y2=110.0)
    assert observation.confidence == 0.88
    assert observation.zone == "left_region"


def test_untracked_detection_does_not_get_long_term_identity() -> None:
    observation = observation_from_detection(
        video_id="video-1",
        frame_index=4,
        fps=25.0,
        frame_width=900,
        bbox=[600.0, 10.0, 900.0, 110.0],
        class_name="person",
        confidence=0.77,
        track_id=None,
        detection_index=2,
    )

    assert observation.track_id == "untracked:4:2"
    assert observation.persistent_id is None
    assert observation.entity_type == "person"
    assert observation.timestamp == 0.16
    assert observation.zone == "right_region"
