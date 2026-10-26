"""CLI-level tests for the R1 persistent-person ROI production path."""

from __future__ import annotations

import json
from pathlib import Path

from recallvision.cli import e1
from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.store import ObservationMemoryStore
from recallvision.pose.roi import compute_person_roi, remap_pose_detection
from recallvision.pose.schema import KeypointObservation, PoseDetection
from recallvision.pose.ultralytics_backend import PoseBackendInfo


def _observation(frame_index: int, persistent_id: str, entity_type: str, bbox: tuple[float, ...]) -> Observation:
    return Observation(
        video_id="video-r1",
        frame_index=frame_index,
        timestamp=frame_index / 10,
        track_id=f"track-{persistent_id}-{frame_index}",
        persistent_id=persistent_id,
        entity_type=entity_type,
        class_name="person" if entity_type == "person" else "bottle",
        bbox=BoundingBox(*bbox),
        confidence=0.9,
    )


def test_e1_run_uses_event_local_persistent_person_rois(monkeypatch, tmp_path: Path) -> None:
    object_id = "obj:bottle:0001"
    person_id = "obj:person:0001"
    event_id = "move-object:video-r1:obj:bottle:0001:1->2"
    events_path = tmp_path / "events.json"
    events_path.write_text(
        json.dumps(
            [
                {
                    "event_id": event_id,
                    "event_type": "MOVE_OBJECT",
                    "start_time": 1.0,
                    "end_time": 1.2,
                    "person_ids": [],
                    "object_ids": [object_id],
                    "evidence": [{"video_id": "video-r1", "frame_indices": [10, 12]}],
                }
            ]
        )
    )
    event_evidence_path = tmp_path / "event_evidence.json"
    event_evidence_path.write_text(
        json.dumps(
            [
                {
                    "event_id": event_id,
                    "left_support_frames": [8, 10],
                    "right_support_frames": [12, 14],
                    "right_support_timestamps": [1.2, 1.4],
                }
            ]
        )
    )
    memory_path = tmp_path / "memory.jsonl"
    observations = []
    for frame in (8, 10, 12, 14, 20):
        observations.extend(
            (
                _observation(frame, object_id, "object", (40, 40, 50, 50)),
                _observation(frame, person_id, "person", (10, 10, 50, 80)),
            )
        )
    for item in observations:
        ObservationMemoryStore(memory_path).append(item)
    entity_summary_path = tmp_path / "entity_summary.json"
    entity_summary_path.write_text(
        json.dumps(
            {
                "video_metadata": {"width": 100, "height": 100},
                "p2a_r1_identity_validation_status": "PENDING_HUMAN_REVIEW",
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "minimum_keypoint_confidence": 0.50,
                "minimum_keypoint_visibility": None,
                "negative_control_horizon_seconds": 0.50,
            }
        )
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"test video placeholder")
    model_path = tmp_path / "yolo11n-pose.pt"
    model_path.write_bytes(b"test model placeholder")
    calls: list[tuple[int, ...]] = []

    class FakeBackend:
        def __init__(self, *, model_path: Path, device: str, confidence_threshold: float) -> None:
            assert device == "cpu"
            assert confidence_threshold == 0.25

        def info(self) -> PoseBackendInfo:
            return PoseBackendInfo(
                backend="fake",
                version="test",
                model_identifier="yolo11n-pose.pt",
                model_path=str(model_path),
                model_sha256="fake-sha",
                keypoint_confidence_source="test",
                visibility_available=False,
            )

        def infer_persistent_person_rois(self, *, person_observations, **kwargs):
            selected = tuple(person_observations)
            calls.append(tuple(item.frame_index for item in selected))
            result = []
            for person in selected:
                roi = compute_person_roi(person.bbox, kwargs["frame_width"], kwargs["frame_height"])
                local = PoseDetection(
                    video_id=person.video_id,
                    frame_index=person.frame_index,
                    timestamp=person.timestamp,
                    detection_index=0,
                    pose_bbox=BoundingBox(0, 0, roi.crop_width, roi.crop_height),
                    keypoints=(
                        KeypointObservation("left_wrist", 30, 30, 0.9),
                        KeypointObservation("right_wrist", 30, 30, 0.9),
                    ),
                    pose_confidence=0.9,
                )
                result.append(remap_pose_detection(local, roi, persistent_person_id=person.persistent_id))
            return tuple(result)

    monkeypatch.setattr(e1, "UltralyticsPoseBackend", FakeBackend)

    record = e1.run(
        video_path=video_path,
        events_path=events_path,
        event_evidence_path=event_evidence_path,
        consolidated_memory=memory_path,
        entity_summary_path=entity_summary_path,
        pose_model=model_path,
        output_dir=tmp_path / "r1-output",
        config_path=config_path,
    )

    assert calls == [(8, 10, 12, 14)]
    assert record["pose_inference_policy"] == "PERSISTENT_PERSON_ROI"
    assert record["configuration"]["roi_margin"] == 0.0
    assert record["pose_observation_provenance"]["roi_margin"] == 0.0
    assert record["pose_observation_provenance"]["selected_frames"] == [8, 10, 12, 14]
    assert record["persons"][person_id]["person_identity_ready_for_attribution"] is False
    assert record["can_define_structured_agent_association_rule"] == "NO"
    assert record["source_move_object_event"]["person_ids"] == []
    assert all(item["inference_policy"] == "PERSISTENT_PERSON_ROI" for item in record["pose_observations"])
