"""Run event-local AGENT-E1 pose/wrist interaction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from recallvision.memory.store import ObservationMemoryStore
from recallvision.pose.association import associate_frame_poses
from recallvision.pose.evidence import EvidenceConfig, build_interaction_evidence, select_event_local_frames
from recallvision.pose.roi import compute_person_roi, select_person_roi_observations
from recallvision.pose.ultralytics_backend import UltralyticsPoseBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(path: Path, label: str) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{label} must contain a JSON object or list")
    return value


def _json_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = _json_value(path, label)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def _frame_dimensions(summary: dict[str, Any]) -> tuple[float, float]:
    metadata = summary.get("video_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("entity summary is missing video_metadata")
    width, height = metadata.get("width"), metadata.get("height")
    if isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0:
        raise ValueError("video_metadata.width must be positive")
    if isinstance(height, bool) or not isinstance(height, (int, float)) or height <= 0:
        raise ValueError("video_metadata.height must be positive")
    return float(width), float(height)


def _select_event(events: list[dict[str, Any]], event_id: str | None) -> dict[str, Any]:
    candidates = [item for item in events if item.get("event_type") == "MOVE_OBJECT"]
    if event_id is not None:
        candidates = [item for item in candidates if item.get("event_id") == event_id]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one MOVE_OBJECT event, found {len(candidates)}")
    return candidates[0]


def _select_diagnostic(diagnostics: list[dict[str, Any]], event_id: str) -> dict[str, Any]:
    matches = [item for item in diagnostics if item.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one P2D diagnostic for {event_id}, found {len(matches)}")
    return matches[0]


def _config_from_path(path: Path | None) -> EvidenceConfig:
    if path is None:
        return EvidenceConfig()
    value = _json_value(path, "E1 config")
    if not isinstance(value, dict):
        raise ValueError("E1 config must be a JSON object")
    return EvidenceConfig(
        minimum_keypoint_confidence=float(value.get("minimum_keypoint_confidence", 0.50)),
        minimum_keypoint_visibility=(
            None
            if value.get("minimum_keypoint_visibility") is None
            else float(value["minimum_keypoint_visibility"])
        ),
        negative_control_horizon_seconds=float(value.get("negative_control_horizon_seconds", 0.50)),
        roi_margin=float(value.get("roi_margin", 0.0)),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_head() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _roi_attempts(
    *,
    person_observations: tuple[Any, ...],
    pose_detections: tuple[Any, ...],
    frame_width: float,
    frame_height: float,
    roi_margin: float,
) -> list[dict[str, Any]]:
    """Record every event-local canonical ROI, including zero-pose frames."""

    detections_by_key: dict[tuple[int, str], list[Any]] = {}
    for detection in pose_detections:
        person_id = (detection.roi_provenance or {}).get("persistent_person_id")
        if person_id is None:
            continue
        detections_by_key.setdefault((detection.frame_index, str(person_id)), []).append(detection)
    attempts: list[dict[str, Any]] = []
    for person in person_observations:
        if person.persistent_id is None:
            continue
        roi = compute_person_roi(
            person.bbox,
            frame_width,
            frame_height,
            margin=roi_margin,
        )
        attempts.append(
            {
                "video_id": person.video_id,
                "frame_index": person.frame_index,
                "timestamp": float(person.timestamp),
                "persistent_person_id": str(person.persistent_id),
                **roi.to_record(),
                "pose_detection_count": len(
                    detections_by_key.get((person.frame_index, str(person.persistent_id)), [])
                ),
            }
        )
    return attempts


def run(
    *,
    video_path: Path,
    events_path: Path,
    event_evidence_path: Path,
    consolidated_memory: Path,
    entity_summary_path: Path,
    pose_model: Path,
    output_dir: Path,
    event_id: str | None = None,
    config_path: Path | None = None,
    device: str = "cpu",
    pose_confidence: float = 0.25,
) -> dict[str, Any]:
    """Run only event-local pose inference over frozen memory/event artifacts."""

    events = _json_list(events_path, "MOVE_OBJECT events")
    event = _select_event(events, event_id)
    diagnostics = _json_list(event_evidence_path, "P2D evidence")
    diagnostic = _select_diagnostic(diagnostics, str(event["event_id"]))
    observations = tuple(ObservationMemoryStore(consolidated_memory).load_all())
    if not observations:
        raise ValueError("consolidated memory is empty")
    entity_summary = _json_value(entity_summary_path, "entity summary")
    if not isinstance(entity_summary, dict):
        raise ValueError("entity summary must be a JSON object")
    frame_width, frame_height = _frame_dimensions(entity_summary)
    config = _config_from_path(config_path)
    window = select_event_local_frames(
        event,
        diagnostic,
        observations,
        horizon_seconds=config.negative_control_horizon_seconds,
    )
    backend = UltralyticsPoseBackend(
        model_path=pose_model,
        device=device,
        confidence_threshold=pose_confidence,
    )
    backend_info = backend.info()
    video_id = str(event["evidence"][0]["video_id"])
    event_local_person_observations = select_person_roi_observations(
        observations,
        selected_frames=window.all_frames,
    )
    raw_pose_detections = backend.infer_persistent_person_rois(
        video_path=video_path,
        video_id=video_id,
        person_observations=event_local_person_observations,
        frame_width=frame_width,
        frame_height=frame_height,
        roi_margin=config.roi_margin,
    )
    persons_by_frame: dict[int, list[Any]] = {}
    for observation in observations:
        if observation.entity_type == "person" and observation.persistent_id is not None:
            persons_by_frame.setdefault(observation.frame_index, []).append(observation)
    pose_observations = []
    association_diagnostics: list[dict[str, Any]] = []
    for frame_index in sorted(set(item.frame_index for item in raw_pose_detections)):
        frame_detections = [item for item in raw_pose_detections if item.frame_index == frame_index]
        association = associate_frame_poses(
            frame_detections,
            persons_by_frame.get(frame_index, []),
            frame_width=frame_width,
            frame_height=frame_height,
            pose_backend=f"ultralytics:{backend_info.version}",
        )
        pose_observations.extend(association.observations)
        association_diagnostics.extend(association.diagnostics)
    identity_status = entity_summary.get("p2a_r1_identity_validation_status")
    evidence = build_interaction_evidence(
        event=event,
        event_window=window,
        observations=observations,
        pose_observations=tuple(pose_observations),
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
        association_diagnostics=tuple(association_diagnostics),
        identity_status=identity_status if isinstance(identity_status, str) else None,
    )
    input_paths: dict[str, Path] = {
        "video": video_path,
        "move_object_events": events_path,
        "move_object_evidence": event_evidence_path,
        "consolidated_memory": consolidated_memory,
        "entity_summary": entity_summary_path,
    }
    if config_path is not None:
        input_paths["config"] = config_path
    record = {
        **evidence,
        "source_move_object_event": event,
        "pose_inference_policy": "PERSISTENT_PERSON_ROI",
        "roi_margin": 0.0,
        "pose_backend": backend_info.to_record(),
        "pose_observation_provenance": {
            "git_head": _git_head(),
            "pose_model_not_retrained": True,
            "pose_inference_policy": "PERSISTENT_PERSON_ROI",
            "roi_margin": 0.0,
            "selected_frames": list(window.all_frames),
            "event_local_person_observation_count": len(event_local_person_observations),
            "raw_pose_detection_count": len(raw_pose_detections),
            "associated_pose_observation_count": len(pose_observations),
            "association_diagnostic_count": len(association_diagnostics),
            "pose_observations_are_same_frame_associated": True,
            "long_term_pose_identity_created": False,
        },
        "roi_attempts": _roi_attempts(
            person_observations=event_local_person_observations,
            pose_detections=raw_pose_detections,
            frame_width=frame_width,
            frame_height=frame_height,
            roi_margin=config.roi_margin,
        ),
        "pose_observations": [
            item.to_record()
            for item in sorted(pose_observations, key=lambda item: (item.frame_index, item.persistent_person_id))
        ],
        "association_diagnostics": sorted(
            association_diagnostics,
            key=lambda item: (item["frame_index"], item["detection_index"], item["status"]),
        ),
        "input_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(input_paths.items())
        },
        "configuration": {
            "device": device,
            "pose_confidence": pose_confidence,
            "pose_inference_policy": "PERSISTENT_PERSON_ROI",
            "roi_margin": 0.0,
            "evidence": config.to_record(),
            "event_local_policy": window.to_record(),
        },
        "runtime_flags": {
            "object_detector_rerun": False,
            "tracker_rerun": False,
            "pose_inference": True,
            "training": False,
            "fine_tuning": False,
            "vlm": False,
            "optical_flow": False,
            "move_object_modified": False,
            "person_ids_populated": False,
            "who_moved_object_implemented": False,
        },
    }
    output_path = output_dir / "person_object_interaction_evidence.json"
    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    record["output_artifact"] = {
        "path": str(output_path),
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "hash_scope": "serialized_record_before_output_artifact_metadata",
    }
    _write_json(output_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-evidence", type=Path, required=True)
    parser.add_argument("--consolidated-memory", type=Path, required=True)
    parser.add_argument("--entity-summary", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pose-confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        video_path=arguments.video,
        events_path=arguments.events,
        event_evidence_path=arguments.event_evidence,
        consolidated_memory=arguments.consolidated_memory,
        entity_summary_path=arguments.entity_summary,
        pose_model=arguments.pose_model,
        output_dir=arguments.output_dir,
        event_id=arguments.event_id,
        config_path=arguments.config,
        device=arguments.device,
        pose_confidence=arguments.pose_confidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
