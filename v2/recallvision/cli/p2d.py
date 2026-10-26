"""Generate evidence-bounded, agent-neutral P2D MOVE_OBJECT events."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from recallvision.cli.p2c import _load_timelines
from recallvision.events.motion_candidates import (
    MotionCandidateConfig,
    RobustMotionEventDetector,
)
from recallvision.events.movement_evidence import MovementObservation
from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.store import ObservationMemoryStore


def run(
    *,
    timeline_path: Path,
    zone_events_path: Path,
    consolidated_memory: Path,
    persistent_entities: Path,
    entity_summary: Path,
    config_path: Path,
    output_dir: Path,
    motion_observations_path: Path | None = None,
    effective_identity_mapping_path: Path | None = None,
    motion_config_path: Path | None = None,
    shot_boundaries_path: Path | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Generate MOVE_OBJECTs from robust windows and frozen observations.

    P2C zone transitions remain candidate sources. Track-level observations are
    preferred for the new same-frame duplicate consensus; older callers may
    omit them and use their consolidated memory as the available source.
    """

    timelines = _load_timelines(timeline_path)
    zone_events = _load_events(zone_events_path)
    consolidated_observations = ObservationMemoryStore(consolidated_memory).load_all()
    entity_records = _load_json_list(persistent_entities, "persistent_entities")
    source_summary = _load_json_object(entity_summary, "entity_consolidation_summary")
    frame_width, frame_height = _frame_dimensions(source_summary)
    _validate_inputs(consolidated_observations, entity_records, source_summary)

    observation_path = motion_observations_path or consolidated_memory
    motion_observations = ObservationMemoryStore(observation_path).load_all()
    effective_mapping = _load_effective_mapping(effective_identity_mapping_path, source_summary)
    if effective_mapping:
        remapped_observations = []
        for observation in motion_observations:
            source_id = observation.persistent_id
            if source_id is None or source_id not in effective_mapping:
                raise ValueError(f"raw observation has no effective identity mapping: {source_id}")
            remapped_observations.append(
                replace(observation, persistent_id=effective_mapping[source_id])
            )
        motion_observations = remapped_observations
    _validate_inputs(motion_observations, entity_records, source_summary)

    source_video_id = source_summary.get("video_id")
    if video_path is not None:
        if not video_path.is_file():
            raise FileNotFoundError(f"P2D source video does not exist: {video_path}")
        expected_video_sha = source_summary.get("video_sha256")
        if expected_video_sha and _sha256(video_path) != expected_video_sha:
            raise ValueError("P2D source video SHA-256 does not match the entity summary")
    video_ids = {item.video_id for item in motion_observations}
    if source_video_id is None:
        source_video_id = next(iter(video_ids), None)
    if video_ids and video_ids != {source_video_id}:
        raise ValueError("motion observations do not belong to the entity summary video")

    movement_observations = [
        MovementObservation(
            persistent_id=observation.persistent_id,
            frame_index=observation.frame_index,
            timestamp=observation.timestamp,
            zone=observation.zone or "unknown",
            center_normalized=(
                ((observation.bbox.x1 + observation.bbox.x2) / 2.0) / frame_width,
                ((observation.bbox.y1 + observation.bbox.y2) / 2.0) / frame_height,
            ),
            source_track_ids=observation.source_track_ids or (observation.track_id,),
            track_id=observation.track_id,
            bbox=(
                observation.bbox.x1,
                observation.bbox.y1,
                observation.bbox.x2,
                observation.bbox.y2,
            ),
            confidence=observation.confidence,
            bbox_size_normalized=(
                (observation.bbox.x2 - observation.bbox.x1) / frame_width,
                (observation.bbox.y2 - observation.bbox.y1) / frame_height,
            ),
        )
        for observation in motion_observations
        if observation.persistent_id is not None
    ]
    identity_provenance = _identity_provenance_records(entity_records, source_summary)
    config_file = motion_config_path or (
        Path(__file__).resolve().parents[2] / "configs/p3b1_robust_motion.yaml"
    )
    if config_file.is_file():
        config_record = _load_json_or_yaml(config_file)
        config = MotionCandidateConfig.from_mapping(config_record)
    else:
        config = MotionCandidateConfig()
    shot_boundaries = _load_shot_boundaries(shot_boundaries_path)
    detector = RobustMotionEventDetector(config)
    result = detector.evaluate(
        movement_observations,
        video_id=str(source_video_id or _first_video_id(zone_events)),
        zone_events=zone_events,
        identity_provenance=identity_provenance,
        shot_boundary_frames=shot_boundaries,
        video_path=video_path,
    )

    accepted_events = result["events"]
    diagnostics = result["diagnostics"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "move_object_events.json", accepted_events)
    _write_json(output_dir / "move_object_evidence.json", diagnostics)
    input_paths: dict[str, Path] = {
        "timeline": timeline_path,
        "zone_events": zone_events_path,
        "consolidated_memory": consolidated_memory,
        "motion_observations": observation_path,
        "persistent_entities": persistent_entities,
        "entity_summary": entity_summary,
        "legacy_p2d_config": config_path,
        "motion_config": config_file,
    }
    if effective_identity_mapping_path is not None:
        input_paths["effective_identity_mapping"] = effective_identity_mapping_path
    if shot_boundaries_path is not None:
        input_paths["shot_boundaries"] = shot_boundaries_path
    if video_path is not None:
        input_paths["source_video"] = video_path
    summary = _build_summary(
        timelines=timelines,
        zone_events=zone_events,
        accepted_events=accepted_events,
        diagnostics=diagnostics,
        detector_summary=result["summary"],
        config=config,
        inputs=input_paths,
    )
    _write_json(output_dir / "move_object_event_summary.json", summary)
    return summary


def _build_summary(
    *,
    timelines: list[Any],
    zone_events: list[EventRecord],
    accepted_events: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    detector_summary: dict[str, Any],
    config: MotionCandidateConfig,
    inputs: dict[str, Path],
) -> dict[str, Any]:
    reason_counts = Counter(
        item["reason"] for item in diagnostics if item.get("status") != "ACCEPTED"
    )
    return {
        "p2d_status": "PASS" if accepted_events else "NO_ACCEPTED_EVENTS",
        "algorithm": "P3B1_ROBUST_MOTION",
        "event_type": EventType.MOVE_OBJECT.value,
        "config": config.to_record(),
        "video_id": timelines[0].video_id if timelines else None,
        "timeline_count": len(timelines),
        "trigger_zone_change_event_count": len(zone_events),
        "candidate_count": len(diagnostics),
        "continuous_candidate_count": detector_summary["continuous_candidate_count"],
        "zone_candidate_count": detector_summary["zone_trigger_candidate_count"],
        "move_object_event_count": len(accepted_events),
        "event_ids": [item["event_id"] for item in accepted_events],
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "insufficient_evidence_count": detector_summary["insufficient_evidence_count"],
        "duplicate_event_count": detector_summary["duplicate_event_count"],
        "same_zone_accepted_event_count": detector_summary["same_zone_accepted_event_count"],
        "input_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(inputs.items()) if path.is_file()
        },
        "events_are_derived": True,
        "raw_observations_mutated": False,
        "perception_rerun": False,
        "training_used": False,
        "vlm_used": False,
        "missing_observations_interpolated": False,
        "zone_change_is_optional_candidate_source": True,
        "automatic_identity_review_promoted": False,
        "shot_boundary_policy": config.to_record()["shot_boundary_policy"],
    }


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required to read the motion config") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("motion config must be an object")
    return value


def _load_effective_mapping(
    path: Path | None, source_summary: dict[str, Any]
) -> dict[str, str]:
    if path is None:
        return {}
    record = _load_json_object(path, "effective_entity_mapping")
    if record.get("source_video_sha256") not in (None, source_summary.get("video_sha256")):
        raise ValueError("effective identity mapping belongs to another source video")
    mapping = record.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("effective identity mapping must contain a mapping object")
    result: dict[str, str] = {}
    for source_id, item in mapping.items():
        if not isinstance(source_id, str) or not isinstance(item, dict):
            raise ValueError("effective identity mapping contains a malformed entry")
        effective_id = item.get("effective_entity_id")
        if not isinstance(effective_id, str) or not effective_id:
            raise ValueError("effective identity mapping contains an invalid entity ID")
        result[source_id] = effective_id
    return result


def _identity_provenance_records(
    entities: list[dict[str, Any]], source_summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    review_status = source_summary.get("p2a_r1_identity_validation_status", "NOT_RECORDED")
    result = {}
    for item in entities:
        persistent_id = item.get("persistent_id")
        if not isinstance(persistent_id, str):
            continue
        result[persistent_id] = {
            "identity_provenance": item.get("identity_provenance", "AUTOMATIC_SOURCE"),
            "source_identity_review_status": item.get("source_identity_review_status", review_status),
            "source_entity_ids": item.get("source_persistent_ids", [persistent_id]),
            "human_confirmation_operation_ids": item.get("identity_confirmation_operation_ids", []),
            "overlay_revision": item.get("overlay_revision"),
        }
    return result


def _load_shot_boundaries(path: Path | None) -> set[int]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("frame_indices")
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("shot boundaries must be a list of non-negative frame indices")
    return set(value)


def _first_video_id(events: list[EventRecord]) -> str:
    for event in events:
        video_id = _event_video_id(event)
        if video_id:
            return video_id
    return "unknown-video"

def _load_events(path: Path) -> list[EventRecord]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("P2C zone events must be a JSON list")
    return sorted((_event_from_record(item) for item in records), key=lambda item: item.event_id)


def _event_from_record(record: Any) -> EventRecord:
    if not isinstance(record, dict):
        raise ValueError("event record must be a JSON object")
    try:
        event_type = EventType(record["event_type"])
    except (KeyError, ValueError) as error:
        raise ValueError("event record has an invalid event_type") from error
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("event evidence must be a JSON list")
    motion_keys = (
        "candidate_type",
        "effective_entity_id",
        "position_before",
        "position_after",
        "displacement",
        "relative_displacement_vector_normalized",
        "trigger_provenance",
        "evidence_status",
        "identity_provenance",
        "observation_references",
        "scene_continuity_status",
    )
    motion_metadata = {key: record[key] for key in motion_keys if key in record}
    return EventRecord(
        event_id=_required_text(record, "event_id"),
        event_type=event_type,
        start_time=_required_number(record, "start_time"),
        end_time=_required_number(record, "end_time"),
        person_ids=_required_string_list(record, "person_ids"),
        object_ids=_required_string_list(record, "object_ids"),
        zone_before=record.get("zone_before"),
        zone_after=record.get("zone_after"),
        evidence=[_evidence_from_record(item) for item in evidence],
        source_segment_ids=tuple(_required_string_list(record, "source_segment_ids")),
        motion_metadata=motion_metadata or None,
    )


def _evidence_from_record(record: Any) -> EvidenceRef:
    if not isinstance(record, dict):
        raise ValueError("event evidence item must be a JSON object")
    frames = record.get("frame_indices")
    if not isinstance(frames, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in frames):
        raise ValueError("event evidence frame_indices must be a list of integers")
    timestamp = record.get("timestamp")
    if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))):
        raise ValueError("event evidence timestamp must be numeric or null")
    return EvidenceRef(
        video_id=_required_text(record, "video_id"),
        timestamp=None if timestamp is None else float(timestamp),
        start_time=_optional_number(record.get("start_time"), "start_time"),
        end_time=_optional_number(record.get("end_time"), "end_time"),
        frame_indices=frames,
        segment_ref=record.get("segment_ref"),
    )


def _event_video_id(event: EventRecord) -> str | None:
    if event.evidence:
        return event.evidence[0].video_id
    if event.source_segment_ids:
        return event.source_segment_ids[0].split("#", 1)[0]
    return None


def _frame_dimensions(summary: dict[str, Any]) -> tuple[float, float]:
    metadata = summary.get("video_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("entity summary is missing video_metadata")
    width = metadata.get("width")
    height = metadata.get("height")
    if isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0:
        raise ValueError("video_metadata.width must be positive")
    if isinstance(height, bool) or not isinstance(height, (int, float)) or height <= 0:
        raise ValueError("video_metadata.height must be positive")
    return float(width), float(height)


def _validate_inputs(
    observations: list[Any], entity_records: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    if summary.get("p2a_r1_pipeline_status") != "PASS":
        raise ValueError("R1 summary does not report pipeline PASS")
    entity_ids = {
        record.get("persistent_id")
        for record in entity_records
        if isinstance(record.get("persistent_id"), str)
    }
    observed_ids = {
        observation.persistent_id
        for observation in observations
        if observation.persistent_id is not None
    }
    if not observed_ids.issubset(entity_ids):
        raise ValueError("consolidated memory contains IDs absent from persistent_entities.json")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def _required_text(record: dict[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_number(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    return float(value)


def _required_string_list(record: dict[str, Any], name: str) -> list[str]:
    value = record.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--zone-events", type=Path, required=True)
    parser.add_argument("--consolidated-memory", type=Path, required=True)
    parser.add_argument("--persistent-entities", type=Path, required=True)
    parser.add_argument("--entity-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion-config", type=Path)
    parser.add_argument("--motion-observations", type=Path)
    parser.add_argument("--effective-identity-mapping", type=Path)
    parser.add_argument("--shot-boundaries", type=Path)
    parser.add_argument("--video", type=Path, help="Source video for sparse camera-motion evidence")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        timeline_path=arguments.timeline,
        zone_events_path=arguments.zone_events,
        consolidated_memory=arguments.consolidated_memory,
        persistent_entities=arguments.persistent_entities,
        entity_summary=arguments.entity_summary,
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        motion_observations_path=arguments.motion_observations,
        effective_identity_mapping_path=arguments.effective_identity_mapping,
        motion_config_path=arguments.motion_config,
        shot_boundaries_path=arguments.shot_boundaries,
        video_path=arguments.video,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["p2d_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
