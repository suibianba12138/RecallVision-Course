"""Generate evidence-bounded, agent-neutral P2D MOVE_OBJECT events."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recallvision.cli.p2c import _load_timelines
from recallvision.events.movement_evidence import (
    EVIDENCE_WINDOW_POLICY,
    MovementEvidenceConfig,
    MovementEvidenceEvaluator,
    MovementObservation,
    SUPPORT_HORIZON_SOURCE,
)
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
) -> dict[str, Any]:
    """Evaluate existing P2C events and write inspectable P2D artifacts."""

    config_record = _load_json_object(config_path, "P2D config")
    config = MovementEvidenceConfig.from_mapping(config_record)
    timelines = _load_timelines(timeline_path)
    events = _load_events(zone_events_path)
    observations = ObservationMemoryStore(consolidated_memory).load_all()
    entity_records = _load_json_list(persistent_entities, "persistent_entities")
    source_summary = _load_json_object(entity_summary, "entity_consolidation_summary")
    frame_width, frame_height = _frame_dimensions(source_summary)
    _validate_inputs(observations, entity_records, source_summary)

    movement_by_video: dict[str, list[MovementObservation]] = {}
    for observation in observations:
        if not observation.persistent_id:
            continue
        movement_by_video.setdefault(observation.video_id, []).append(
            MovementObservation(
                persistent_id=observation.persistent_id,
                frame_index=observation.frame_index,
                timestamp=observation.timestamp,
                zone=observation.zone,
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
            )
        )

    timelines_by_key = {(item.video_id, item.persistent_id): item for item in timelines}
    evaluator = MovementEvidenceEvaluator(config)
    accepted_events: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for trigger in events:
        persistent_id = trigger.object_ids[0] if len(trigger.object_ids) == 1 else None
        video_id = _event_video_id(trigger)
        timeline = timelines_by_key.get((video_id, persistent_id)) if persistent_id else None
        if timeline is None:
            diagnostic = _missing_timeline_diagnostic(trigger, config)
        else:
            result = evaluator.evaluate(
                timeline,
                movement_by_video.get(video_id, []),
                trigger,
            )
            diagnostic = {
                "event_id": result.event.event_id if result.event else trigger.event_id,
                "trigger_event_id": trigger.event_id,
                "reason": result.reason,
                **result.evidence,
            }
            if result.accepted and result.event is not None:
                accepted_events.append(result.event.to_record())
        diagnostics.append(diagnostic)

    accepted_events.sort(key=lambda item: str(item["event_id"]))
    diagnostics.sort(key=lambda item: str(item["event_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "move_object_events.json", accepted_events)
    _write_json(output_dir / "move_object_evidence.json", diagnostics)
    summary = _build_summary(
        config=config,
        config_path=config_path,
        timelines=timelines,
        triggers=events,
        accepted_events=accepted_events,
        diagnostics=diagnostics,
        inputs={
            "timeline": timeline_path,
            "zone_events": zone_events_path,
            "consolidated_memory": consolidated_memory,
            "persistent_entities": persistent_entities,
            "entity_summary": entity_summary,
            "config": config_path,
        },
    )
    _write_json(output_dir / "move_object_event_summary.json", summary)
    return summary


def _build_summary(
    *,
    config: MovementEvidenceConfig,
    config_path: Path,
    timelines: list[Any],
    triggers: list[EventRecord],
    accepted_events: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    inputs: dict[str, Path],
) -> dict[str, Any]:
    rejection_counts = Counter(
        item["reason"] for item in diagnostics if item.get("status") != "ACCEPTED"
    )
    return {
        "p2d_status": "PASS" if accepted_events else "NO_ACCEPTED_EVENTS",
        "event_type": EventType.MOVE_OBJECT.value,
        "config_file": config_path.name,
        "config": config.to_record(),
        "evidence_window_policy": EVIDENCE_WINDOW_POLICY,
        "support_horizon_seconds": config.max_transition_gap_seconds,
        "support_horizon_source": SUPPORT_HORIZON_SOURCE,
        "video_id": timelines[0].video_id if timelines else None,
        "timeline_count": len(timelines),
        "trigger_zone_change_event_count": len(triggers),
        "candidate_count": len(diagnostics),
        "move_object_event_count": len(accepted_events),
        "event_ids": [item["event_id"] for item in accepted_events],
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "input_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(inputs.items())
        },
        "events_are_derived": True,
        "raw_observations_mutated": False,
        "perception_rerun": False,
        "training_used": False,
        "vlm_used": False,
        "zone_change_alone_sufficient": False,
        "exact_trajectory_claimed": False,
        "exact_movement_time_claimed": False,
        "person_causality_inferred": False,
        "full_source_segments_used_for_cluster_gate": False,
        "outside_support_window_can_change_event": False,
        "event_end_extended_to_support_window": False,
    }


def _missing_timeline_diagnostic(
    trigger: EventRecord, config: MovementEvidenceConfig
) -> dict[str, Any]:
    return {
        "event_id": trigger.event_id,
        "trigger_event_id": trigger.event_id,
        "status": "REJECTED",
        "reason": "IDENTITY_MISMATCH",
        "evidence_window_policy": EVIDENCE_WINDOW_POLICY,
        "support_horizon_seconds": config.max_transition_gap_seconds,
        "support_horizon_source": SUPPORT_HORIZON_SOURCE,
        "left_support_frames": [],
        "right_support_frames": [],
        "left_support_time_range": None,
        "right_support_time_range": None,
        "left_cluster_center": None,
        "right_cluster_center": None,
        "left_cluster_radius": None,
        "right_cluster_radius": None,
        "cluster_separation": None,
        "cluster_nonoverlap_margin": None,
        "persistent_id": trigger.object_ids[0] if trigger.object_ids else None,
        "config": config.to_record(),
        "gates": {
            "temporal": {"passed": False},
            "identity": {"passed": False, "reason": "timeline not found"},
            "displacement": {"passed": False},
            "cluster_separation": {"passed": False},
            "sufficient_evidence": {"passed": False},
        },
        "local_variation": {"count": 0, "median": None, "p95": None, "max": None, "samples": []},
    }


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
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["p2d_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
