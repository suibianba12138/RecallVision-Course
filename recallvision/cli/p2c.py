"""Run P2C observed zone-transition events from P2B timeline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.events.zone_transitions import ZoneTransitionEventBuilder
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline


def run(
    *,
    timeline_path: Path,
    summary_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate P2C events without loading video or perception dependencies."""

    config = _load_config(config_path)
    summary = _load_object(summary_path, "P2B summary")
    _validate_p2b_summary(summary)
    timelines = _load_timelines(timeline_path)
    events = ZoneTransitionEventBuilder().build_all(timelines)

    output_dir.mkdir(parents=True, exist_ok=True)
    event_records = [event.to_record() for event in events]
    bottle_events = [
        event for event in events if event.object_ids and _class_from_id(event.object_ids[0]) == "bottle"
    ]
    _write_json(output_dir / "zone_change_events.json", event_records)
    _write_text(output_dir / "bottle_zone_change_events.txt", _bottle_audit(bottle_events))
    result = _build_summary(
        config=config,
        summary=summary,
        timeline_path=timeline_path,
        timelines=timelines,
        events=events,
        bottle_events=bottle_events,
    )
    _write_json(output_dir / "zone_change_event_summary.json", result)
    return result


def _build_summary(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    timeline_path: Path,
    timelines: list[ObjectStateTimeline],
    events: list[EventRecord],
    bottle_events: list[EventRecord],
) -> dict[str, Any]:
    return {
        "p2c_status": "PASS",
        "event_type": EventType.ZONE_CHANGE.value,
        "source": config["source"],
        "exact_transition_time_claimed": config["exact_transition_time_claimed"],
        "input_artifact": timeline_path.name,
        "input_artifact_sha256": _sha256(timeline_path),
        "input_p2b_status": summary["p2b_status"],
        "video_id": timelines[0].video_id if timelines else None,
        "timeline_count": len(timelines),
        "event_count": len(events),
        "bottle_event_count": len(bottle_events),
        "event_ids": [event.event_id for event in events],
        "events_are_derived": True,
        "raw_observations_mutated": False,
        "perception_rerun": False,
        "training_used": False,
        "vlm_used": False,
        "move_object_inferred": False,
    }


def _load_timelines(path: Path) -> list[ObjectStateTimeline]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("P2B timeline artifact must be a JSON list")
    timelines = [_timeline_from_record(record) for record in records]
    return sorted(timelines, key=lambda item: (item.video_id, item.persistent_id))


def _timeline_from_record(record: Any) -> ObjectStateTimeline:
    if not isinstance(record, dict):
        raise ValueError("timeline record must be a JSON object")
    segments = record.get("segments")
    if not isinstance(segments, list):
        raise ValueError("timeline segments must be a JSON list")
    return ObjectStateTimeline(
        persistent_id=_required_text(record, "persistent_id"),
        class_name=_required_text(record, "class_name"),
        video_id=_required_text(record, "video_id"),
        segments=tuple(_segment_from_record(item) for item in segments),
    )


def _segment_from_record(record: Any) -> ObjectStateRecord:
    if not isinstance(record, dict):
        raise ValueError("state segment must be a JSON object")
    return ObjectStateRecord(
        persistent_id=_required_text(record, "persistent_id"),
        class_name=_required_text(record, "class_name"),
        video_id=_required_text(record, "video_id"),
        start_frame=_required_int(record, "start_frame"),
        end_frame=_required_int(record, "end_frame"),
        start_time=_required_number(record, "start_time"),
        end_time=_required_number(record, "end_time"),
        zone=record.get("zone"),
        observation_count=_required_int(record, "observation_count"),
        first_evidence=_evidence_from_record(record.get("first_evidence"), "first_evidence"),
        last_evidence=_evidence_from_record(record.get("last_evidence"), "last_evidence"),
        source_track_ids=tuple(_required_string_list(record, "source_track_ids")),
        source_frame_indices=tuple(_required_int_list(record, "source_frame_indices")),
        last_confidence=record.get("last_confidence"),
    )


def _evidence_from_record(record: Any, label: str) -> EvidenceRef:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    frames = record.get("frame_indices")
    if not isinstance(frames, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in frames):
        raise ValueError(f"{label}.frame_indices must be a list of integers")
    timestamp = record.get("timestamp")
    if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))):
        raise ValueError(f"{label}.timestamp must be numeric or null")
    return EvidenceRef(
        video_id=_required_text(record, "video_id"),
        timestamp=None if timestamp is None else float(timestamp),
        start_time=_optional_number(record, "start_time"),
        end_time=_optional_number(record, "end_time"),
        frame_indices=frames,
        segment_ref=_optional_text(record, "segment_ref"),
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_object(path, "P2C config")
    if config.get("event_type") != EventType.ZONE_CHANGE.value:
        raise ValueError("P2C config event_type must be ZONE_CHANGE")
    if config.get("source") != "p2b_object_state_timelines":
        raise ValueError("P2C config source must be p2b_object_state_timelines")
    if config.get("exact_transition_time_claimed") is not False:
        raise ValueError("P2C must not claim an exact transition time")
    return config


def _validate_p2b_summary(summary: dict[str, Any]) -> None:
    if summary.get("p2b_status") != "PASS":
        raise ValueError("P2B summary must report PASS")
    if summary.get("perception_rerun") is not False:
        raise ValueError("P2B summary reports an unexpected perception rerun")


def _bottle_audit(events: list[EventRecord]) -> str:
    lines = [
        "P2C observed bottle zone-transition events",
        "ZONE_CHANGE is an observed state transition; it is not MOVE_OBJECT.",
        "Transition time is bounded by supporting observations, not an exact instant.",
    ]
    for event in events:
        lines.append(
            f"event_id={event.event_id},persistent_id={event.object_ids[0]},"
            f"from_zone={event.zone_before},to_zone={event.zone_after},"
            f"from_last_observed_frame={event.evidence[0].frame_indices[0]},"
            f"to_first_observed_frame={event.evidence[1].frame_indices[0]},"
            f"from_last_observed_timestamp={event.start_time},"
            f"to_first_observed_timestamp={event.end_time},"
            f"source_segment_ids={'|'.join(event.source_segment_ids)}"
        )
    return "\n".join(lines) + "\n"


def _class_from_id(persistent_id: str) -> str:
    parts = persistent_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "obj" else ""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _required_text(record: dict[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_int(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_number(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _optional_number(record: dict[str, Any], name: str) -> float | None:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    return float(value)


def _optional_text(record: dict[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _required_string_list(record: dict[str, Any], name: str) -> list[str]:
    value = record.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return value


def _required_int_list(record: dict[str, Any], name: str) -> list[int]:
    value = record.get(name)
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must be a list of integers")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        timeline_path=arguments.timeline,
        summary_path=arguments.summary,
        config_path=arguments.config,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["p2c_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
