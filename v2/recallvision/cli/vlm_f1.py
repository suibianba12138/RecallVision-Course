"""Build a production-ready, provider-independent VLM-F1 evidence package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.store import ObservationMemoryStore
from recallvision.vlm_f1.builder import build_evidence_package
from recallvision.vlm_f1.errors import VlmF1InputError
from recallvision.vlm_f1.frames import probe_video
from recallvision.vlm_f1.hashing import package_sha256


def run(
    *,
    events_path: Path,
    event_evidence_path: Path,
    consolidated_memory_path: Path,
    persistent_entities_path: Path,
    video_path: Path,
    output_root: Path,
    event_id: str | None = None,
    repository_head: str | None = None,
) -> dict[str, Any]:
    """Build one event package from existing structured artifacts only."""

    events = _load_events(events_path)
    selected = _select_event(events, event_id)
    diagnostics = _load_json_list(event_evidence_path, "event evidence")
    diagnostic = next((item for item in diagnostics if item.get("event_id") == selected.event_id), None)
    if diagnostic is None:
        raise VlmF1InputError(f"no diagnostic found for event: {selected.event_id}")
    observations = ObservationMemoryStore(consolidated_memory_path).load_all()
    entities = _load_json_list(persistent_entities_path, "persistent entities")
    persistent_ids = {
        item.get("persistent_id") for item in entities if isinstance(item.get("persistent_id"), str)
    }
    video_id = _event_video_id(selected)
    if video_id is None:
        raise VlmF1InputError("selected event has no video ID")
    package_dir = output_root / video_id / _safe_event_id(selected.event_id)
    head = repository_head or _git_head()
    package = build_evidence_package(
        event=selected,
        event_diagnostic=diagnostic,
        object_observations=observations,
        person_observations=observations,
        persistent_entity_ids=persistent_ids,
        video_path=video_path,
        output_dir=package_dir,
        repository_head=head,
        video_info=probe_video(video_path),
        input_artifacts={
            "events": events_path,
            "event_evidence": event_evidence_path,
            "consolidated_memory": consolidated_memory_path,
            "persistent_entities": persistent_entities_path,
        },
    )
    return {
        "package_path": str(package.root),
        "manifest_path": str(package.manifest_path),
        "package_hash": package_sha256(package.root),
        "candidate_mapping": package.manifest["candidate_mapping"],
        "move_object_event_id": selected.event_id,
        "vlm_inference": False,
        "person_ids_populated": False,
        "who_moved_object_implemented": False,
    }


def _load_events(path: Path) -> list[EventRecord]:
    records = _load_json_list(path, "MOVE_OBJECT events")
    return [_event_from_record(record) for record in records]


def _event_from_record(record: dict[str, Any]) -> EventRecord:
    try:
        event_type = EventType(record["event_type"])
        evidence = [
            EvidenceRef(
                video_id=_required_text(item, "video_id"),
                timestamp=_optional_number(item.get("timestamp"), "timestamp"),
                start_time=_optional_number(item.get("start_time"), "start_time"),
                end_time=_optional_number(item.get("end_time"), "end_time"),
                frame_indices=_required_int_list(item, "frame_indices"),
                segment_ref=item.get("segment_ref"),
            )
            for item in record["evidence"]
        ]
        return EventRecord(
            event_id=_required_text(record, "event_id"),
            event_type=event_type,
            start_time=_required_number(record, "start_time"),
            end_time=_required_number(record, "end_time"),
            person_ids=_required_string_list(record, "person_ids"),
            object_ids=_required_string_list(record, "object_ids"),
            zone_before=record.get("zone_before"),
            zone_after=record.get("zone_after"),
            evidence=evidence,
            source_segment_ids=tuple(_required_string_list(record, "source_segment_ids")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VlmF1InputError(f"malformed event record: {error}") from error


def _select_event(events: list[EventRecord], event_id: str | None) -> EventRecord:
    if event_id is not None:
        matches = [item for item in events if item.event_id == event_id]
    else:
        matches = events
    if len(matches) != 1:
        raise VlmF1InputError("CLI requires exactly one selected MOVE_OBJECT event")
    return matches[0]


def _event_video_id(event: EventRecord) -> str | None:
    if event.evidence:
        return event.evidence[0].video_id
    return None


def _safe_event_id(event_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", event_id).strip("_")
    if not safe:
        raise VlmF1InputError("event_id cannot produce a stable output directory")
    return safe


def _git_head() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise VlmF1InputError(f"unable to resolve repository HEAD: {error}") from error
    return result.stdout.strip()


def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VlmF1InputError(f"unable to load {label}: {path}: {error}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise VlmF1InputError(f"{label} must be a JSON list of objects")
    return value


def _required_text(record: dict[str, Any], name: str) -> str:
    value = record[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_number(record: dict[str, Any], name: str) -> float:
    value = record[name]
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
    value = record[name]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _required_int_list(record: dict[str, Any], name: str) -> list[int]:
    value = record[name]
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must be a list of integers")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-evidence", type=Path, required=True)
    parser.add_argument("--consolidated-memory", type=Path, required=True)
    parser.add_argument("--persistent-entities", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-id")
    parser.add_argument("--repository-head")
    arguments = parser.parse_args()
    result = run(
        events_path=arguments.events,
        event_evidence_path=arguments.event_evidence,
        consolidated_memory_path=arguments.consolidated_memory,
        persistent_entities_path=arguments.persistent_entities,
        video_path=arguments.video,
        output_root=arguments.output_root,
        event_id=arguments.event_id,
        repository_head=arguments.repository_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
