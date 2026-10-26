"""Read-only P2D EventRecord loading for F3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef

from .errors import VlmF3InputError


def load_event(path: str | Path, event_id: str) -> EventRecord:
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VlmF3InputError(f"unable to load source MOVE_OBJECT artifact: {source}: {error}") from error
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise VlmF3InputError("source event artifact must be a JSON list of objects")
    matches = [item for item in records if item.get("event_id") == event_id]
    if len(matches) != 1:
        raise VlmF3InputError(f"source event artifact must contain exactly one event: {event_id}")
    return _event_from_record(matches[0])


def event_video_id(event: EventRecord) -> str | None:
    if event.evidence:
        return event.evidence[0].video_id
    if event.source_segment_ids:
        return event.source_segment_ids[0].split("#", 1)[0]
    return None


def _event_from_record(record: dict[str, Any]) -> EventRecord:
    try:
        event_type = EventType(record["event_type"])
        evidence = [_evidence_from_record(item) for item in record["evidence"]]
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
        raise VlmF3InputError(f"malformed source EventRecord: {error}") from error


def _evidence_from_record(record: Any) -> EvidenceRef:
    if not isinstance(record, dict):
        raise ValueError("event evidence item must be an object")
    frames = record.get("frame_indices")
    if not isinstance(frames, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in frames
    ):
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
