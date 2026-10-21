"""Run the artifact-only P3A WHEN_OBJECT_MOVED query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recallvision.cli.p2c import _load_timelines
from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.query.object_movement import (
    MovementAnswerStatus,
    answer_when_object_moved,
)
from recallvision.query.parser import parse_query


def run(
    *,
    events_path: Path,
    timelines_path: Path,
    question: str,
    output_dir: Path,
    video_id: str | None = None,
    question_id: str = "p3a-query",
) -> dict[str, Any]:
    """Answer one movement query from frozen P2B/P2D artifacts."""

    timelines = _load_timelines(timelines_path)
    if not timelines:
        raise ValueError("P2B timeline artifact is empty")
    events = _load_events(events_path)
    timeline_video_ids = sorted({timeline.video_id for timeline in timelines})
    selected_video_id = video_id or _single_video_id(timeline_video_ids)
    query = parse_query(
        question,
        video_id=selected_video_id,
        question_id=question_id,
    )
    result = answer_when_object_moved(query, timelines, events)
    record = result.to_record()
    record["input_artifacts"] = {
        "events": {"path": str(events_path), "sha256": _sha256(events_path)},
        "timelines": {"path": str(timelines_path), "sha256": _sha256(timelines_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "query_result.json", record)
    return record


def _single_video_id(video_ids: list[str]) -> str:
    if len(video_ids) != 1:
        raise ValueError("P3A requires one video ID or a single-video timeline artifact")
    return video_ids[0]


def _load_events(path: Path) -> list[EventRecord]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("P2D MOVE_OBJECT artifact must be a JSON list")
    return [_event_from_record(record) for record in records]


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
    if not isinstance(frames, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in frames
    ):
        raise ValueError("event evidence frame_indices must be a list of integers")
    timestamp = record.get("timestamp")
    if timestamp is not None and (
        isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
    ):
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


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id")
    arguments = parser.parse_args()
    result = run(
        events_path=arguments.events,
        timelines_path=arguments.timelines,
        question=arguments.question,
        output_dir=arguments.output_dir,
        video_id=arguments.video_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        MovementAnswerStatus.MOVEMENT_FOUND.value,
        MovementAnswerStatus.NO_SUPPORTED_EVENT.value,
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
