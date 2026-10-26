"""A transparent JSONL store for P1A observation memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recallvision.memory.schema import BoundingBox, Observation


class MemoryFormatError(ValueError):
    """Raised when an observation JSONL record is malformed."""


class ObservationMemoryStore:
    """Append and load ``Observation`` records from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, observation: Observation) -> None:
        """Append one observation as one stable JSON object line."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_observation_to_record(observation), sort_keys=True))
            handle.write("\n")

    def load_all(self) -> list[Observation]:
        """Load all records in file order and fail on malformed lines."""

        if not self.path.exists():
            return []

        observations: list[Observation] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MemoryFormatError(
                        f"Malformed JSONL at {self.path}: line {line_number} is empty"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MemoryFormatError(
                        f"Malformed JSONL at {self.path}: line {line_number}: {error.msg}"
                    ) from error
                try:
                    observations.append(_record_to_observation(record))
                except (KeyError, TypeError, ValueError) as error:
                    raise MemoryFormatError(
                        f"Malformed JSONL at {self.path}: line {line_number}: {error}"
                    ) from error
        return observations

    def find_by_class(self, class_name: str) -> list[Observation]:
        """Return observations whose class matches, preserving file order."""

        return [
            observation
            for observation in self.load_all()
            if observation.class_name == class_name
        ]


def _observation_to_record(observation: Observation) -> dict[str, Any]:
    return {
        "video_id": observation.video_id,
        "frame_index": observation.frame_index,
        "timestamp": observation.timestamp,
        "track_id": observation.track_id,
        "persistent_id": observation.persistent_id,
        "entity_type": observation.entity_type,
        "class_name": observation.class_name,
        "bbox": {
            "x1": observation.bbox.x1,
            "y1": observation.bbox.y1,
            "x2": observation.bbox.x2,
            "y2": observation.bbox.y2,
        },
        "confidence": observation.confidence,
        "zone": observation.zone,
        "source_track_ids": list(observation.source_track_ids),
    }


def _record_to_observation(record: Any) -> Observation:
    if not isinstance(record, dict):
        raise TypeError("record must be a JSON object")
    bbox_record = record["bbox"]
    if not isinstance(bbox_record, dict):
        raise TypeError("bbox must be a JSON object")
    return Observation(
        video_id=_required_text(record, "video_id"),
        frame_index=_required_int(record, "frame_index"),
        timestamp=_required_number(record, "timestamp"),
        track_id=_required_text(record, "track_id"),
        persistent_id=_optional_text(record, "persistent_id"),
        entity_type=_required_text(record, "entity_type"),
        class_name=_required_text(record, "class_name"),
        bbox=BoundingBox(
            x1=_required_number(bbox_record, "x1"),
            y1=_required_number(bbox_record, "y1"),
            x2=_required_number(bbox_record, "x2"),
            y2=_required_number(bbox_record, "y2"),
        ),
        confidence=_optional_number(record, "confidence"),
        zone=_optional_text(record, "zone"),
        source_track_ids=_source_track_ids(record),
    )


def _required_text(record: dict[str, Any], field_name: str) -> str:
    value = record[field_name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_text(record: dict[str, Any], field_name: str) -> str | None:
    value = record[field_name]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return value


def _required_int(record: dict[str, Any], field_name: str) -> int:
    value = record[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _required_number(record: dict[str, Any], field_name: str) -> float:
    value = record[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _optional_number(record: dict[str, Any], field_name: str) -> float | None:
    value = record[field_name]
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{field_name} must be numeric or null")
    return None if value is None else float(value)


def _source_track_ids(record: dict[str, Any]) -> tuple[str, ...]:
    """Load optional R1 duplicate provenance while accepting older P1 records."""

    value = record.get("source_track_ids")
    if value is None:
        return (record["track_id"],)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("source_track_ids must be a list of non-empty strings")
    return tuple(value)
