"""Stable contracts for events extracted from visual memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from recallvision.memory.schema import EvidenceRef


class EventType(str, Enum):
    """The V1 event taxonomy."""

    PERSON_ENTER = "PERSON_ENTER"
    PERSON_EXIT = "PERSON_EXIT"
    OBJECT_APPEAR = "OBJECT_APPEAR"
    OBJECT_DISAPPEAR = "OBJECT_DISAPPEAR"
    ZONE_CHANGE = "ZONE_CHANGE"
    MOVE_OBJECT = "MOVE_OBJECT"
    PICK_UP = "PICK_UP"
    PUT_DOWN = "PUT_DOWN"


@dataclass
class EventRecord:
    """A temporally localized event involving people and/or objects."""

    event_id: str
    event_type: EventType
    start_time: float
    end_time: float
    person_ids: list[str] = field(default_factory=list)
    object_ids: list[str] = field(default_factory=list)
    zone_before: str | None = None
    zone_after: str | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    source_segment_ids: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        """Serialize the stable event contract without detector-specific state."""

        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "person_ids": list(self.person_ids),
            "object_ids": list(self.object_ids),
            "zone_before": self.zone_before,
            "zone_after": self.zone_after,
            "evidence": [
                {
                    "video_id": item.video_id,
                    "timestamp": item.timestamp,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "frame_indices": list(item.frame_indices),
                    "segment_ref": item.segment_ref,
                }
                for item in self.evidence
            ],
            "source_segment_ids": list(self.source_segment_ids),
        }
