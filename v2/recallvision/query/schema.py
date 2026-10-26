"""Stable contracts for supported V1 questions and answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from recallvision.memory.schema import EvidenceRef


class QueryType(str, Enum):
    """The four supported V1 query types."""

    WHERE_OBJECT = "WHERE_OBJECT"
    WHERE_OBJECT_BEFORE = "WHERE_OBJECT_BEFORE"
    WHEN_OBJECT_MOVED = "WHEN_OBJECT_MOVED"
    WHO_MOVED_OBJECT = "WHO_MOVED_OBJECT"


@dataclass(frozen=True)
class TargetReference:
    """Query-derived target semantics, deliberately independent of GT IDs."""

    raw_phrase: str
    canonical_label: str
    semantic_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.raw_phrase, str):
            raise TypeError("raw_phrase must be a string")
        if not isinstance(self.canonical_label, str):
            raise TypeError("canonical_label must be a string")
        if not self.canonical_label and self.raw_phrase:
            raise ValueError("canonical_label must be non-empty for a target phrase")
        if any(not isinstance(item, str) or not item for item in self.semantic_aliases):
            raise ValueError("semantic_aliases must contain non-empty strings")


class AnswerStatus(str, Enum):
    """Explicit result states for deterministic query handlers."""

    OK = "OK"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    NO_REFERENCE_EVENT = "NO_REFERENCE_EVENT"
    AMBIGUOUS_MULTIPLE_EVENT_REFERENCE = "AMBIGUOUS_MULTIPLE_EVENT_REFERENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class Query:
    """A structured representation of a natural-language question."""

    question_id: str
    video_id: str
    question: str
    query_type: QueryType
    target_entity_ids: list[str] = field(default_factory=list)
    target_class: str | None = None
    target_reference: TargetReference | None = None
    reference_event_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the optional event-reference field without coupling handlers."""

        if isinstance(self.target_reference, dict):
            object.__setattr__(self, "target_reference", TargetReference(**self.target_reference))
        elif self.target_reference is not None and not isinstance(self.target_reference, TargetReference):
            raise TypeError("target_reference must be a TargetReference or None")

        if self.reference_event_id is None:
            return
        if not isinstance(self.reference_event_id, str):
            raise TypeError("reference_event_id must be a string or None")
        if not self.reference_event_id.strip():
            raise ValueError("reference_event_id must be non-empty when provided")


@dataclass(frozen=True)
class Answer:
    """An answer paired with temporally or visually localized evidence."""

    question_id: str
    answer: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    status: AnswerStatus = AnswerStatus.OK
    target_entity_id: str | None = None
    target_class: str | None = None
    zone: str | None = None
    timestamp: float | None = None
    frame_index: int | None = None
    query_type: QueryType | None = None
    reference_event_id: str | None = None
    from_source_segment_id: str | None = None
    target_resolution_status: str | None = None
    zone_after: str | None = None
    position_before_normalized: tuple[float, float] | None = None
    position_after_normalized: tuple[float, float] | None = None
    stable_position_before_normalized: tuple[float, float] | None = None
    stable_position_after_normalized: tuple[float, float] | None = None
    displacement_normalized: float | None = None
    position_coordinate_space: str | None = None
    position_after_timestamp: float | None = None
    position_after_frame_index: int | None = None
    identity_provenance: dict[str, object] | None = None
    overlay_revision: int | str | None = None

    def to_record(self) -> dict[str, object]:
        """Serialize an answer while retaining query and provenance metadata."""

        record: dict[str, object] = {
            "question_id": self.question_id,
            "query_type": self.query_type.value if self.query_type is not None else None,
            "answer": self.answer,
            "status": self.status.value,
            "target_entity_id": self.target_entity_id,
            "target_class": self.target_class,
            "persistent_id": self.target_entity_id,
            "class_name": self.target_class,
            "zone": self.zone,
            "from_zone": self.zone,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "last_observed_timestamp": self.timestamp,
            "last_observed_frame": self.frame_index,
            "reference_event_id": self.reference_event_id,
            "from_source_segment_id": self.from_source_segment_id,
            "from_segment_id": self.from_source_segment_id,
            "target_resolution_status": self.target_resolution_status,
            "evidence": [_evidence_to_record(item) for item in self.evidence],
        }
        optional_fields = {
            "zone_after": self.zone_after,
            "position_before_normalized": list(self.position_before_normalized) if self.position_before_normalized is not None else None,
            "position_after_normalized": list(self.position_after_normalized) if self.position_after_normalized is not None else None,
            "stable_position_before_normalized": list(self.stable_position_before_normalized) if self.stable_position_before_normalized is not None else None,
            "stable_position_after_normalized": list(self.stable_position_after_normalized) if self.stable_position_after_normalized is not None else None,
            "displacement_normalized": self.displacement_normalized,
            "position_coordinate_space": self.position_coordinate_space,
            "position_after_timestamp": self.position_after_timestamp,
            "position_after_frame_index": self.position_after_frame_index,
            "identity_provenance": self.identity_provenance,
            "overlay_revision": self.overlay_revision,
        }
        record.update({key: value for key, value in optional_fields.items() if value is not None})
        return record


def _evidence_to_record(evidence: EvidenceRef) -> dict[str, object]:
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }
