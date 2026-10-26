"""P2B last-seen WHERE_OBJECT retrieval over observed state timelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import (
    LastSeenState,
    ObjectStateTimeline,
    select_last_seen_state,
)
from recallvision.query.schema import Query, QueryType
from recallvision.query.target_resolution import (
    TargetResolutionStatus,
    candidates_from_timelines,
    resolve_query_target,
)


class StateAnswerStatus(str, Enum):
    """P2B result statuses, including explicit ambiguity and last-seen meaning."""

    TARGET_LAST_SEEN = "TARGET_LAST_SEEN"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


@dataclass(frozen=True)
class StateQueryResult:
    """Typed P2B answer with last-seen and staleness metadata."""

    question_id: str
    answer: str
    status: StateAnswerStatus
    persistent_id: str | None = None
    class_name: str | None = None
    zone: str | None = None
    last_seen_timestamp: float | None = None
    frame_index: int | None = None
    confidence: float | None = None
    video_end_timestamp: float | None = None
    staleness_seconds: float | None = None
    evidence_ref: EvidenceRef | None = None
    target_resolution_status: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "status": self.status.value,
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "zone": self.zone,
            "last_seen_timestamp": self.last_seen_timestamp,
            "frame_index": self.frame_index,
            "confidence": self.confidence,
            "video_end_timestamp": self.video_end_timestamp,
            "staleness_seconds": self.staleness_seconds,
            "evidence_ref": _evidence_to_record(self.evidence_ref),
            "target_resolution_status": self.target_resolution_status,
        }


def answer_where_object_state(
    query: Query,
    timelines: Sequence[ObjectStateTimeline],
    *,
    video_end_timestamp: float,
) -> StateQueryResult:
    """Answer only ``WHERE_OBJECT`` using persistent timelines, never raw observations."""

    if query.query_type is not QueryType.WHERE_OBJECT:
        return StateQueryResult(
            question_id=query.question_id,
            answer=f"UNSUPPORTED_QUERY: {query.query_type.value}",
            status=StateAnswerStatus.UNSUPPORTED_QUERY,
        )

    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    if resolution.status is not TargetResolutionStatus.RESOLVED:
        label = resolution.reference.canonical_label or ", ".join(query.target_entity_ids) or "target"
        status = (
            StateAnswerStatus.AMBIGUOUS_TARGET
            if resolution.status is TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES
            else StateAnswerStatus.TARGET_NOT_FOUND
        )
        message = (
            f"AMBIGUOUS_TARGET: multiple persistent entities match {label}."
            if status is StateAnswerStatus.AMBIGUOUS_TARGET
            else f"TARGET_NOT_FOUND: no observed state for {label}."
        )
        return StateQueryResult(
            question_id=query.question_id,
            answer=message,
            status=status,
            target_resolution_status=resolution.status.value,
        )

    candidate_id = resolution.resolved_entity_id
    assert candidate_id is not None
    target = next(
        timeline for timeline in timelines if timeline.persistent_id == candidate_id and timeline.video_id == query.video_id
    )
    last_seen = select_last_seen_state(
        target,
        video_end_timestamp=video_end_timestamp,
    )
    return _result_from_last_seen(
        query,
        last_seen,
        target_class=resolution.reference.canonical_label or target.class_name,
        target_resolution_status=resolution.status.value,
    )


def _result_from_last_seen(
    query: Query,
    state: LastSeenState,
    *,
    target_class: str | None = None,
    target_resolution_status: str | None = None,
) -> StateQueryResult:
    display_zone = state.zone.replace("_", " ") if state.zone else "an unknown region"
    display_class = target_class or state.class_name
    return StateQueryResult(
        question_id=query.question_id,
        answer=(
            f"The {display_class} was last observed in the {display_zone} "
            f"at {state.timestamp:.2f} s."
        ),
        status=StateAnswerStatus.TARGET_LAST_SEEN,
        persistent_id=state.persistent_id,
        class_name=display_class,
        zone=state.zone,
        last_seen_timestamp=state.timestamp,
        frame_index=state.frame_index,
        confidence=state.confidence,
        video_end_timestamp=state.video_end_timestamp,
        staleness_seconds=state.staleness_seconds,
        evidence_ref=state.evidence_ref,
        target_resolution_status=target_resolution_status,
    )


def _evidence_to_record(evidence: EvidenceRef | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }
