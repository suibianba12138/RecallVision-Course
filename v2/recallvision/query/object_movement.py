"""Deterministic WHEN_OBJECT_MOVED retrieval over derived event memory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateTimeline
from recallvision.query.schema import Query, QueryType
from recallvision.query.target_resolution import (
    TargetResolutionStatus,
    candidates_from_timelines,
    resolve_query_target,
)


class MovementAnswerStatus(str, Enum):
    """Explicit result states for event-grounded movement retrieval."""

    MOVEMENT_FOUND = "MOVEMENT_FOUND"
    NO_SUPPORTED_EVENT = "NO_SUPPORTED_EVENT"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


@dataclass(frozen=True)
class MovementQueryResult:
    """A structured, potentially multi-event movement answer."""

    question_id: str
    answer: str
    status: MovementAnswerStatus
    persistent_id: str | None = None
    class_name: str | None = None
    events: tuple[EventRecord, ...] = ()
    time_intervals: tuple[tuple[float, float], ...] = ()
    frame_intervals: tuple[tuple[int | None, int | None], ...] = ()
    target_resolution_status: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query_type": QueryType.WHEN_OBJECT_MOVED.value,
            "answer": self.answer,
            "status": self.status.value,
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "matched_event_count": len(self.events),
            "event_ids": [event.event_id for event in self.events],
            "time_intervals": [
                {
                    "event_id": event.event_id,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                for event, (start_time, end_time) in zip(self.events, self.time_intervals)
            ],
            "frame_intervals": [
                {
                    "event_id": event.event_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
                for event, (start_frame, end_frame) in zip(self.events, self.frame_intervals)
            ],
            "events": [event.to_record() for event in self.events],
            "evidence": [
                {
                    "event_id": event.event_id,
                    "references": [_evidence_to_record(item) for item in event.evidence],
                }
                for event in self.events
            ],
            "target_resolution_status": self.target_resolution_status,
        }


def answer_when_object_moved(
    query: Query,
    timelines: Sequence[ObjectStateTimeline],
    events: Sequence[EventRecord],
) -> MovementQueryResult:
    """Retrieve supported movement events for one resolved persistent object."""

    if query.query_type is not QueryType.WHEN_OBJECT_MOVED:
        return MovementQueryResult(
            question_id=query.question_id,
            answer=f"UNSUPPORTED_QUERY: {query.query_type.value}",
            status=MovementAnswerStatus.UNSUPPORTED_QUERY,
        )

    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    target_label = resolution.reference.canonical_label or ", ".join(query.target_entity_ids) or "target"
    if resolution.status is not TargetResolutionStatus.RESOLVED:
        status = (
            MovementAnswerStatus.AMBIGUOUS_TARGET
            if resolution.status is TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES
            else MovementAnswerStatus.TARGET_NOT_FOUND
        )
        return MovementQueryResult(
            question_id=query.question_id,
            answer=(
                f"AMBIGUOUS_TARGET: multiple persistent entities match {target_label}."
                if status is MovementAnswerStatus.AMBIGUOUS_TARGET
                else f"TARGET_NOT_FOUND: no observed object for {target_label}."
            ),
            status=status,
            target_resolution_status=resolution.status.value,
        )

    target_id = resolution.resolved_entity_id
    assert target_id is not None
    target = next(
        timeline for timeline in timelines if timeline.video_id == query.video_id and timeline.persistent_id == target_id
    )
    display_class = resolution.reference.canonical_label or target.class_name
    matched = tuple(
        sorted(
            (
                event
                for event in events
                if event.event_type is EventType.MOVE_OBJECT
                and target.persistent_id in event.object_ids
                and _event_video_id(event) == query.video_id
            ),
            key=lambda event: (event.start_time, event.end_time, event.event_id),
        )
    )
    if not matched:
        return MovementQueryResult(
            question_id=query.question_id,
            answer=f"No supported movement event was found for the {display_class}.",
            status=MovementAnswerStatus.NO_SUPPORTED_EVENT,
            persistent_id=target.persistent_id,
            class_name=display_class,
            target_resolution_status=resolution.status.value,
        )

    time_intervals = tuple((event.start_time, event.end_time) for event in matched)
    frame_intervals = tuple(_boundary_frame_interval(event) for event in matched)
    return MovementQueryResult(
        question_id=query.question_id,
        answer=_render_answer(display_class, matched),
        status=MovementAnswerStatus.MOVEMENT_FOUND,
        persistent_id=target.persistent_id,
        class_name=display_class,
        target_resolution_status=resolution.status.value,
        events=matched,
        time_intervals=time_intervals,
        frame_intervals=frame_intervals,
    )


def _target_candidates(
    query: Query, timelines: Sequence[ObjectStateTimeline]
) -> list[ObjectStateTimeline]:
    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    if resolution.status is not TargetResolutionStatus.RESOLVED:
        return []
    assert resolution.resolved_entity_id is not None
    return [
        timeline
        for timeline in timelines
        if timeline.video_id == query.video_id
        and timeline.persistent_id == resolution.resolved_entity_id
    ]


def _event_video_id(event: EventRecord) -> str | None:
    if event.evidence:
        return event.evidence[0].video_id
    if event.source_segment_ids:
        return event.source_segment_ids[0].split("#", 1)[0]
    return None


def _boundary_frame_interval(event: EventRecord) -> tuple[int | None, int | None]:
    """Read only the two P2D boundary references, never confirmation context."""

    if len(event.evidence) < 2:
        return (None, None)
    start_frames = event.evidence[0].frame_indices
    end_frames = event.evidence[1].frame_indices
    return (
        start_frames[0] if start_frames else None,
        end_frames[0] if end_frames else None,
    )


def _render_answer(class_name: str, events: Sequence[EventRecord]) -> str:
    intervals = [
        f"sometime between approximately {event.start_time:.2f} s and {event.end_time:.2f} s"
        for event in events
    ]
    if len(intervals) == 1:
        return f"The {class_name} was observed to move {intervals[0]}."
    return (
        f"The {class_name} was observed to move during {len(intervals)} supported "
        f"movement intervals: {'; '.join(intervals)}."
    )


def _evidence_to_record(evidence: EvidenceRef) -> dict[str, Any]:
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }
