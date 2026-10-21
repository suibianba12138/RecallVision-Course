"""Deterministic event-relative WHERE_OBJECT_BEFORE retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import math

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline
from recallvision.query.object_movement import _event_video_id, _target_candidates
from recallvision.query.schema import Answer, AnswerStatus, Query, QueryType


def answer_where_object_before(
    query: Query,
    timelines: Sequence[ObjectStateTimeline],
    events: Sequence[EventRecord],
) -> Answer:
    """Retrieve the observed FROM state for one supported movement event."""

    if query.query_type is not QueryType.WHERE_OBJECT_BEFORE:
        return _error(
            query,
            f"UNSUPPORTED_QUERY: {query.query_type.value}",
            AnswerStatus.UNSUPPORTED_QUERY,
        )

    candidates = _target_candidates(query, timelines)
    target_label = query.target_class or ", ".join(query.target_entity_ids) or "target"
    if not candidates:
        return _error(
            query,
            f"TARGET_NOT_FOUND: no observed object for {target_label}.",
            AnswerStatus.TARGET_NOT_FOUND,
            target_class=query.target_class,
        )
    if len(candidates) != 1:
        return _error(
            query,
            f"AMBIGUOUS_TARGET: multiple persistent entities match {target_label}.",
            AnswerStatus.AMBIGUOUS_TARGET,
            target_class=query.target_class,
        )

    target = candidates[0]
    if query.reference_event_id is None:
        matching = [
            event
            for event in events
            if event.event_type is EventType.MOVE_OBJECT
            and target.persistent_id in event.object_ids
            and _event_video_id(event) == query.video_id
        ]
        if not matching:
            return _error(
                query,
                f"No supported movement event was found to use as a reference for the {target.class_name}.",
                AnswerStatus.NO_REFERENCE_EVENT,
                target_entity_id=target.persistent_id,
                target_class=target.class_name,
            )
        if len(matching) > 1:
            return _error(
                query,
                f"AMBIGUOUS_MULTIPLE_EVENT_REFERENCE: multiple supported movement events match the {target.class_name}; provide reference_event_id.",
                AnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE,
                target_entity_id=target.persistent_id,
                target_class=target.class_name,
            )
        event = matching[0]
    else:
        references = [event for event in events if event.event_id == query.reference_event_id]
        if len(references) != 1:
            return _invalid_reference(
                query,
                f"event {query.reference_event_id!r} does not identify exactly one event",
                target,
            )
        event = references[0]

    invalid = _validate_event_reference(query, target, event)
    if invalid is not None:
        return _invalid_reference(query, invalid, target)

    resolved = _resolve_source_segments(query, target, event, timelines)
    if isinstance(resolved, str):
        return _invalid_reference(query, resolved, target, provenance=True)
    from_segment_id, from_segment = resolved
    zone = from_segment.zone
    assert zone is not None
    last_evidence = from_segment.last_evidence
    timestamp = last_evidence.timestamp
    if timestamp is None or not last_evidence.frame_indices:
        return _invalid_reference(
            query,
            "FROM source segment lacks usable last observation evidence",
            target,
            provenance=True,
        )
    display_zone = zone.replace("_", " ")
    return Answer(
        question_id=query.question_id,
        answer=(
            f"Before that supported movement event, the {target.class_name} was last observed "
            f"in the {display_zone}."
        ),
        evidence=[last_evidence],
        status=AnswerStatus.OK,
        target_entity_id=target.persistent_id,
        target_class=target.class_name,
        zone=zone,
        timestamp=timestamp,
        frame_index=last_evidence.frame_indices[-1],
        query_type=query.query_type,
        reference_event_id=event.event_id,
        from_source_segment_id=from_segment_id,
    )


def _validate_event_reference(
    query: Query,
    target: ObjectStateTimeline,
    event: EventRecord,
) -> str | None:
    if event.event_type is not EventType.MOVE_OBJECT:
        return f"event {event.event_id!r} is not a MOVE_OBJECT event"
    if _event_video_id(event) != query.video_id:
        return f"event {event.event_id!r} does not belong to query video {query.video_id!r}"
    if any(evidence.video_id != query.video_id for evidence in event.evidence):
        return f"event {event.event_id!r} contains evidence from another video"
    if target.persistent_id not in event.object_ids:
        return f"event {event.event_id!r} does not reference target {target.persistent_id!r}"
    return None


def _resolve_source_segments(
    query: Query,
    target: ObjectStateTimeline,
    event: EventRecord,
    timelines: Sequence[ObjectStateTimeline],
) -> tuple[str, ObjectStateRecord] | str:
    source_ids = event.source_segment_ids
    if len(source_ids) != 2 or any(not isinstance(item, str) or not item for item in source_ids):
        return f"event {event.event_id!r} has unusable source_segment_ids provenance"

    from_match = _find_segment(source_ids[0], timelines)
    to_match = _find_segment(source_ids[1], timelines)
    if from_match is None:
        return f"INVALID_PROVENANCE: FROM source segment {source_ids[0]!r} was not found"
    if to_match is None:
        return f"INVALID_PROVENANCE: TO source segment {source_ids[1]!r} was not found"
    from_timeline, from_segment = from_match
    to_timeline, to_segment = to_match
    if from_timeline.video_id != query.video_id or to_timeline.video_id != query.video_id:
        return "INVALID_PROVENANCE: source segments do not belong to query video"
    if from_timeline.persistent_id != target.persistent_id or to_timeline.persistent_id != target.persistent_id:
        return "INVALID_PROVENANCE: source segments do not belong to the resolved target"
    if from_segment.persistent_id != target.persistent_id or to_segment.persistent_id != target.persistent_id:
        return "INVALID_PROVENANCE: source segment persistent identity does not match target"
    from_error = _validate_state_segment(from_timeline, from_segment, label="FROM")
    if from_error is not None:
        return from_error
    to_error = _validate_state_segment(to_timeline, to_segment, label="TO")
    if to_error is not None:
        return to_error
    return source_ids[0], from_segment


def _validate_state_segment(
    timeline: ObjectStateTimeline,
    segment: ObjectStateRecord,
    *,
    label: str,
) -> str | None:
    if segment.video_id != timeline.video_id:
        return f"INVALID_PROVENANCE: {label} source segment video does not match timeline"
    if segment.persistent_id != timeline.persistent_id:
        return f"INVALID_PROVENANCE: {label} source segment persistent ID does not match timeline"
    if not isinstance(segment.zone, str) or not segment.zone:
        return f"INVALID_PROVENANCE: {label} source segment has no observed zone/state"
    for evidence_label, evidence, expected_timestamp, expected_frame in (
        ("first", segment.first_evidence, segment.start_time, segment.start_frame),
        ("last", segment.last_evidence, segment.end_time, segment.end_frame),
    ):
        if evidence is None:
            return f"INVALID_PROVENANCE: {label} {evidence_label} evidence is missing"
        if evidence.video_id != timeline.video_id:
            return f"INVALID_PROVENANCE: {label} {evidence_label} evidence video does not match segment"
        if evidence.timestamp is None or not math.isfinite(evidence.timestamp):
            return f"INVALID_PROVENANCE: {label} {evidence_label} evidence timestamp is missing"
        if not math.isclose(evidence.timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-12):
            return f"INVALID_PROVENANCE: {label} {evidence_label} evidence timestamp does not match segment"
        if not evidence.frame_indices or expected_frame not in evidence.frame_indices:
            return f"INVALID_PROVENANCE: {label} {evidence_label} evidence frame does not match segment"
    return None


def _find_segment(
    source_segment_id: str,
    timelines: Sequence[ObjectStateTimeline],
) -> tuple[ObjectStateTimeline, ObjectStateRecord] | None:
    matches: list[tuple[ObjectStateTimeline, ObjectStateRecord]] = []
    for timeline in timelines:
        for index, segment in enumerate(_ordered_segments(timeline), start=1):
            expected_id = _segment_id(timeline, index)
            if expected_id == source_segment_id:
                matches.append((timeline, segment))
    if len(matches) != 1:
        return None
    return matches[0]


def _ordered_segments(timeline: ObjectStateTimeline) -> tuple[ObjectStateRecord, ...]:
    return tuple(
        sorted(
            timeline.segments,
            key=lambda item: (item.start_time, item.start_frame, item.end_time, item.end_frame),
        )
    )


def _segment_id(timeline: ObjectStateTimeline, index: int) -> str:
    return f"{timeline.video_id}#{timeline.persistent_id}#segment={index}"


def _invalid_reference(
    query: Query,
    reason: str,
    target: ObjectStateTimeline,
    *,
    provenance: bool = False,
) -> Answer:
    prefix = "INVALID_PROVENANCE" if provenance and not reason.startswith("INVALID_PROVENANCE") else ""
    message = f"{prefix}: {reason}" if prefix else reason
    return _error(
        query,
        f"INVALID_REFERENCE: {message}",
        AnswerStatus.UNSUPPORTED_QUERY,
        target_entity_id=target.persistent_id,
        target_class=target.class_name,
    )


def _error(
    query: Query,
    answer: str,
    status: AnswerStatus,
    *,
    target_entity_id: str | None = None,
    target_class: str | None = None,
) -> Answer:
    return Answer(
        question_id=query.question_id,
        answer=answer,
        status=status,
        target_entity_id=target_entity_id,
        target_class=target_class,
        query_type=query.query_type,
    )
