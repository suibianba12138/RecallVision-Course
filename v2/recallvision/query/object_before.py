"""Deterministic event-relative WHERE_OBJECT_BEFORE retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline
from recallvision.query.object_movement import _event_video_id
from recallvision.query.schema import Answer, AnswerStatus, Query, QueryType
from recallvision.query.target_resolution import (
    TargetResolutionStatus,
    candidates_from_timelines,
    resolve_query_target,
)


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

    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    target_label = resolution.reference.canonical_label or ", ".join(query.target_entity_ids) or "target"
    if resolution.status is not TargetResolutionStatus.RESOLVED:
        status = (
            AnswerStatus.AMBIGUOUS_TARGET
            if resolution.status is TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES
            else AnswerStatus.TARGET_NOT_FOUND
        )
        return _error(
            query,
            (
                f"AMBIGUOUS_TARGET: multiple persistent entities match {target_label}."
                if status is AnswerStatus.AMBIGUOUS_TARGET
                else f"TARGET_NOT_FOUND: no observed object for {target_label}."
            ),
            status,
            target_class=query.target_class,
            target_resolution_status=resolution.status.value,
        )

    target_id = resolution.resolved_entity_id
    assert target_id is not None
    target = next(
        timeline for timeline in timelines if timeline.video_id == query.video_id and timeline.persistent_id == target_id
    )
    display_class = resolution.reference.canonical_label or target.class_name
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
                f"No supported movement event was found to use as a reference for the {display_class}.",
                AnswerStatus.NO_REFERENCE_EVENT,
                target_entity_id=target.persistent_id,
                target_class=display_class,
                target_resolution_status=resolution.status.value,
            )
        if len(matching) > 1:
            return _error(
                query,
                f"AMBIGUOUS_MULTIPLE_EVENT_REFERENCE: multiple supported movement events match the {display_class}; provide reference_event_id.",
                AnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE,
                target_entity_id=target.persistent_id,
                target_class=display_class,
                target_resolution_status=resolution.status.value,
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

    motion = event.motion_metadata or {}
    trigger_provenance = motion.get("trigger_provenance", ())
    has_continuous_trigger = (
        isinstance(trigger_provenance, (list, tuple))
        and "CONTINUOUS_DISPLACEMENT" in trigger_provenance
    )
    if has_continuous_trigger or (motion and not event.source_segment_ids and "position_before" in motion):
        resolved_motion = _resolve_motion_observations(query, target, event)
        if isinstance(resolved_motion, str):
            if resolved_motion.startswith("INSUFFICIENT_EVIDENCE:"):
                return _motion_insufficient(query, target, event, resolved_motion)
            return _invalid_reference(query, resolved_motion, target, provenance=True)
        return _answer_motion_before(query, target, event, resolved_motion)

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
            f"Before that supported movement event, the {display_class} was last observed "
            f"in the {display_zone}."
        ),
        evidence=[last_evidence],
        status=AnswerStatus.OK,
        target_entity_id=target.persistent_id,
        target_class=display_class,
        zone=zone,
        timestamp=timestamp,
        frame_index=last_evidence.frame_indices[-1],
        query_type=query.query_type,
        reference_event_id=event.event_id,
        from_source_segment_id=from_segment_id,
        target_resolution_status=resolution.status.value,
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
    effective_entity_id = (event.motion_metadata or {}).get("effective_entity_id")
    if effective_entity_id is not None and effective_entity_id != target.persistent_id:
        return (
            f"event {event.event_id!r} effective entity {effective_entity_id!r} "
            f"does not match resolved target {target.persistent_id!r}"
        )
    if target.persistent_id not in event.object_ids:
        return f"event {event.event_id!r} does not reference target {target.persistent_id!r}"
    return None


def _resolve_motion_observations(
    query: Query,
    target: ObjectStateTimeline,
    event: EventRecord,
) -> dict[str, Any] | str:
    """Resolve direct detection evidence for an accepted continuous event."""

    metadata = event.motion_metadata or {}
    if metadata.get("evidence_status") != "SUPPORTED_EVENT":
        return "INSUFFICIENT_EVIDENCE: movement event is not marked as evidence-supported"
    if metadata.get("effective_entity_id") != target.persistent_id:
        return "INVALID_PROVENANCE: movement effective entity does not match the resolved target"

    before_stable = _normalized_position(metadata.get("position_before"))
    after_stable = _normalized_position(metadata.get("position_after"))
    if before_stable is None or after_stable is None:
        return "INSUFFICIENT_EVIDENCE: event lacks reliable normalized image positions"
    if len(event.evidence) < 2:
        return "INSUFFICIENT_EVIDENCE: event lacks separate before and after frame evidence"

    references = metadata.get("observation_references")
    if not isinstance(references, list):
        return "INSUFFICIENT_EVIDENCE: event has no direct observation references"
    before_observation = _select_window_observation(
        references,
        event.evidence[0],
        before_stable,
        video_id=query.video_id,
        entity_id=target.persistent_id,
    )
    after_observation = _select_window_observation(
        references,
        event.evidence[1],
        after_stable,
        video_id=query.video_id,
        entity_id=target.persistent_id,
    )
    if before_observation is None:
        return "INSUFFICIENT_EVIDENCE: no direct pre-event detection matches the event window"
    if after_observation is None:
        return "INSUFFICIENT_EVIDENCE: no direct post-event detection matches the event window"
    if (
        before_observation["frame_index"] >= after_observation["frame_index"]
        or before_observation["timestamp"] >= after_observation["timestamp"]
    ):
        return "INVALID_PROVENANCE: before and after detection references are not time ordered"

    before_zone = _position_zone(metadata.get("position_before"), event.zone_before)
    after_zone = _position_zone(metadata.get("position_after"), event.zone_after)
    if event.zone_before is not None and before_zone != event.zone_before:
        return "INVALID_PROVENANCE: pre-event position zone conflicts with event zone_before"
    if event.zone_after is not None and after_zone != event.zone_after:
        return "INVALID_PROVENANCE: post-event position zone conflicts with event zone_after"

    identity = metadata.get("identity_provenance")
    if identity is not None and not isinstance(identity, dict):
        return "INVALID_PROVENANCE: identity_provenance must be an object"
    identity = dict(identity or {})
    displacement = metadata.get("displacement")
    stable_displacement = None
    if isinstance(displacement, dict):
        value = displacement.get("normalized_l2")
        if _finite_number(value):
            stable_displacement = float(value)

    return {
        "before_observation": before_observation,
        "after_observation": after_observation,
        "stable_before": before_stable,
        "stable_after": after_stable,
        "before_zone": before_zone,
        "after_zone": after_zone,
        "identity_provenance": identity,
        "overlay_revision": identity.get("overlay_revision"),
        "displacement": stable_displacement,
    }


def _select_window_observation(
    references: list[Any],
    evidence: Any,
    stable_position: tuple[float, float],
    *,
    video_id: str,
    entity_id: str,
) -> dict[str, Any] | None:
    if not evidence.frame_indices:
        return None
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        frame = reference.get("frame_index")
        timestamp = reference.get("timestamp")
        center = _normalized_pair(reference.get("center_normalized"))
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            continue
        if frame not in evidence.frame_indices or not _finite_number(timestamp) or center is None:
            continue
        if reference.get("video_id", video_id) != video_id:
            continue
        if reference.get("effective_entity_id", entity_id) != entity_id:
            continue
        timestamp = float(timestamp)
        if evidence.start_time is not None and timestamp < evidence.start_time - 1e-9:
            continue
        if evidence.end_time is not None and timestamp > evidence.end_time + 1e-9:
            continue
        distance = math.hypot(center[0] - stable_position[0], center[1] - stable_position[1])
        candidates.append((distance, frame, {
            "frame_index": frame,
            "timestamp": timestamp,
            "center_normalized": center,
        }))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _normalized_position(record: Any) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    return _normalized_pair(record.get("center_normalized"))


def _normalized_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(_finite_number(item) and 0.0 <= float(item) <= 1.0 for item in value):
        return None
    return float(value[0]), float(value[1])


def _position_zone(record: Any, fallback: str | None) -> str | None:
    if isinstance(record, dict):
        zone = record.get("zone")
        if isinstance(zone, str) and zone:
            return zone
    return fallback


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _answer_motion_before(
    query: Query,
    target: ObjectStateTimeline,
    event: EventRecord,
    resolved: dict[str, Any],
) -> Answer:
    before = resolved["before_observation"]
    after = resolved["after_observation"]
    before_center = before["center_normalized"]
    after_center = after["center_normalized"]
    before_zone = resolved["before_zone"]
    after_zone = resolved["after_zone"]
    zone_phrase = f" in the {before_zone.replace('_', ' ')} zone" if before_zone else ""
    answer = (
        f"Before the supported movement, the {target.class_name} was detected{zone_phrase} "
        f"at normalized image bounding-box center ({before_center[0]:.4f}, {before_center[1]:.4f}) "
        f"at {before['timestamp']:.3f} seconds (frame {before['frame_index']}). "
        f"Afterward its detected center was ({after_center[0]:.4f}, {after_center[1]:.4f}) "
        f"at {after['timestamp']:.3f} seconds (frame {after['frame_index']}). "
        "These are 2D image coordinates, not real-world or 3D positions."
    )
    evidence = []
    for record, window_evidence in (
        (before, event.evidence[0]),
        (after, event.evidence[1]),
    ):
        timestamp = record["timestamp"]
        evidence.append(EvidenceRef(
            video_id=query.video_id,
            timestamp=timestamp,
            start_time=timestamp,
            end_time=timestamp,
            frame_indices=[record["frame_index"]],
            segment_ref=window_evidence.segment_ref,
        ))
    return Answer(
        question_id=query.question_id,
        answer=answer,
        evidence=evidence,
        status=AnswerStatus.OK,
        target_entity_id=target.persistent_id,
        target_class=target.class_name,
        zone=before_zone,
        timestamp=before["timestamp"],
        frame_index=before["frame_index"],
        query_type=query.query_type,
        reference_event_id=event.event_id,
        target_resolution_status="RESOLVED",
        zone_after=after_zone,
        position_before_normalized=tuple(before_center),
        position_after_normalized=tuple(after_center),
        stable_position_before_normalized=tuple(resolved["stable_before"]),
        stable_position_after_normalized=tuple(resolved["stable_after"]),
        position_coordinate_space="NORMALIZED_IMAGE_BBOX_CENTER",
        position_after_timestamp=after["timestamp"],
        position_after_frame_index=after["frame_index"],
        identity_provenance=resolved["identity_provenance"],
        overlay_revision=resolved["overlay_revision"],
        displacement_normalized=resolved["displacement"],
    )


def _motion_insufficient(
    query: Query,
    target: ObjectStateTimeline,
    event: EventRecord,
    reason: str,
) -> Answer:
    identity = (event.motion_metadata or {}).get("identity_provenance")
    identity = dict(identity) if isinstance(identity, dict) else None
    return Answer(
        question_id=query.question_id,
        answer=reason,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        target_entity_id=target.persistent_id,
        target_class=target.class_name,
        query_type=query.query_type,
        reference_event_id=event.event_id,
        target_resolution_status="RESOLVED",
        identity_provenance=identity,
        overlay_revision=identity.get("overlay_revision") if identity else None,
    )


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
    target_resolution_status: str | None = None,
) -> Answer:
    return Answer(
        question_id=query.question_id,
        answer=answer,
        status=status,
        target_entity_id=target_entity_id,
        target_class=target_class,
        query_type=query.query_type,
        target_resolution_status=target_resolution_status,
    )
