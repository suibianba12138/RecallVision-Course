"""Deterministic observed zone-transition events from P2B timelines."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline


class ZoneTransitionEventBuilder:
    """Convert validated adjacent observed state segments into events."""

    def build(self, timeline: ObjectStateTimeline) -> list[EventRecord]:
        """Build events for one entity without mutating its source timeline."""

        segments = _validate_and_order_segments(timeline)
        events: list[EventRecord] = []
        for index, (from_segment, to_segment) in enumerate(zip(segments, segments[1:]), start=1):
            if from_segment.zone == to_segment.zone:
                continue
            source_segment_ids = (
                _segment_id(timeline, index),
                _segment_id(timeline, index + 1),
            )
            events.append(
                EventRecord(
                    event_id=f"zone-change:{timeline.video_id}:{timeline.persistent_id}:{index}->{index + 1}",
                    event_type=EventType.ZONE_CHANGE,
                    start_time=from_segment.end_time,
                    end_time=to_segment.start_time,
                    object_ids=[timeline.persistent_id],
                    zone_before=from_segment.zone,
                    zone_after=to_segment.zone,
                    evidence=[from_segment.last_evidence, to_segment.first_evidence],
                    source_segment_ids=source_segment_ids,
                )
            )
        return events

    def build_all(self, timelines: Iterable[ObjectStateTimeline]) -> list[EventRecord]:
        """Build independently per timeline, sorted for stable multi-entity output."""

        ordered = sorted(timelines, key=lambda item: (item.video_id, item.persistent_id))
        events: list[EventRecord] = []
        for timeline in ordered:
            events.extend(self.build(timeline))
        return events


def _validate_and_order_segments(
    timeline: ObjectStateTimeline,
) -> tuple[ObjectStateRecord, ...]:
    if not timeline.persistent_id:
        raise ValueError("timeline persistent_id is required")
    if not timeline.class_name:
        raise ValueError("timeline class_name is required")
    if not timeline.video_id:
        raise ValueError("timeline video_id is required")

    checked: list[ObjectStateRecord] = []
    for segment in timeline.segments:
        if segment.persistent_id != timeline.persistent_id:
            raise ValueError("state segments must use the same persistent_id")
        if segment.class_name != timeline.class_name:
            raise ValueError("state segments must use the same class_name")
        if segment.video_id != timeline.video_id:
            raise ValueError("state segments must use the same video_id")
        if not isinstance(segment.zone, str) or not segment.zone:
            raise ValueError("state segment zone is required")
        if segment.start_frame < 0 or segment.end_frame < 0 or segment.start_frame > segment.end_frame:
            raise ValueError("segment frame bounds are invalid")
        if not math.isfinite(segment.start_time) or not math.isfinite(segment.end_time):
            raise ValueError("segment time bounds must be finite")
        if segment.start_time < 0 or segment.end_time < 0 or segment.start_time > segment.end_time:
            raise ValueError("segment temporal bounds are invalid")
        _validate_boundary_evidence(
            segment.first_evidence,
            expected_video_id=timeline.video_id,
            expected_timestamp=segment.start_time,
            expected_frame=segment.start_frame,
            label="first_evidence",
        )
        _validate_boundary_evidence(
            segment.last_evidence,
            expected_video_id=timeline.video_id,
            expected_timestamp=segment.end_time,
            expected_frame=segment.end_frame,
            label="last_evidence",
        )
        checked.append(segment)

    ordered = tuple(
        sorted(
            checked,
            key=lambda item: (item.start_time, item.start_frame, item.end_time, item.end_frame),
        )
    )
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_time < previous.end_time or current.start_frame <= previous.end_frame:
            raise ValueError("state segments overlap or are temporally inverted")
    return ordered


def _validate_boundary_evidence(
    evidence: EvidenceRef | None,
    *,
    expected_video_id: str,
    expected_timestamp: float,
    expected_frame: int,
    label: str,
) -> None:
    if evidence is None:
        raise ValueError(f"{label} is required")
    if evidence.video_id != expected_video_id:
        raise ValueError(f"{label} video_id does not match timeline")
    if evidence.timestamp is None or not math.isfinite(evidence.timestamp):
        raise ValueError(f"{label} timestamp is required")
    if not math.isclose(evidence.timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} timestamp does not match segment boundary")
    if expected_frame not in evidence.frame_indices:
        raise ValueError(f"{label} frame does not match segment boundary")
    if not evidence.segment_ref:
        raise ValueError(f"{label} segment_ref is required")


def _segment_id(timeline: ObjectStateTimeline, index: int) -> str:
    return f"{timeline.video_id}#{timeline.persistent_id}#segment={index}"
