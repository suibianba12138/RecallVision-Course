"""Deterministic, evidence-bounded MOVE_OBJECT event evaluation.

P2D deliberately starts from an existing P2C ``ZONE_CHANGE`` event.  This
module never attributes an action to a person and never interpolates a path
between observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Sequence

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline


EVIDENCE_WINDOW_POLICY = "BOUNDARY_LOCAL_TIME_WINDOW"
SUPPORT_HORIZON_SOURCE = (
    "config.max_transition_gap_seconds (P2B max_observation_gap_seconds)"
)


@dataclass(frozen=True)
class MovementObservation:
    """The observation fields required by the P2D spatial evaluator."""

    persistent_id: str
    frame_index: int
    timestamp: float
    zone: str | None
    center_normalized: tuple[float, float]
    source_track_ids: tuple[str, ...] = ()
    track_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.persistent_id:
            raise ValueError("persistent_id is required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a non-negative finite number")
        if not isinstance(self.zone, str) or not self.zone:
            raise ValueError("zone is required")
        if len(self.center_normalized) != 2 or any(
            not math.isfinite(value) for value in self.center_normalized
        ):
            raise ValueError("center_normalized must contain two finite values")
        if any(not isinstance(item, str) or not item for item in self.source_track_ids):
            raise ValueError("source_track_ids must contain non-empty strings")
        if self.track_id is not None and not self.track_id:
            raise ValueError("track_id must be non-empty or None")
        if self.bbox is not None and (
            len(self.bbox) != 4 or any(not math.isfinite(value) for value in self.bbox)
        ):
            raise ValueError("bbox must contain four finite values")
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite or None")


@dataclass(frozen=True)
class MovementEvidenceConfig:
    """Configurable P2D gates.

    The transition gap is inherited from P2B.  The support floors are small,
    explicit minimums for a robust summary and are not fitted to one video.
    """

    max_transition_gap_seconds: float = 0.50
    minimum_local_variation_samples: int = 3
    minimum_segment_observations: int = 3
    minimum_cluster_observations: int = 3
    local_variation_requires_same_track: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_transition_gap_seconds) or self.max_transition_gap_seconds < 0:
            raise ValueError("max_transition_gap_seconds must be non-negative and finite")
        for name in (
            "minimum_local_variation_samples",
            "minimum_segment_observations",
            "minimum_cluster_observations",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MovementEvidenceConfig":
        return cls(
            max_transition_gap_seconds=float(mapping.get("max_transition_gap_seconds", 0.50)),
            minimum_local_variation_samples=int(mapping.get("minimum_local_variation_samples", 3)),
            minimum_segment_observations=int(mapping.get("minimum_segment_observations", 3)),
            minimum_cluster_observations=int(mapping.get("minimum_cluster_observations", 3)),
            local_variation_requires_same_track=bool(
                mapping.get("local_variation_requires_same_track", True)
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "max_transition_gap_seconds": self.max_transition_gap_seconds,
            "minimum_local_variation_samples": self.minimum_local_variation_samples,
            "minimum_segment_observations": self.minimum_segment_observations,
            "minimum_cluster_observations": self.minimum_cluster_observations,
            "local_variation_requires_same_track": self.local_variation_requires_same_track,
            "evidence_window_policy": EVIDENCE_WINDOW_POLICY,
            "support_horizon_source": SUPPORT_HORIZON_SOURCE,
        }


@dataclass(frozen=True)
class MovementEvidenceResult:
    """One accepted event or one explicit fail-closed diagnostic."""

    accepted: bool
    reason: str
    event: EventRecord | None
    evidence: dict[str, Any]


class MovementEvidenceEvaluator:
    """Evaluate one existing P2C zone transition without mutating inputs."""

    def __init__(self, config: MovementEvidenceConfig | None = None) -> None:
        self.config = config or MovementEvidenceConfig()

    def evaluate(
        self,
        timeline: ObjectStateTimeline,
        observations: Iterable[MovementObservation],
        trigger_event: EventRecord | None,
    ) -> MovementEvidenceResult:
        ordered_observations = tuple(
            sorted(
                observations,
                key=lambda item: (
                    item.frame_index,
                    item.timestamp,
                    item.persistent_id,
                    item.source_track_ids,
                ),
            )
        )
        evidence = _initial_evidence(timeline, trigger_event, self.config)
        if trigger_event is None or trigger_event.event_type is not EventType.ZONE_CHANGE:
            return _rejected("ZONE_CHANGE_TRIGGER_REQUIRED", evidence)

        evidence["trigger_event_id"] = trigger_event.event_id
        if not _trigger_targets_timeline(trigger_event, timeline):
            evidence["gates"]["identity"] = {
                "passed": False,
                "reason": "trigger object or source segment does not match timeline",
                "raw_track_fragment_change_allowed": True,
            }
            return _rejected("IDENTITY_MISMATCH", evidence)

        segments = _segments_for_trigger(timeline, trigger_event)
        if segments is None:
            evidence["gates"]["identity"] = {
                "passed": False,
                "reason": "trigger source segments are not adjacent timeline segments",
                "raw_track_fragment_change_allowed": True,
            }
            return _rejected("IDENTITY_MISMATCH", evidence)
        before_segment, after_segment = segments
        evidence["source_segment_ids"] = list(trigger_event.source_segment_ids)
        evidence["from_zone"] = before_segment.zone
        evidence["to_zone"] = after_segment.zone
        evidence["boundary"] = _boundary_record(before_segment, after_segment)

        gap = trigger_event.end_time - trigger_event.start_time
        temporal_passed = (
            math.isfinite(gap)
            and gap >= 0
            and gap <= self.config.max_transition_gap_seconds
        )
        evidence["gates"]["temporal"] = {
            "passed": temporal_passed,
            "gap_seconds": gap,
            "max_gap_seconds": self.config.max_transition_gap_seconds,
            "start_time": trigger_event.start_time,
            "end_time": trigger_event.end_time,
        }
        if not temporal_passed:
            return _rejected("TEMPORAL_GAP_TOO_LARGE", evidence)

        candidate_frame_observations = [
            item
            for item in ordered_observations
            if before_segment.start_frame <= item.frame_index <= after_segment.end_frame
        ]
        candidate_ids = {item.persistent_id for item in candidate_frame_observations}
        if candidate_ids and timeline.persistent_id not in candidate_ids:
            evidence["gates"]["identity"] = {
                "passed": False,
                "persistent_id": timeline.persistent_id,
                "observed_persistent_ids": sorted(candidate_ids),
                "reason": "candidate boundary frame observations belong to another persistent entity",
                "raw_track_fragment_change_allowed": True,
            }
            return _rejected("IDENTITY_MISMATCH", evidence)

        before_observations = _segment_observations(
            ordered_observations, before_segment, timeline.persistent_id
        )
        after_observations = _segment_observations(
            ordered_observations, after_segment, timeline.persistent_id
        )
        evidence["segment_support"] = {
            "before_count": len(before_observations),
            "after_count": len(after_observations),
            "minimum_segment_observations": self.config.minimum_segment_observations,
        }
        support_passed = all(
            len(items) >= self.config.minimum_segment_observations
            for items in (before_observations, after_observations)
        )
        evidence["gates"]["sufficient_evidence"] = {
            "passed": support_passed,
            "minimum_segment_observations": self.config.minimum_segment_observations,
            "minimum_cluster_observations": self.config.minimum_cluster_observations,
            "minimum_local_variation_samples": self.config.minimum_local_variation_samples,
        }
        if not support_passed:
            return _rejected("INSUFFICIENT_SEGMENT_SUPPORT", evidence)

        left_support = _boundary_local_support(
            before_observations,
            boundary_time=before_segment.end_time,
            side="left",
            horizon_seconds=self.config.max_transition_gap_seconds,
        )
        right_support = _boundary_local_support(
            after_observations,
            boundary_time=after_segment.start_time,
            side="right",
            horizon_seconds=self.config.max_transition_gap_seconds,
        )
        evidence.update(
            _support_evidence(
                left_support,
                right_support,
                horizon_seconds=self.config.max_transition_gap_seconds,
            )
        )
        minimum_boundary_support = max(
            self.config.minimum_segment_observations,
            self.config.minimum_cluster_observations,
        )
        boundary_support_passed = all(
            len(items) >= minimum_boundary_support for items in (left_support, right_support)
        )
        evidence["segment_support"].update(
            {
                "left_support_count": len(left_support),
                "right_support_count": len(right_support),
                "minimum_boundary_support": minimum_boundary_support,
            }
        )
        evidence["gates"]["sufficient_evidence"] = {
            **evidence["gates"]["sufficient_evidence"],
            "passed": boundary_support_passed,
            "boundary_support_passed": boundary_support_passed,
        }
        if not boundary_support_passed:
            return _rejected("INSUFFICIENT_SEGMENT_SUPPORT", evidence)

        boundary_before = _boundary_observation(before_observations, before_segment.end_frame)
        boundary_after = _boundary_observation(after_observations, after_segment.start_frame)
        identity_passed = (
            timeline.persistent_id in candidate_ids
            and boundary_before is not None
            and boundary_after is not None
        )
        track_fragments = sorted(
            {
                track_id
                for item in (*before_observations, *after_observations)
                for track_id in item.source_track_ids
            }
        )
        evidence["gates"]["identity"] = {
            "passed": identity_passed,
            "persistent_id": timeline.persistent_id,
            "observed_persistent_ids": sorted(candidate_ids),
            "raw_track_ids": track_fragments,
            "raw_track_fragment_change_allowed": True,
        }
        if not identity_passed:
            return _rejected("IDENTITY_MISMATCH", evidence)

        assert boundary_before is not None
        assert boundary_after is not None
        local_variation = _local_variation(
            (left_support, right_support),
            requires_same_track=self.config.local_variation_requires_same_track,
        )
        evidence["local_variation"] = local_variation
        local_passed = local_variation["count"] >= self.config.minimum_local_variation_samples
        evidence["gates"]["sufficient_evidence"] = {
            **evidence["gates"]["sufficient_evidence"],
            "passed": local_passed,
            "local_variation_count": local_variation["count"],
        }
        if not local_passed:
            return _rejected("INSUFFICIENT_LOCAL_VARIATION", evidence)

        transition_displacement = _distance(
            boundary_before.center_normalized, boundary_after.center_normalized
        )
        displacement_passed = transition_displacement > local_variation["p95"]
        evidence["gates"]["displacement"] = {
            "passed": displacement_passed,
            "transition_normalized_displacement": transition_displacement,
            "local_variation_p95": local_variation["p95"],
            "strictly_greater": True,
        }
        if not displacement_passed:
            return _rejected("DISPLACEMENT_NOT_MATERIAL", evidence)

        before_cluster = _cluster_summary(left_support)
        after_cluster = _cluster_summary(right_support)
        evidence["before_cluster"] = before_cluster
        evidence["after_cluster"] = after_cluster
        evidence["bbox_diagnostics"] = {
            "before": _bbox_diagnostics(left_support),
            "after": _bbox_diagnostics(right_support),
        }
        cluster_separation = _distance(
            before_cluster["center_normalized"], after_cluster["center_normalized"]
        )
        envelope_sum = before_cluster["radius_p95"] + after_cluster["radius_p95"]
        cluster_nonoverlap_margin = cluster_separation - envelope_sum
        evidence.update(
            {
                "left_cluster_center": before_cluster["center_normalized"],
                "right_cluster_center": after_cluster["center_normalized"],
                "left_cluster_radius": before_cluster["radius_p95"],
                "right_cluster_radius": after_cluster["radius_p95"],
                "cluster_separation": cluster_separation,
                "cluster_nonoverlap_margin": cluster_nonoverlap_margin,
                "support_observations": {
                    "left": [_observation_to_record(item) for item in left_support],
                    "right": [_observation_to_record(item) for item in right_support],
                },
                "event_boundary_evidence": {
                    "left": _observation_to_record(boundary_before),
                    "right": _observation_to_record(boundary_after),
                },
                "local_state_confirmation_evidence": {
                    "left": [_observation_to_record(item) for item in left_support],
                    "right": [_observation_to_record(item) for item in right_support],
                },
            }
        )
        cluster_support_passed = all(
            item["count"] >= self.config.minimum_cluster_observations
            for item in (before_cluster, after_cluster)
        )
        evidence["gates"]["sufficient_evidence"] = {
            **evidence["gates"]["sufficient_evidence"],
            "passed": cluster_support_passed,
            "cluster_support_passed": cluster_support_passed,
        }
        if not cluster_support_passed:
            return _rejected("INSUFFICIENT_SEGMENT_SUPPORT", evidence)

        cluster_passed = cluster_separation > envelope_sum
        evidence["gates"]["cluster_separation"] = {
            "passed": cluster_passed,
            "separation": cluster_separation,
            "before_radius_p95": before_cluster["radius_p95"],
            "after_radius_p95": after_cluster["radius_p95"],
            "envelope_sum": envelope_sum,
            "nonoverlap_margin": cluster_nonoverlap_margin,
            "strictly_greater": True,
        }
        if not cluster_passed:
            return _rejected("CLUSTER_ENVELOPES_OVERLAP", evidence)

        event = build_move_object_event(trigger_event, timeline.persistent_id)
        evidence["event_provenance"] = {
            "move_object_event_id": event.event_id,
            "trigger_zone_change_event_id": trigger_event.event_id,
            "persistent_entity_id": timeline.persistent_id,
            "source_segment_ids": list(trigger_event.source_segment_ids),
            "boundary_evidence": [
                _evidence_to_record(before_segment.last_evidence),
                _evidence_to_record(after_segment.first_evidence),
            ],
        }
        evidence["gates"]["sufficient_evidence"] = {
            **evidence["gates"]["sufficient_evidence"],
            "passed": True,
        }
        evidence["status"] = "ACCEPTED"
        return MovementEvidenceResult(
            accepted=True,
            reason="ACCEPTED",
            event=event,
            evidence=evidence,
        )


def build_move_object_event(trigger_event: EventRecord, persistent_id: str) -> EventRecord:
    """Reuse the bounded P2C event evidence for an agent-neutral P2D event."""

    if trigger_event.event_type is not EventType.ZONE_CHANGE:
        raise ValueError("MOVE_OBJECT requires a ZONE_CHANGE trigger")
    if not persistent_id:
        raise ValueError("persistent_id is required")
    return EventRecord(
        event_id=f"move-object:{trigger_event.event_id.removeprefix('zone-change:')}",
        event_type=EventType.MOVE_OBJECT,
        start_time=trigger_event.start_time,
        end_time=trigger_event.end_time,
        person_ids=[],
        object_ids=[persistent_id],
        zone_before=trigger_event.zone_before,
        zone_after=trigger_event.zone_after,
        evidence=list(trigger_event.evidence),
        source_segment_ids=tuple(trigger_event.source_segment_ids),
    )


def _initial_evidence(
    timeline: ObjectStateTimeline,
    trigger_event: EventRecord | None,
    config: MovementEvidenceConfig,
) -> dict[str, Any]:
    return {
        "status": "REJECTED",
        "evidence_window_policy": EVIDENCE_WINDOW_POLICY,
        "support_horizon_seconds": config.max_transition_gap_seconds,
        "support_horizon_source": SUPPORT_HORIZON_SOURCE,
        "left_support_frames": [],
        "right_support_frames": [],
        "left_support_timestamps": [],
        "right_support_timestamps": [],
        "left_support_time_range": None,
        "right_support_time_range": None,
        "left_cluster_center": None,
        "right_cluster_center": None,
        "left_cluster_radius": None,
        "right_cluster_radius": None,
        "cluster_separation": None,
        "cluster_nonoverlap_margin": None,
        "persistent_id": timeline.persistent_id,
        "trigger_event_id": trigger_event.event_id if trigger_event is not None else None,
        "source_segment_ids": list(trigger_event.source_segment_ids) if trigger_event else [],
        "config": config.to_record(),
        "gates": {
            "temporal": {"passed": False},
            "identity": {"passed": False},
            "displacement": {"passed": False},
            "cluster_separation": {"passed": False},
            "sufficient_evidence": {"passed": False},
        },
        "local_variation": _empty_local_variation(),
        "provenance": {
            "input_timeline_persistent_id": timeline.persistent_id,
            "input_event_type": trigger_event.event_type.value if trigger_event else None,
        },
    }


def _rejected(reason: str, evidence: dict[str, Any]) -> MovementEvidenceResult:
    evidence["status"] = "REJECTED"
    evidence["rejection_reason"] = reason
    return MovementEvidenceResult(False, reason, None, evidence)


def _trigger_targets_timeline(event: EventRecord, timeline: ObjectStateTimeline) -> bool:
    return (
        event.event_type is EventType.ZONE_CHANGE
        and event.object_ids == [timeline.persistent_id]
        and len(event.source_segment_ids) == 2
        and len(event.evidence) == 2
    )


def _segments_for_trigger(
    timeline: ObjectStateTimeline, event: EventRecord
) -> tuple[ObjectStateRecord, ObjectStateRecord] | None:
    ordered = tuple(sorted(timeline.segments, key=lambda item: (item.start_time, item.start_frame)))
    expected = {
        f"{timeline.video_id}#{timeline.persistent_id}#segment={index}": segment
        for index, segment in enumerate(ordered, start=1)
    }
    if any(item not in expected for item in event.source_segment_ids):
        return None
    before = expected[event.source_segment_ids[0]]
    after = expected[event.source_segment_ids[1]]
    if ordered.index(after) != ordered.index(before) + 1:
        return None
    if event.zone_before != before.zone or event.zone_after != after.zone:
        return None
    return before, after


def _segment_observations(
    observations: Sequence[MovementObservation],
    segment: ObjectStateRecord,
    persistent_id: str,
) -> list[MovementObservation]:
    return [
        item
        for item in observations
        if item.persistent_id == persistent_id
        and segment.start_frame <= item.frame_index <= segment.end_frame
        and item.zone == segment.zone
    ]


def _boundary_local_support(
    observations: Sequence[MovementObservation],
    *,
    boundary_time: float,
    side: str,
    horizon_seconds: float,
) -> list[MovementObservation]:
    """Return only observations in the inclusive local window at one boundary."""

    if side == "left":
        lower = boundary_time - horizon_seconds
        upper = boundary_time
    elif side == "right":
        lower = boundary_time
        upper = boundary_time + horizon_seconds
    else:
        raise ValueError("side must be 'left' or 'right'")
    return sorted(
        [
            item
            for item in observations
            if lower <= item.timestamp <= upper
        ],
        key=lambda item: (item.frame_index, item.timestamp, item.source_track_ids),
    )


def _boundary_observation(
    observations: Sequence[MovementObservation], frame_index: int
) -> MovementObservation | None:
    candidates = [item for item in observations if item.frame_index == frame_index]
    return candidates[0] if candidates else None


def _local_variation(
    segments: Sequence[Sequence[MovementObservation]],
    *,
    requires_same_track: bool,
) -> dict[str, Any]:
    distances: list[float] = []
    for segment in segments:
        ordered = sorted(segment, key=lambda item: (item.frame_index, item.timestamp, item.source_track_ids))
        for left, right in zip(ordered, ordered[1:]):
            if requires_same_track and not _same_source_track(left, right):
                continue
            distances.append(_distance(left.center_normalized, right.center_normalized))
    if not distances:
        return _empty_local_variation()
    return {
        "count": len(distances),
        "median": median(distances),
        "p95": _quantile(distances, 0.95),
        "max": max(distances),
        "samples": distances,
    }


def _empty_local_variation() -> dict[str, Any]:
    return {"count": 0, "median": None, "p95": None, "max": None, "samples": []}


def _support_evidence(
    left_support: Sequence[MovementObservation],
    right_support: Sequence[MovementObservation],
    *,
    horizon_seconds: float,
) -> dict[str, Any]:
    def time_range(items: Sequence[MovementObservation]) -> list[float] | None:
        if not items:
            return None
        return [items[0].timestamp, items[-1].timestamp]

    return {
        "evidence_window_policy": EVIDENCE_WINDOW_POLICY,
        "support_horizon_seconds": horizon_seconds,
        "support_horizon_source": SUPPORT_HORIZON_SOURCE,
        "left_support_frames": [item.frame_index for item in left_support],
        "right_support_frames": [item.frame_index for item in right_support],
        "left_support_timestamps": [item.timestamp for item in left_support],
        "right_support_timestamps": [item.timestamp for item in right_support],
        "left_support_time_range": time_range(left_support),
        "right_support_time_range": time_range(right_support),
    }


def _same_source_track(left: MovementObservation, right: MovementObservation) -> bool:
    if left.track_id is not None and right.track_id is not None:
        return left.track_id == right.track_id
    return bool(set(left.source_track_ids).intersection(right.source_track_ids))


def _cluster_summary(observations: Sequence[MovementObservation]) -> dict[str, Any]:
    xs = [item.center_normalized[0] for item in observations]
    ys = [item.center_normalized[1] for item in observations]
    center = (median(xs), median(ys))
    radial = [_distance(item.center_normalized, center) for item in observations]
    return {
        "count": len(observations),
        "center_normalized": center,
        "radial_p95": _quantile(radial, 0.95),
        "radius_p95": _quantile(radial, 0.95),
        "radial_max": max(radial),
    }


def _bbox_diagnostics(observations: Sequence[MovementObservation]) -> dict[str, Any]:
    bboxes = [item.bbox for item in observations if item.bbox is not None]
    confidences = [item.confidence for item in observations if item.confidence is not None]
    if not bboxes:
        return {"count": 0, "available": False}
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    return {
        "count": len(bboxes),
        "available": True,
        "width_median": median(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_median": median(heights),
        "height_min": min(heights),
        "height_max": max(heights),
        "confidence_min": min(confidences) if confidences else None,
        "confidence_median": median(confidences) if confidences else None,
    }


def _observation_to_record(observation: MovementObservation) -> dict[str, Any]:
    return {
        "persistent_id": observation.persistent_id,
        "frame_index": observation.frame_index,
        "timestamp": observation.timestamp,
        "zone": observation.zone,
        "center_normalized": list(observation.center_normalized),
        "source_track_ids": list(observation.source_track_ids),
        "track_id": observation.track_id,
    }


def _boundary_record(
    before: ObjectStateRecord, after: ObjectStateRecord
) -> dict[str, Any]:
    return {
        "from_frame": before.end_frame,
        "to_frame": after.start_frame,
        "from_timestamp": before.end_time,
        "to_timestamp": after.start_time,
        "from_zone": before.zone,
        "to_zone": after.zone,
    }


def _evidence_to_record(evidence: EvidenceRef) -> dict[str, Any]:
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
