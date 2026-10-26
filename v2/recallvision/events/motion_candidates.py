"""Robust MOVE_OBJECT candidates from observed temporal position windows.

The detector consumes frozen observations only. It collapses same-frame boxes
with a coordinate-wise median, never interpolates missing detections, and uses
one evidence gate for both P2C zone transitions and independent 2D motion
candidates.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from recallvision.events.movement_evidence import MovementObservation
from recallvision.events.schema import EventRecord, EventType
from recallvision.events.camera_motion import VideoCameraMotionEstimator


@dataclass(frozen=True)
class MotionCandidateConfig:
    """Video-independent controls for the P3-B1 evidence evaluator."""

    max_observation_gap_seconds: float = 0.50
    window_observation_count: int = 3
    maximum_window_radius_normalized: float = 0.025
    minimum_displacement_normalized: float = 0.030
    maximum_same_frame_duplicate_spread_normalized: float = 0.025
    maximum_bbox_width_normalized: float = 0.80
    maximum_bbox_scale_ratio: float = 1.50
    suspected_scene_min_entities: int = 3
    maximum_scene_anchor_residual_normalized: float = 0.025
    excluded_entity_classes: tuple[str, ...] = ("person",)

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_observation_gap_seconds) or self.max_observation_gap_seconds < 0:
            raise ValueError("max_observation_gap_seconds must be finite and non-negative")
        if self.window_observation_count < 2:
            raise ValueError("window_observation_count must be at least two")
        for name in (
            "maximum_window_radius_normalized",
            "minimum_displacement_normalized",
            "maximum_same_frame_duplicate_spread_normalized",
            "maximum_bbox_width_normalized",
            "maximum_bbox_scale_ratio",
            "maximum_scene_anchor_residual_normalized",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.suspected_scene_min_entities < 1:
            raise ValueError("suspected_scene_min_entities must be at least one")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MotionCandidateConfig":
        return cls(
            max_observation_gap_seconds=float(mapping.get("max_observation_gap_seconds", 0.50)),
            window_observation_count=int(mapping.get("window_observation_count", 3)),
            maximum_window_radius_normalized=float(mapping.get("maximum_window_radius_normalized", 0.025)),
            minimum_displacement_normalized=float(mapping.get("minimum_displacement_normalized", 0.030)),
            maximum_same_frame_duplicate_spread_normalized=float(
                mapping.get("maximum_same_frame_duplicate_spread_normalized", 0.025)
            ),
            maximum_bbox_width_normalized=float(mapping.get("maximum_bbox_width_normalized", 0.80)),
            maximum_bbox_scale_ratio=float(mapping.get("maximum_bbox_scale_ratio", 1.50)),
            suspected_scene_min_entities=int(mapping.get("suspected_scene_min_entities", 3)),
            maximum_scene_anchor_residual_normalized=float(
                mapping.get("maximum_scene_anchor_residual_normalized", 0.025)
            ),
            excluded_entity_classes=tuple(mapping.get("excluded_entity_classes", ("person",))),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "max_observation_gap_seconds": self.max_observation_gap_seconds,
            "window_observation_count": self.window_observation_count,
            "maximum_window_radius_normalized": self.maximum_window_radius_normalized,
            "minimum_displacement_normalized": self.minimum_displacement_normalized,
            "maximum_same_frame_duplicate_spread_normalized": self.maximum_same_frame_duplicate_spread_normalized,
            "maximum_bbox_width_normalized": self.maximum_bbox_width_normalized,
            "maximum_bbox_scale_ratio": self.maximum_bbox_scale_ratio,
            "same_frame_position_aggregation": "COORDINATE_WISE_MEDIAN",
            "window_position": "MEDIAN_OF_FRAME_CONSENSUS_CENTERS",
            "missing_observation_policy": "NO_INTERPOLATION; MAX_DIRECT_GAP_GATE",
            "shot_boundary_policy": "EXPLICIT_BOUNDARIES_AND_VIDEO_FEATURE_OR_COMMON_TRACK_COMPENSATION",
            "suspected_scene_min_entities": self.suspected_scene_min_entities,
            "maximum_scene_anchor_residual_normalized": self.maximum_scene_anchor_residual_normalized,
            "excluded_entity_classes": list(self.excluded_entity_classes),
        }


@dataclass(frozen=True)
class _FramePoint:
    persistent_id: str
    frame_index: int
    timestamp: float
    center: tuple[float, float]
    zone: str
    track_ids: tuple[str, ...]
    bboxes: tuple[tuple[float, float, float, float], ...]
    confidences: tuple[float, ...]
    raw_count: int
    duplicate_spread: float
    duplicate_geometry_conflict: bool
    bbox_width_normalized: float | None
    bbox_height_normalized: float | None
    observation_refs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _PositionWindow:
    points: tuple[_FramePoint, ...]
    center: tuple[float, float]
    radius: float

    @property
    def first(self) -> _FramePoint:
        return self.points[0]

    @property
    def last(self) -> _FramePoint:
        return self.points[-1]

    @property
    def frames(self) -> list[int]:
        return [item.frame_index for item in self.points]

    @property
    def track_ids(self) -> set[str]:
        return {track_id for item in self.points for track_id in item.track_ids}

    @property
    def zone(self) -> str:
        counts = Counter(item.zone for item in self.points)
        return sorted(counts, key=lambda item: (-counts[item], item))[0]

    @property
    def maximum_bbox_width(self) -> float | None:
        values = [item.bbox_width_normalized for item in self.points if item.bbox_width_normalized is not None]
        return max(values) if values else None

    @property
    def median_bbox_size(self) -> tuple[float | None, float | None]:
        widths = [item.bbox_width_normalized for item in self.points if item.bbox_width_normalized is not None]
        heights = [item.bbox_height_normalized for item in self.points if item.bbox_height_normalized is not None]
        return (
            float(median(widths)) if widths else None,
            float(median(heights)) if heights else None,
        )

    @property
    def duplicate_geometry_conflicts(self) -> list[int]:
        return [item.frame_index for item in self.points if item.duplicate_geometry_conflict]


@dataclass(frozen=True)
class _Candidate:
    persistent_id: str
    candidate_type: str
    before: _PositionWindow
    after: _PositionWindow
    zone_before: str | None = None
    zone_after: str | None = None
    source_segment_ids: tuple[str, ...] = ()
    source_event_id: str | None = None


class RobustMotionEventDetector:
    """Generate and assess zone-triggered and continuous displacement candidates."""

    def __init__(self, config: MotionCandidateConfig | None = None) -> None:
        self.config = config or MotionCandidateConfig()

    def evaluate(
        self,
        observations: Iterable[MovementObservation],
        *,
        video_id: str,
        zone_events: Iterable[EventRecord] = (),
        identity_provenance: Mapping[str, Mapping[str, Any]] | None = None,
        shot_boundary_frames: Iterable[int] = (),
        video_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if not video_id:
            raise ValueError("video_id is required")
        source = tuple(observations)
        points_by_entity = _frame_consensus(source, self.config)
        provenance = identity_provenance or {}
        boundaries = set(shot_boundary_frames)
        camera_estimator = VideoCameraMotionEstimator(video_path) if video_path is not None else None
        candidates: list[_Candidate] = []
        candidates.extend(_continuous_candidates(points_by_entity, self.config))
        for event in zone_events:
            if event.event_type is not EventType.ZONE_CHANGE or len(event.object_ids) != 1:
                continue
            persistent_id = event.object_ids[0]
            points = points_by_entity.get(persistent_id, ())
            candidate = _zone_candidate(event, points, self.config)
            if candidate is not None:
                candidates.append(candidate)
            else:
                candidates.append(_empty_zone_candidate(event, persistent_id, points, self.config))

        candidates.sort(
            key=lambda item: (
                item.persistent_id,
                item.before.first.frame_index if item.before.points else -1,
                item.after.first.frame_index if item.after.points else -1,
                item.candidate_type,
            )
        )
        diagnostics: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        duplicate_count = 0
        for candidate in candidates:
            pid_points = points_by_entity.get(candidate.persistent_id, ())
            identity = _identity_record(candidate.persistent_id, provenance)
            diagnostic, event_record = _assess_candidate(
                candidate,
                pid_points,
                points_by_entity,
                video_id,
                identity,
                self.config,
                boundaries,
                camera_estimator,
            )
            if event_record is not None:
                duplicate = next(
                    (
                        prior
                        for prior in accepted
                        if _same_supported_transition(prior, event_record)
                    ),
                    None,
                )
                if duplicate is not None:
                    duplicate_count += 1
                    duplicate["trigger_provenance"] = sorted(
                        set(duplicate["trigger_provenance"] + event_record["trigger_provenance"]),
                        key=_trigger_order,
                    )
                    duplicate["source_event_ids"] = sorted(
                        set(duplicate.get("source_event_ids", []) + event_record.get("source_event_ids", []))
                    )
                    diagnostic["status"] = "REJECTED"
                    diagnostic["reason"] = "DUPLICATE_EVENT"
                    diagnostic["evidence_status"] = "REJECTED"
                    diagnostic["duplicate_of_event_id"] = duplicate["event_id"]
                else:
                    accepted.append(event_record)
            diagnostics.append(diagnostic)

        if camera_estimator is not None:
            camera_estimator.close()

        accepted.sort(key=lambda item: (item["start_time"], item["effective_entity_id"], item["event_id"]))
        accepted, coalesced_count = _coalesce_motion_episodes(accepted, diagnostics)
        diagnostics.sort(key=lambda item: (item.get("start_time", -1), item["effective_entity_id"], item["candidate_id"]))
        reason_counts = Counter(
            item["reason"] for item in diagnostics if item.get("status") != "ACCEPTED"
        )
        return {
            "events": accepted,
            "diagnostics": diagnostics,
            "summary": {
                "candidate_count": len(diagnostics),
                "accepted_event_count": len(accepted),
                "rejected_candidate_count": sum(item["status"] == "REJECTED" for item in diagnostics),
                "insufficient_evidence_count": sum(item["status"] == "INSUFFICIENT_EVIDENCE" for item in diagnostics),
                "duplicate_event_count": duplicate_count,
                "coalesced_event_count": coalesced_count,
                "reason_counts": dict(sorted(reason_counts.items())),
                "zone_trigger_candidate_count": sum(item["candidate_type"] == "ZONE_CHANGE" for item in diagnostics),
                "continuous_candidate_count": sum(item["candidate_type"] == "CONTINUOUS_DISPLACEMENT" for item in diagnostics),
                "same_zone_accepted_event_count": sum(
                    item.get("zone_before") == item.get("zone_after") for item in accepted
                ),
                "config": self.config.to_record(),
                "observations_are_frozen_inputs": True,
                "missing_observations_interpolated": False,
                "shot_boundaries_supplied": sorted(boundaries),
            },
        }


def _frame_consensus(
    observations: Sequence[MovementObservation],
    config: MotionCandidateConfig,
) -> dict[str, tuple[_FramePoint, ...]]:
    grouped: dict[tuple[str, int], list[MovementObservation]] = defaultdict(list)
    for item in observations:
        grouped[(item.persistent_id, item.frame_index)].append(item)
    by_entity: dict[str, list[_FramePoint]] = defaultdict(list)
    for (persistent_id, frame_index), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (item.timestamp, item.track_id or "", item.source_track_ids))
        timestamp = median(item.timestamp for item in ordered)
        center = (
            median(item.center_normalized[0] for item in ordered),
            median(item.center_normalized[1] for item in ordered),
        )
        spread = max(_distance(item.center_normalized, center) for item in ordered)
        widths = [item.bbox_size_normalized[0] for item in ordered if item.bbox_size_normalized is not None]
        heights = [item.bbox_size_normalized[1] for item in ordered if item.bbox_size_normalized is not None]
        zone_counts = Counter(item.zone for item in ordered if item.zone)
        zone = sorted(zone_counts, key=lambda value: (-zone_counts[value], value))[0] if zone_counts else "unknown"
        track_ids = sorted(
            {
                track_id
                for item in ordered
                for track_id in ((*item.source_track_ids, item.track_id) if item.track_id else item.source_track_ids)
                if track_id
            }
        )
        bboxes = tuple(
            item.bbox for item in ordered if item.bbox is not None
        )
        confidences = tuple(item.confidence for item in ordered if item.confidence is not None)
        refs = tuple(
            {
                "frame_index": item.frame_index,
                "timestamp": item.timestamp,
                "track_id": item.track_id,
                "source_track_ids": list(item.source_track_ids),
                "center_normalized": list(item.center_normalized),
                "bbox": list(item.bbox) if item.bbox is not None else None,
                "bbox_size_normalized": list(item.bbox_size_normalized) if item.bbox_size_normalized is not None else None,
                "confidence": item.confidence,
            }
            for item in ordered
        )
        by_entity[persistent_id].append(
            _FramePoint(
                persistent_id=persistent_id,
                frame_index=frame_index,
                timestamp=float(timestamp),
                center=(float(center[0]), float(center[1])),
                zone=zone,
                track_ids=tuple(track_ids),
                bboxes=bboxes,
                confidences=confidences,
                raw_count=len(ordered),
                duplicate_spread=float(spread),
                duplicate_geometry_conflict=(
                    len(ordered) > 1
                    and spread > config.maximum_same_frame_duplicate_spread_normalized
                ),
                bbox_width_normalized=float(median(widths)) if widths else None,
                bbox_height_normalized=float(median(heights)) if heights else None,
                observation_refs=refs,
            )
        )
    return {
        pid: tuple(sorted(points, key=lambda item: (item.frame_index, item.timestamp)))
        for pid, points in by_entity.items()
    }


def _continuous_candidates(
    points_by_entity: Mapping[str, Sequence[_FramePoint]], config: MotionCandidateConfig
) -> list[_Candidate]:
    result: list[_Candidate] = []
    size = config.window_observation_count
    for persistent_id, points in sorted(points_by_entity.items()):
        runs = _split_runs(points, config.max_observation_gap_seconds)
        run_states: list[list[_PositionWindow]] = []
        for run in runs:
            states = _stable_states(run, config)
            run_states.append(states)
            for before_state, after_state in zip(states, states[1:]):
                before = _tail_window(before_state.points, size)
                after = _head_window(after_state.points, size)
                if before.points and after.points and _distance(before.center, after.center) > config.minimum_displacement_normalized:
                    result.append(_Candidate(persistent_id, "CONTINUOUS_DISPLACEMENT", before, after))
            if not states and len(run) >= 2 * size:
                before = _position_window(tuple(run[:size]))
                after = _position_window(tuple(run[-size:]))
                if _distance(before.center, after.center) > config.minimum_displacement_normalized:
                    result.append(_Candidate(persistent_id, "CONTINUOUS_DISPLACEMENT", before, after))
        for left_run, right_run in zip(runs, runs[1:]):
            before = _tail_window(left_run, size)
            after = _head_window(right_run, size)
            if before.points and after.points and _distance(before.center, after.center) > config.minimum_displacement_normalized:
                result.append(_Candidate(persistent_id, "CONTINUOUS_DISPLACEMENT", before, after))
    return result


def _split_runs(
    points: Sequence[_FramePoint], max_gap: float
) -> list[tuple[_FramePoint, ...]]:
    if not points:
        return []
    result: list[list[_FramePoint]] = [[points[0]]
    ]
    for item in points[1:]:
        if item.timestamp - result[-1][-1].timestamp > max_gap:
            result.append([])
        result[-1].append(item)
    return [tuple(run) for run in result]


def _stable_states(
    run: Sequence[_FramePoint], config: MotionCandidateConfig
) -> list[_PositionWindow]:
    size = config.window_observation_count
    windows = [
        _position_window(tuple(run[index : index + size]))
        for index in range(max(0, len(run) - size + 1))
    ]
    stable = [
        item for item in windows
        if item.radius <= config.maximum_window_radius_normalized
    ]
    states: list[list[_FramePoint]] = []
    for window in stable:
        if states:
            existing_frames = {item.frame_index for item in states[-1]}
            overlap = existing_frames.intersection(window.frames)
            combined = _unique_points((*states[-1], *window.points))
            combined_window = _position_window(combined)
            if overlap and combined_window.radius <= config.maximum_window_radius_normalized:
                states[-1] = list(combined)
                continue
        states.append(list(window.points))
    return [_position_window(tuple(items)) for items in states]


def _position_window(points: tuple[_FramePoint, ...]) -> _PositionWindow:
    if not points:
        return _PositionWindow((), (0.0, 0.0), math.inf)
    center = (
        median(item.center[0] for item in points),
        median(item.center[1] for item in points),
    )
    radius = max(_distance(item.center, center) for item in points)
    return _PositionWindow(points, (float(center[0]), float(center[1])), float(radius))


def _unique_points(points: Iterable[_FramePoint]) -> tuple[_FramePoint, ...]:
    by_frame: dict[int, _FramePoint] = {}
    for item in points:
        by_frame[item.frame_index] = item
    return tuple(by_frame[index] for index in sorted(by_frame))


def _tail_window(points: Sequence[_FramePoint], size: int) -> _PositionWindow:
    return _position_window(tuple(points[-size:])) if points else _position_window(())


def _head_window(points: Sequence[_FramePoint], size: int) -> _PositionWindow:
    return _position_window(tuple(points[:size])) if points else _position_window(())


def _zone_candidate(
    event: EventRecord,
    points: Sequence[_FramePoint],
    config: MotionCandidateConfig,
) -> _Candidate | None:
    if len(event.evidence) < 2:
        return None
    before_frames = event.evidence[0].frame_indices
    after_frames = event.evidence[1].frame_indices
    if not before_frames or not after_frames:
        return None
    before_boundary = max(before_frames)
    after_boundary = min(after_frames)
    before_points = [item for item in points if item.frame_index <= before_boundary]
    after_points = [item for item in points if item.frame_index >= after_boundary]
    if not before_points or not after_points:
        return None
    # Zone transitions have explicit boundary evidence. Two observed points on
    # each side let an actively moving object pass without treating a single
    # box as a stable state. The shared evidence gate still checks spread,
    # duplicates, direct gaps, identity, and displacement.
    boundary_size = min(2, config.window_observation_count)
    return _Candidate(
        event.object_ids[0],
        "ZONE_CHANGE",
        _tail_window(before_points, boundary_size),
        _head_window(after_points, boundary_size),
        zone_before=event.zone_before,
        zone_after=event.zone_after,
        source_segment_ids=event.source_segment_ids,
        source_event_id=event.event_id,
    )


def _empty_zone_candidate(
    event: EventRecord,
    persistent_id: str,
    points: Sequence[_FramePoint],
    config: MotionCandidateConfig,
) -> _Candidate:
    if points:
        sample = _position_window((points[-1],))
        empty = _position_window(())
        return _Candidate(persistent_id, "ZONE_CHANGE", sample, empty, event.zone_before, event.zone_after, event.source_segment_ids, event.event_id)
    empty = _position_window(())
    return _Candidate(persistent_id, "ZONE_CHANGE", empty, empty, event.zone_before, event.zone_after, event.source_segment_ids, event.event_id)


def _assess_candidate(
    candidate: _Candidate,
    entity_points: Sequence[_FramePoint],
    all_points: Mapping[str, Sequence[_FramePoint]],
    video_id: str,
    identity: Mapping[str, Any],
    config: MotionCandidateConfig,
    shot_boundaries: set[int],
    camera_estimator: VideoCameraMotionEstimator | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before, after = candidate.before, candidate.after
    candidate_id = (
        f"{candidate.candidate_type.lower()}:{candidate.persistent_id}:"
        f"{before.first.frame_index if before.points else 'missing'}-"
        f"{after.last.frame_index if after.points else 'missing'}"
    )
    diagnostic: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_type": candidate.candidate_type,
        "effective_entity_id": candidate.persistent_id,
        "source_event_id": candidate.source_event_id,
        "trigger_provenance": [candidate.candidate_type],
        "status": "REJECTED",
        "reason": "UNASSESSED",
        "start_time": before.first.timestamp if before.points else None,
        "end_time": after.last.timestamp if after.points else None,
        "start_evidence": _window_record(before),
        "end_evidence": _window_record(after),
        "position_before": list(before.center) if before.points else None,
        "position_after": list(after.center) if after.points else None,
        "displacement_normalized": _distance(before.center, after.center) if before.points and after.points else None,
        "window_radii_normalized": {"before": before.radius, "after": after.radius},
        "observation_frames": sorted(set(before.frames + after.frames)),
        "duplicate_frame_count": sum(1 for item in (*before.points, *after.points) if item.raw_count > 1),
        "identity_provenance": dict(identity),
        "scene_continuity_status": "UNVERIFIED_NO_RELIABLE_SHOT_BOUNDARY_ARTIFACT",
        "gates": {},
    }
    entity_class = _entity_class(candidate.persistent_id)
    diagnostic["entity_class"] = entity_class
    if entity_class in config.excluded_entity_classes:
        return _finish(diagnostic, "REJECTED", "NON_OBJECT_ENTITY_CLASS"), None
    if len(before.points) < config.window_observation_count or len(after.points) < config.window_observation_count:
        if candidate.candidate_type == "ZONE_CHANGE" and len(before.points) >= 2 and len(after.points) >= 2:
            pass
        else:
            return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_WINDOW_SUPPORT"), None

    span_start = before.first.frame_index
    span_end = after.last.frame_index
    crossed = sorted(frame for frame in shot_boundaries if span_start < frame <= span_end)
    diagnostic["gates"]["shot_boundary"] = {"passed": not crossed, "crossed_frames": crossed}
    if crossed:
        diagnostic["scene_continuity_status"] = "EXPLICIT_BOUNDARY_CROSSED"
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "SHOT_BOUNDARY_CROSSED"), None

    conflicts = sorted(
        item.frame_index
        for item in entity_points
        if span_start <= item.frame_index <= span_end and item.duplicate_geometry_conflict
    )
    diagnostic["gates"]["duplicate_geometry"] = {
        "passed": not conflicts,
        "conflicting_frames": conflicts,
        "maximum_spread_normalized": config.maximum_same_frame_duplicate_spread_normalized,
        "policy": "FAIL_CLOSED_WHEN_SAME_ENTITY_BOXES_DISAGREE_WITHIN_FRAME",
    }
    if conflicts:
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "AMBIGUOUS_DUPLICATE_GEOMETRY"), None

    support_by_frame = {
        item.frame_index: item
        for item in (*before.points, *after.points)
    }
    support_by_frame.update(
        {
            item.frame_index: item
            for item in entity_points
            if before.first.frame_index <= item.frame_index <= after.last.frame_index
        }
    )
    support_points = [support_by_frame[index] for index in sorted(support_by_frame)]
    max_gap = max(
        (right.timestamp - left.timestamp for left, right in zip(support_points, support_points[1:])),
        default=0.0,
    )
    direct_gap = after.first.timestamp - before.last.timestamp
    temporal_passed = max_gap <= config.max_observation_gap_seconds
    diagnostic["gap_seconds"] = round(max_gap, 9)
    diagnostic["support_boundary_gap_seconds"] = round(direct_gap, 9)
    diagnostic["gates"]["temporal_continuity"] = {
        "passed": temporal_passed,
        "maximum_direct_observation_gap_seconds": max_gap,
        "configured_maximum_seconds": config.max_observation_gap_seconds,
        "no_interpolation": True,
    }
    if not temporal_passed:
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "TEMPORAL_GAP_TOO_LARGE"), None

    maximum_width = max(
        (value for value in (before.maximum_bbox_width, after.maximum_bbox_width) if value is not None),
        default=None,
    )
    bbox_localized = maximum_width is None or maximum_width <= config.maximum_bbox_width_normalized
    diagnostic["gates"]["bbox_localization"] = {
        "passed": bbox_localized,
        "maximum_observed_width_normalized": maximum_width,
        "maximum_supported_width_normalized": config.maximum_bbox_width_normalized,
        "position_scale_evidence": {
            "before_widths": [item.bbox_width_normalized for item in before.points],
            "after_widths": [item.bbox_width_normalized for item in after.points],
            "before_heights": [item.bbox_height_normalized for item in before.points],
            "after_heights": [item.bbox_height_normalized for item in after.points],
        },
    }
    if not bbox_localized:
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "BBOX_LOCALIZATION_TOO_BROAD"), None

    bbox_before = before.median_bbox_size
    bbox_after = after.median_bbox_size
    scale_ratios: dict[str, float | None] = {}
    invalid_size_axes: list[str] = []
    for axis, left, right in zip(("width", "height"), bbox_before, bbox_after):
        if left is None or right is None:
            scale_ratios[axis] = None
        elif min(left, right) <= 1e-9:
            scale_ratios[axis] = None
            invalid_size_axes.append(axis)
        else:
            scale_ratios[axis] = max(left / right, right / left)
    measured_scale_ratios = [value for value in scale_ratios.values() if value is not None]
    maximum_scale_ratio = max(measured_scale_ratios, default=None)
    scale_consistent = (
        not invalid_size_axes
        and (maximum_scale_ratio is None or maximum_scale_ratio <= config.maximum_bbox_scale_ratio)
    )
    diagnostic["gates"]["bbox_scale_consistency"] = {
        "passed": scale_consistent,
        "median_size_before_normalized": list(bbox_before),
        "median_size_after_normalized": list(bbox_after),
        "axis_scale_ratios": scale_ratios,
        "invalid_size_axes": invalid_size_axes,
        "maximum_axis_scale_ratio": maximum_scale_ratio,
        "maximum_supported_axis_scale_ratio": config.maximum_bbox_scale_ratio,
    }
    if not scale_consistent:
        reason = "BBOX_SIZE_INVALID" if invalid_size_axes else "BBOX_SCALE_CHANGE_TOO_LARGE"
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", reason), None

    stable_before = before.radius <= config.maximum_window_radius_normalized
    stable_after = after.radius <= config.maximum_window_radius_normalized
    diagnostic["gates"]["stable_windows"] = {
        "passed": stable_before and stable_after,
        "before_radius": before.radius,
        "after_radius": after.radius,
        "maximum_radius": config.maximum_window_radius_normalized,
        "before_observation_count": len(before.points),
        "after_observation_count": len(after.points),
    }
    if not stable_before or not stable_after:
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "POSITION_WINDOW_NOT_STABLE"), None

    track_overlap = sorted(before.track_ids.intersection(after.track_ids))
    identity_verified = bool(track_overlap) or identity.get("identity_provenance") in {
        "HUMAN_CONFIRMED",
        "HUMAN_SUPPORTED_EVALUATION",
    }
    diagnostic["gates"]["identity_continuity"] = {
        "passed": identity_verified,
        "shared_raw_track_ids": track_overlap,
        "identity_provenance": identity.get("identity_provenance"),
        "source_identity_review_status": identity.get("source_identity_review_status"),
    }
    if not identity_verified:
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "IDENTITY_CONTINUITY_UNVERIFIED"), None

    raw_vector = (after.center[0] - before.center[0], after.center[1] - before.center[1])
    raw_displacement = _distance(after.center, before.center)
    scene_record, scene_transform = _estimate_scene_transform(
        candidate.persistent_id, before, after, all_points, config, camera_estimator
    )
    diagnostic["gates"]["scene_compensation"] = scene_record
    visual_estimate_missing = (
        camera_estimator is None
        or scene_record.get("video_motion", {}).get("status") != "ESTIMATED"
    )
    scene_verified = (
        scene_record.get("source") == "VIDEO_FEATURES"
        or scene_record["anchor_count"] >= config.suspected_scene_min_entities
    )
    if not scene_verified and visual_estimate_missing:
        diagnostic["scene_continuity_status"] = "UNVERIFIED_TOO_FEW_COMMON_TRACK_ANCHORS"
        return _finish(diagnostic, "INSUFFICIENT_EVIDENCE", "SCENE_CONTINUITY_UNVERIFIED"), None

    camera_predicted_after = _apply_normalized_transform(before.center, scene_transform)
    effective_vector = (
        after.center[0] - camera_predicted_after[0],
        after.center[1] - camera_predicted_after[1],
    )
    displacement = _distance(effective_vector, (0.0, 0.0))
    separation_margin = config.minimum_displacement_normalized + before.radius + after.radius
    displacement_passed = displacement > separation_margin
    diagnostic["raw_displacement_normalized"] = raw_displacement
    diagnostic["scene_translation_normalized"] = [scene_transform[0][2], scene_transform[1][2]]
    diagnostic["compensated_displacement_normalized"] = displacement
    diagnostic["gates"]["material_displacement"] = {
        "passed": displacement_passed,
        "raw_displacement_normalized": raw_displacement,
        "compensated_displacement_normalized": displacement,
        "minimum_plus_window_radii": separation_margin,
        "minimum_displacement_normalized": config.minimum_displacement_normalized,
    }
    if not displacement_passed:
        reason = (
            "SCENE_SHIFT_EXPLAINS_DISPLACEMENT"
            if raw_displacement > separation_margin and scene_record["compensation_applied"]
            else "DISPLACEMENT_NOT_MATERIAL"
        )
        return _finish(diagnostic, "REJECTED", reason), None

    if scene_record.get("source") == "VIDEO_FEATURES":
        diagnostic["scene_continuity_status"] = "VIDEO_FEATURE_MOTION_COMPENSATED"
    elif scene_record["compensation_applied"]:
        diagnostic["scene_continuity_status"] = "COMMON_TRACK_TRANSLATION_COMPENSATED"
    elif scene_record["anchor_count"] >= config.suspected_scene_min_entities:
        diagnostic["scene_continuity_status"] = "COMMON_TRACKS_SHOW_NO_COHERENT_GLOBAL_TRANSLATION"
    else:
        diagnostic["scene_continuity_status"] = "ZONE_TRIGGER_WITH_UNVERIFIED_SCENE_CONTINUITY"

    event_record = _event_record(candidate, video_id, identity, displacement, diagnostic, effective_vector)
    diagnostic["event_id"] = event_record["event_id"]
    diagnostic = _finish(diagnostic, "ACCEPTED", "ACCEPTED")
    return diagnostic, event_record


def _estimate_scene_transform(
    persistent_id: str,
    before: _PositionWindow,
    after: _PositionWindow,
    all_points: Mapping[str, Sequence[_FramePoint]],
    config: MotionCandidateConfig,
    camera_estimator: VideoCameraMotionEstimator | None,
) -> tuple[dict[str, Any], list[list[float]]]:
    identity_transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    if not before.points or not after.points:
        return {"anchor_count": 0, "compensation_applied": False, "reason": "missing_support"}, identity_transform

    video_motion: dict[str, Any] | None = None
    if camera_estimator is not None:
        video_motion = _estimate_video_motion(camera_estimator, before, after)
        if video_motion.get("status") == "ESTIMATED":
            return {
                "passed": True,
                "reason": "video_feature_transform_estimated",
                "source": "VIDEO_FEATURES",
                "anchor_count": 0,
                "coherent_translation": True,
                "compensation_applied": True,
                "video_motion": video_motion,
                "transform_normalized": video_motion["transform_normalized"],
            }, video_motion["transform_normalized"]
    shifts: list[tuple[float, float]] = []
    anchor_records: list[dict[str, Any]] = []
    for other_id, points in all_points.items():
        if other_id == persistent_id:
            continue
        left_items = [
            item for item in points
            if before.first.timestamp - 0.10 <= item.timestamp <= before.last.timestamp + 0.10
            and not item.duplicate_geometry_conflict
        ]
        right_items = [
            item for item in points
            if after.first.timestamp - 0.10 <= item.timestamp <= after.last.timestamp + 0.10
            and not item.duplicate_geometry_conflict
        ]
        if not left_items or not right_items:
            continue
        shared_tracks = set().union(*(item.track_ids for item in left_items)) & set().union(*(item.track_ids for item in right_items))
        if not shared_tracks:
            continue
        left_points = [item for item in left_items if set(item.track_ids).intersection(shared_tracks)]
        right_points = [item for item in right_items if set(item.track_ids).intersection(shared_tracks)]
        left = (
            median(item.center[0] for item in left_points),
            median(item.center[1] for item in left_points),
        )
        right = (
            median(item.center[0] for item in right_points),
            median(item.center[1] for item in right_points),
        )
        shift = (right[0] - left[0], right[1] - left[1])
        shifts.append(shift)
        anchor_records.append({"entity_id": other_id, "shared_track_ids": sorted(shared_tracks), "shift_normalized": list(shift)})
    if len(shifts) < config.suspected_scene_min_entities:
        return {
            "passed": False,
            "reason": "too_few_common_track_anchors",
            "anchor_count": len(shifts),
            "minimum_anchor_count": config.suspected_scene_min_entities,
            "anchors": anchor_records,
            "coherent_translation": False,
            "compensation_applied": False,
            "translation_normalized": [0.0, 0.0],
            "transform_normalized": identity_transform,
            "video_motion": video_motion or {"status": "NOT_REQUESTED"},
        }, identity_transform
    median_shift = (median(item[0] for item in shifts), median(item[1] for item in shifts))
    median_magnitude = _distance(median_shift, (0.0, 0.0))
    residuals = [_distance(shift, median_shift) for shift in shifts]
    median_residual = median(residuals)
    coherent = (
        median_magnitude >= config.minimum_displacement_normalized
        and median_residual <= config.maximum_scene_anchor_residual_normalized
    )
    transform = [
        [1.0, 0.0, median_shift[0]],
        [0.0, 1.0, median_shift[1]],
    ] if coherent else identity_transform
    record = {
        "passed": True,
        "reason": "coherent_anchor_translation" if coherent else "no_coherent_anchor_translation",
        "anchor_count": len(shifts),
        "minimum_anchor_count": config.suspected_scene_min_entities,
        "anchors": anchor_records,
        "translation_normalized": list(median_shift),
        "translation_magnitude_normalized": median_magnitude,
        "median_anchor_residual_normalized": median_residual,
        "maximum_anchor_residual_normalized": config.maximum_scene_anchor_residual_normalized,
        "coherent_translation": coherent,
        "compensation_applied": coherent,
        "source": "COMMON_TRACK_ANCHORS" if coherent else "COMMON_TRACKS",
        "transform_normalized": transform,
        "video_motion": video_motion or {"status": "NOT_REQUESTED"},
    }
    return record, transform


def _estimate_video_motion(
    estimator: VideoCameraMotionEstimator,
    before: _PositionWindow,
    after: _PositionWindow,
) -> dict[str, Any]:
    before_point = min(before.points, key=lambda item: abs(item.frame_index - median(before.frames)))
    after_point = min(after.points, key=lambda item: abs(item.frame_index - median(after.frames)))
    return estimator.estimate(
        before_point.frame_index,
        after_point.frame_index,
        before_bboxes=before_point.bboxes,
        after_bboxes=after_point.bboxes,
    )


def _apply_normalized_transform(
    point: tuple[float, float], transform: Sequence[Sequence[float]]
) -> tuple[float, float]:
    return (
        transform[0][0] * point[0] + transform[0][1] * point[1] + transform[0][2],
        transform[1][0] * point[0] + transform[1][1] * point[1] + transform[1][2],
    )


def _event_record(
    candidate: _Candidate,
    video_id: str,
    identity: Mapping[str, Any],
    displacement: float,
    diagnostic: Mapping[str, Any],
    effective_vector: tuple[float, float],
) -> dict[str, Any]:
    before, after = candidate.before, candidate.after
    start_time = before.last.timestamp
    end_time = after.first.timestamp
    start_frames = before.frames
    end_frames = after.frames
    event_id = (
        f"move-object:{video_id}:{candidate.persistent_id}:"
        f"{before.last.frame_index}->{after.first.frame_index}"
    )
    trigger = [candidate.candidate_type]
    event_zone_before = candidate.zone_before if candidate.zone_before is not None else before.zone
    event_zone_after = candidate.zone_after if candidate.zone_after is not None else after.zone
    evidence = [
        {
            "video_id": video_id,
            "timestamp": start_time,
            "start_time": before.first.timestamp,
            "end_time": before.last.timestamp,
            "frame_indices": start_frames,
            "segment_ref": f"{video_id}#motion-start={before.first.frame_index}-{before.last.frame_index}",
        },
        {
            "video_id": video_id,
            "timestamp": end_time,
            "start_time": after.first.timestamp,
            "end_time": after.last.timestamp,
            "frame_indices": end_frames,
            "segment_ref": f"{video_id}#motion-end={after.first.frame_index}-{after.last.frame_index}",
        },
    ]
    observations = [
        ref
        for point in (*before.points, *after.points)
        for ref in point.observation_refs
    ]
    return {
        "event_id": event_id,
        "event_type": EventType.MOVE_OBJECT.value,
        "start_time": start_time,
        "end_time": end_time,
        "person_ids": [],
        "object_ids": [candidate.persistent_id],
        "zone_before": event_zone_before,
        "zone_after": event_zone_after,
        "evidence": evidence,
        "source_segment_ids": list(candidate.source_segment_ids),
        "effective_entity_id": candidate.persistent_id,
        "start_evidence": _window_record(before),
        "end_evidence": _window_record(after),
        "observation_references": observations,
        "position_before": {
            "center_normalized": list(before.center),
            "zone": before.zone,
            "window_radius_normalized": before.radius,
        },
        "position_after": {
            "center_normalized": list(after.center),
            "zone": after.zone,
            "window_radius_normalized": after.radius,
        },
        "displacement": {
            "normalized_l2": displacement,
            "raw_normalized_l2": diagnostic.get("raw_displacement_normalized"),
            "scene_compensated": bool(
                diagnostic.get("gates", {}).get("scene_compensation", {}).get("compensation_applied")
            ),
        },
        "relative_displacement_vector_normalized": list(effective_vector),
        "scene_translation_normalized": diagnostic.get("scene_translation_normalized", [0.0, 0.0]),
        "scene_transform_source": diagnostic.get("gates", {}).get("scene_compensation", {}).get("source", "UNVERIFIED"),
        "scene_compensation_evidence": diagnostic.get("gates", {}).get("scene_compensation", {}),
        "trigger_provenance": trigger,
        "evidence_status": "SUPPORTED_EVENT",
        "identity_provenance": dict(identity),
        "scene_continuity_status": diagnostic.get("scene_continuity_status"),
        "source_event_ids": [candidate.source_event_id] if candidate.source_event_id else [],
    }


def _window_record(window: _PositionWindow) -> dict[str, Any]:
    if not window.points:
        return {"frame_indices": [], "timestamps": [], "center_normalized": None, "radius_normalized": None, "observation_count": 0, "track_ids": [], "bbox_widths_normalized": [], "bbox_heights_normalized": [], "duplicate_geometry_conflict_frames": []}
    return {
        "frame_indices": window.frames,
        "timestamps": [item.timestamp for item in window.points],
        "center_normalized": list(window.center),
        "radius_normalized": window.radius,
        "observation_count": len(window.points),
        "track_ids": sorted(window.track_ids),
        "duplicate_detection_counts": [item.raw_count for item in window.points],
        "same_frame_duplicate_spread_normalized": [item.duplicate_spread for item in window.points],
        "bbox_widths_normalized": [item.bbox_width_normalized for item in window.points],
        "bbox_heights_normalized": [item.bbox_height_normalized for item in window.points],
        "duplicate_geometry_conflict_frames": window.duplicate_geometry_conflicts,
    }


def _entity_class(persistent_id: str) -> str | None:
    prefix, separator, remainder = persistent_id.partition(":")
    if separator and prefix in {"obj", "person", "entity"}:
        return remainder.rsplit(":", 1)[0]
    return None


def _coalesce_motion_episodes(
    events: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Join adjacent same-direction transitions sharing a stable state window."""
    merged: list[dict[str, Any]] = []
    diagnostics_by_event = {item.get("event_id"): item for item in diagnostics if item.get("event_id")}
    coalesced = 0
    for event in events:
        if not merged or not _is_adjacent_same_direction_episode(merged[-1], event):
            merged.append(event)
            continue
        prior = merged[-1]
        root_event_id = prior["event_id"]
        prior_end_frames = set(prior["end_evidence"]["frame_indices"])
        event_start_frames = set(event["start_evidence"]["frame_indices"])
        if not prior_end_frames.intersection(event_start_frames):
            continue
        prior["end_time"] = event["end_time"]
        prior["end_evidence"] = event["end_evidence"]
        prior["position_after"] = event["position_after"]
        prior["zone_after"] = event["zone_after"]
        prior["evidence"][1] = event["evidence"][1]
        prior["source_segment_ids"] = sorted(set(prior.get("source_segment_ids", []) + event.get("source_segment_ids", [])))
        prior["source_event_ids"] = sorted(set(prior.get("source_event_ids", []) + event.get("source_event_ids", [])))
        prior["trigger_provenance"] = sorted(
            set(prior.get("trigger_provenance", []) + event.get("trigger_provenance", [])),
            key=_trigger_order,
        )
        prior["observation_references"] = _unique_observation_references(
            prior["observation_references"] + event["observation_references"]
        )
        first_vector = prior.get("relative_displacement_vector_normalized", [0.0, 0.0])
        next_vector = event.get("relative_displacement_vector_normalized", [0.0, 0.0])
        combined_vector = [first_vector[0] + next_vector[0], first_vector[1] + next_vector[1]]
        prior["relative_displacement_vector_normalized"] = combined_vector
        prior["displacement"]["normalized_l2"] = _distance(tuple(combined_vector), (0.0, 0.0))
        prior.setdefault("coalesced_candidate_event_ids", []).append(event["event_id"])
        current_diag = diagnostics_by_event.get(event["event_id"])
        if current_diag is not None:
            current_diag["reason"] = "COALESCED_INTO_EVENT_EPISODE"
            current_diag["coalesced_into_event_id"] = root_event_id
        coalesced += 1
    return merged, coalesced


def _is_adjacent_same_direction_episode(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if left.get("effective_entity_id") != right.get("effective_entity_id"):
        return False
    if left.get("identity_provenance") != right.get("identity_provenance"):
        return False
    left_end = set(left.get("end_evidence", {}).get("frame_indices", []))
    right_start = set(right.get("start_evidence", {}).get("frame_indices", []))
    if not left_end.intersection(right_start):
        return False
    a = left.get("relative_displacement_vector_normalized", [0.0, 0.0])
    b = right.get("relative_displacement_vector_normalized", [0.0, 0.0])
    magnitude_a = _distance(tuple(a), (0.0, 0.0))
    magnitude_b = _distance(tuple(b), (0.0, 0.0))
    if magnitude_a == 0 or magnitude_b == 0:
        return False
    cosine = (a[0] * b[0] + a[1] * b[1]) / (magnitude_a * magnitude_b)
    return cosine > 0


def _unique_observation_references(references: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in references:
        bbox = item.get("bbox")
        key = (
            item.get("frame_index"),
            item.get("timestamp"),
            item.get("track_id"),
            tuple(bbox) if bbox is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _identity_record(
    persistent_id: str,
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supplied = dict(provenance.get(persistent_id, {}))
    return {
        "identity_provenance": supplied.get("identity_provenance", "AUTOMATIC_SOURCE"),
        "source_identity_review_status": supplied.get("source_identity_review_status", "NOT_RECORDED"),
        "source_entity_ids": supplied.get("source_entity_ids", [persistent_id]),
        "human_confirmation_operation_ids": supplied.get("human_confirmation_operation_ids", []),
        "overlay_revision": supplied.get("overlay_revision"),
    }


def _same_supported_transition(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("effective_entity_id") != right.get("effective_entity_id"):
        return False
    left_start = set(left["start_evidence"]["frame_indices"])
    right_start = set(right["start_evidence"]["frame_indices"])
    left_end = set(left["end_evidence"]["frame_indices"])
    right_end = set(right["end_evidence"]["frame_indices"])
    return bool(left_start.intersection(right_start) and left_end.intersection(right_end))


def _finish(record: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    record["status"] = status
    record["reason"] = reason
    record["evidence_status"] = "SUPPORTED_EVENT" if status == "ACCEPTED" else status
    return record


def _trigger_order(value: str) -> int:
    return {"ZONE_CHANGE": 0, "CONTINUOUS_DISPLACEMENT": 1}.get(value, 2)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])
