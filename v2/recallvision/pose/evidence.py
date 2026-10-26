"""Event-local wrist evidence assembly and descriptive observability summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping

from recallvision.memory.schema import Observation
from recallvision.pose.geometry import (
    WristObjectMeasurement,
    measure_wrist_proximity,
    select_closest_wrist,
)
from recallvision.pose.schema import PersonPoseObservation


@dataclass(frozen=True)
class EvidenceConfig:
    """Explicit backend-score and local-control policy."""

    minimum_keypoint_confidence: float = 0.50
    minimum_keypoint_visibility: float | None = None
    negative_control_horizon_seconds: float = 0.50
    roi_margin: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_keypoint_confidence <= 1.0:
            raise ValueError("minimum_keypoint_confidence must be in [0, 1]")
        if self.minimum_keypoint_visibility is not None and not 0.0 <= self.minimum_keypoint_visibility <= 1.0:
            raise ValueError("minimum_keypoint_visibility must be in [0, 1]")
        if not math.isfinite(self.negative_control_horizon_seconds) or self.negative_control_horizon_seconds < 0:
            raise ValueError("negative_control_horizon_seconds must be non-negative and finite")
        if self.roi_margin != 0.0:
            raise ValueError("roi_margin must be 0.0 for the R1 production policy")

    def to_record(self) -> dict[str, float | None]:
        return {
            "minimum_keypoint_confidence": float(self.minimum_keypoint_confidence),
            "minimum_keypoint_visibility": (
                None
                if self.minimum_keypoint_visibility is None
                else float(self.minimum_keypoint_visibility)
            ),
            "negative_control_horizon_seconds": float(self.negative_control_horizon_seconds),
            "roi_margin": 0.0,
            "missing_keypoint_scores_are_invalid": True,
        }


@dataclass(frozen=True)
class EventFrameWindow:
    """Frame sets used by E1, all derived from structured event/observations."""

    event_frames: tuple[int, ...]
    pre_control_frames: tuple[int, ...]
    post_control_frames: tuple[int, ...]
    boundary_frames: tuple[int, ...]
    support_horizon_seconds: float

    @property
    def all_frames(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(self.event_frames)
                | set(self.pre_control_frames)
                | set(self.post_control_frames)
            )
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "event_frames": list(self.event_frames),
            "pre_control_frames": list(self.pre_control_frames),
            "post_control_frames": list(self.post_control_frames),
            "boundary_frames": list(self.boundary_frames),
            "all_frames": list(self.all_frames),
            "policy": "P2D_BOUNDARY_LOCAL_SUPPORT_PLUS_0.50S_CONTROLS",
            "support_horizon_seconds": float(self.support_horizon_seconds),
            "interpolation_used": False,
        }


def _integer_frames(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_name} must be a list of integers")
    return tuple(sorted(set(value)))


def _event_boundary_frames(event: Mapping[str, Any]) -> tuple[int, ...]:
    frames: set[int] = set()
    evidence = event.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("event evidence must be a list")
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("event evidence item must be an object")
        frames.update(_integer_frames(item.get("frame_indices"), "event evidence frame_indices"))
    return tuple(sorted(frames))


def select_event_local_frames(
    event: Mapping[str, Any],
    event_diagnostic: Mapping[str, Any],
    observations: Iterable[Observation],
    *,
    horizon_seconds: float = 0.50,
) -> EventFrameWindow:
    """Select event/support/control frames without development-video constants."""

    if horizon_seconds < 0 or not math.isfinite(horizon_seconds):
        raise ValueError("horizon_seconds must be non-negative and finite")
    start_time = float(event["start_time"])
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id is required")
    object_ids = event.get("object_ids")
    if not isinstance(object_ids, list) or len(object_ids) != 1 or not isinstance(object_ids[0], str):
        raise ValueError("E1 requires exactly one object_id")
    object_id = object_ids[0]
    support_frames = _integer_frames(event_diagnostic.get("left_support_frames"), "left_support_frames") + _integer_frames(
        event_diagnostic.get("right_support_frames"), "right_support_frames"
    )
    boundary_frames = _event_boundary_frames(event)
    event_frames = tuple(sorted(set(support_frames) | set(boundary_frames)))
    source_observations = tuple(observations)
    object_observations = tuple(
        item for item in source_observations if item.persistent_id == object_id and item.entity_type == "object"
    )
    observed_event_times = [item.timestamp for item in object_observations if item.frame_index in event_frames]
    right_support_timestamps = event_diagnostic.get("right_support_timestamps")
    if isinstance(right_support_timestamps, list) and right_support_timestamps:
        support_end_time = max(float(item) for item in right_support_timestamps)
    elif observed_event_times:
        support_end_time = max(observed_event_times)
    else:
        support_end_time = float(event["end_time"])
    pre_control_frames = tuple(
        sorted(
            {
                item.frame_index
                for item in object_observations
                if start_time - horizon_seconds <= item.timestamp < start_time
            }
        )
    )
    post_control_frames = tuple(
        sorted(
            {
                item.frame_index
                for item in object_observations
                if support_end_time < item.timestamp <= support_end_time + horizon_seconds
            }
        )
    )
    return EventFrameWindow(
        event_frames=event_frames,
        pre_control_frames=pre_control_frames,
        post_control_frames=post_control_frames,
        boundary_frames=boundary_frames,
        support_horizon_seconds=float(horizon_seconds),
    )


def _confidence_summary(measurements: list[WristObjectMeasurement]) -> dict[str, Any]:
    values = [item.confidence for item in measurements]
    return _score_summary(values)


def _score_summary(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def _trend(measurements: list[WristObjectMeasurement]) -> str:
    if len(measurements) < 2:
        return "INSUFFICIENT_VALID_WRIST_FRAMES"
    ordered = measurements
    distances = [item.distance_to_bbox for item in ordered]
    minimum = min(distances)
    first_is_above = distances[0] > minimum
    last_is_above = distances[-1] > minimum
    if first_is_above and last_is_above:
        return "APPROACHING_THEN_SEPARATING"
    if first_is_above:
        return "APPROACHING"
    if last_is_above:
        return "SEPARATING"
    return "NO_CLEAR_TREND"


def _summary_for_frames(
    *,
    person_id: str,
    frames: tuple[int, ...],
    object_by_frame: Mapping[int, Observation],
    person_by_frame: Mapping[int, Observation],
    pose_by_frame: Mapping[int, PersonPoseObservation],
    config: EvidenceConfig,
    frame_width: float,
    frame_height: float,
    boundary_frames: set[int],
) -> dict[str, Any]:
    left_measurements: list[WristObjectMeasurement] = []
    right_measurements: list[WristObjectMeasurement] = []
    left_scores: list[float] = []
    right_scores: list[float] = []
    all_measurements: list[tuple[int, WristObjectMeasurement]] = []
    frame_records: list[dict[str, Any]] = []
    attempted = 0
    missing = 0
    low_confidence = 0
    pose_frames_valid = 0
    for frame_index in frames:
        if frame_index not in person_by_frame or frame_index not in object_by_frame:
            continue
        attempted += 1
        pose = pose_by_frame.get(frame_index)
        object_bbox = object_by_frame[frame_index].bbox
        measurements: list[WristObjectMeasurement] = []
        raw_wrist_present = False
        if pose is not None:
            pose_frames_valid += 1
            for wrist_name, target in (("left_wrist", left_measurements), ("right_wrist", right_measurements)):
                wrist = pose.keypoint(wrist_name)
                if wrist is not None:
                    raw_wrist_present = True
                    if wrist_name == "left_wrist" and wrist.confidence is not None:
                        left_scores.append(float(wrist.confidence))
                    if wrist_name == "right_wrist" and wrist.confidence is not None:
                        right_scores.append(float(wrist.confidence))
                measurement = (
                    None
                    if wrist is None
                    else measure_wrist_proximity(
                        wrist,
                        object_bbox,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        minimum_confidence=config.minimum_keypoint_confidence,
                        minimum_visibility=config.minimum_keypoint_visibility,
                    )
                )
                if measurement is not None:
                    target.append(measurement)
                    measurements.append(measurement)
                    all_measurements.append((frame_index, measurement))
        if not measurements:
            if raw_wrist_present:
                low_confidence += 1
            else:
                missing += 1
        closest = select_closest_wrist(measurements)
        frame_records.append(
            {
                "frame_index": frame_index,
                "timestamp": person_by_frame[frame_index].timestamp,
                "boundary_frame": frame_index in boundary_frames,
                "left_wrist": next((item.to_record() for item in measurements if item.wrist_name == "left_wrist"), None),
                "right_wrist": next((item.to_record() for item in measurements if item.wrist_name == "right_wrist"), None),
                "closest_wrist": None if closest is None else closest.to_record(),
            }
        )
    all_distances = [item.distance_to_bbox for _, item in all_measurements]
    minimum_item = min(
        all_measurements,
        key=lambda item: (item[1].distance_to_bbox, item[0], item[1].wrist_name),
        default=None,
    )
    closest_by_frame = [
        item[1]
        for item in (
            next(
                (
                    candidate
                    for candidate in all_measurements
                    if candidate[0] == frame_index
                    and candidate[1].wrist_name == closest_name
                ),
                None,
            )
            for frame_index, closest_name in (
                (record["frame_index"], record["closest_wrist"]["wrist"])
                for record in frame_records
                if record["closest_wrist"] is not None
            )
        )
        if item is not None
    ]
    return {
        "person_id": person_id,
        "selected_frames": list(frames),
        "frames_attempted": attempted,
        "pose_frames_attempted": attempted,
        "pose_frames_valid": pose_frames_valid,
        "valid_wrist_frames": len(closest_by_frame),
        "missing_wrist_frames": missing,
        "low_confidence_frames": low_confidence,
        "left_wrist_valid_frames": len(left_measurements),
        "right_wrist_valid_frames": len(right_measurements),
        "wrist_confidence_summary": {
            "left_wrist": _score_summary(left_scores),
            "right_wrist": _score_summary(right_scores),
        },
        "minimum_distance": None if not all_distances else min(all_distances),
        "median_distance": None if not all_distances else median(all_distances),
        "minimum_distance_frame": None if minimum_item is None else minimum_item[0],
        "wrist_inside_object_count": sum(item.inside_object_bbox for _, item in all_measurements),
        "proximity_measurement_count": len(all_measurements),
        "approach_pattern": _trend(closest_by_frame),
        "separation_pattern": _trend(list(reversed(closest_by_frame))),
        "measurements": frame_records,
        "interpolation_used": False,
    }


def _identity_ready(identity_status: str | None) -> bool:
    return identity_status in {"PASS_HUMAN_REVIEW_DEV_VIDEO", "PASS"}


def _distinctive(event_summary: Mapping[str, Any], negative_summary: Mapping[str, Any]) -> str:
    event_median = event_summary.get("median_distance")
    negative_median = negative_summary.get("median_distance")
    event_minimum = event_summary.get("minimum_distance")
    negative_minimum = negative_summary.get("minimum_distance")
    if None in (event_median, negative_median, event_minimum, negative_minimum):
        return "UNCERTAIN"
    return "YES" if event_median < negative_median and event_minimum < negative_minimum else "NO"


def _person_level(event_summary: Mapping[str, Any], distinctive: str) -> str:
    if event_summary.get("valid_wrist_frames", 0) == 0:
        return "POSE_LEVEL_0_UNUSABLE"
    if distinctive == "YES" and event_summary.get("wrist_inside_object_count", 0) > 0:
        return "POSE_LEVEL_3_INTERACTION_CANDIDATE"
    if distinctive == "YES":
        return "POSE_LEVEL_2_EVENT_LOCAL_PROXIMITY"
    return "POSE_LEVEL_1_OBSERVABLE"


def build_interaction_evidence(
    *,
    event: Mapping[str, Any],
    event_window: EventFrameWindow,
    observations: Iterable[Observation],
    pose_observations: Iterable[PersonPoseObservation],
    frame_width: float,
    frame_height: float,
    config: EvidenceConfig | None = None,
    association_diagnostics: Iterable[Mapping[str, Any]] = (),
    identity_status: str | None = None,
) -> dict[str, Any]:
    """Build deterministic, descriptive event and no-event wrist evidence."""

    config = config or EvidenceConfig()
    source_observations = tuple(observations)
    object_id = event["object_ids"][0]
    object_by_frame = {
        item.frame_index: item
        for item in source_observations
        if item.persistent_id == object_id and item.entity_type == "object"
    }
    person_observations = tuple(
        item
        for item in source_observations
        if item.entity_type == "person" and item.persistent_id is not None
    )
    pose_by_person_frame = {
        (item.persistent_person_id, item.frame_index): item
        for item in pose_observations
    }
    person_ids = sorted(
        {
            str(item.persistent_id)
            for item in person_observations
            if item.frame_index in event_window.all_frames and item.persistent_id
        }
        | {item.persistent_person_id for item in pose_observations}
    )
    all_diagnostics = tuple(association_diagnostics)
    persons: dict[str, Any] = {}
    for person_id in person_ids:
        person_by_frame = {
            item.frame_index: item
            for item in person_observations
            if item.persistent_id == person_id
        }
        pose_by_frame = {
            frame_index: pose
            for (candidate_id, frame_index), pose in pose_by_person_frame.items()
            if candidate_id == person_id
        }
        event_summary = _summary_for_frames(
            person_id=person_id,
            frames=event_window.event_frames,
            object_by_frame=object_by_frame,
            person_by_frame=person_by_frame,
            pose_by_frame=pose_by_frame,
            config=config,
            frame_width=frame_width,
            frame_height=frame_height,
            boundary_frames=set(event_window.boundary_frames),
        )
        negative_frames = tuple(sorted(set(event_window.pre_control_frames) | set(event_window.post_control_frames)))
        negative_summary = _summary_for_frames(
            person_id=person_id,
            frames=negative_frames,
            object_by_frame=object_by_frame,
            person_by_frame=person_by_frame,
            pose_by_frame=pose_by_frame,
            config=config,
            frame_width=frame_width,
            frame_height=frame_height,
            boundary_frames=set(),
        )
        distinctive = _distinctive(event_summary, negative_summary)
        matched = event_summary["valid_wrist_frames"] + negative_summary["valid_wrist_frames"] > 0
        ambiguous = any(
            item.get("status") == "AMBIGUOUS"
            and any(
                candidate.get("candidate") is True
                and candidate.get("persistent_person_id") == person_id
                for candidate in item.get("candidates", [])
            )
            for item in all_diagnostics
        )
        association_status = "AMBIGUOUS" if ambiguous else ("MATCHED" if matched else "NO_MATCH")
        persons[person_id] = {
            "person_id": person_id,
            "pose_person_association_status": association_status,
            "event_summary": event_summary,
            "negative_control_summary": negative_summary,
            "event_wrist_evidence_distinctive": distinctive,
            "pose_evidence_level": _person_level(event_summary, distinctive),
            "person_identity_ready_for_attribution": _identity_ready(identity_status),
        }
    distinctiveness = [item["event_wrist_evidence_distinctive"] for item in persons.values()]
    if "YES" in distinctiveness:
        overall_distinctive = "YES"
    elif distinctiveness and all(item == "NO" for item in distinctiveness):
        overall_distinctive = "NO"
    else:
        overall_distinctive = "UNCERTAIN"
    level_order = {
        "POSE_LEVEL_0_UNUSABLE": 0,
        "POSE_LEVEL_1_OBSERVABLE": 1,
        "POSE_LEVEL_2_EVENT_LOCAL_PROXIMITY": 2,
        "POSE_LEVEL_3_INTERACTION_CANDIDATE": 3,
    }
    overall_level = max(
        (item["pose_evidence_level"] for item in persons.values()),
        key=lambda value: level_order[value],
        default="POSE_LEVEL_0_UNUSABLE",
    )
    identity_ready = _identity_ready(identity_status)
    return {
        "event_id": event["event_id"],
        "object_id": object_id,
        "candidate_person_ids": person_ids,
        "event_window": event_window.to_record(),
        "config": config.to_record(),
        "persons": persons,
        "event_wrist_evidence_distinctive": overall_distinctive,
        "pose_evidence_level": overall_level,
        "person_identity_status": identity_status,
        "person_identity_ready_for_attribution": identity_ready,
        "can_define_structured_agent_association_rule": "YES"
        if overall_level in {"POSE_LEVEL_2_EVENT_LOCAL_PROXIMITY", "POSE_LEVEL_3_INTERACTION_CANDIDATE"} and identity_ready
        else "NO",
        "can_populate_move_object_person_ids": "NO",
        "can_implement_who_moved_object": "NO",
        "move_object_modified": False,
        "person_ids_populated": False,
        "who_moved_object_implemented": False,
        "causal_manipulation_inferred": False,
        "contact_claimed": False,
    }
