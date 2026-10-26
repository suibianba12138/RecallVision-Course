"""Fail-closed, same-frame association of pose detections to person memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.pose.schema import PersonPoseObservation, PoseDetection


@dataclass(frozen=True)
class AssociationConfig:
    """Generic geometry floors; these are not fitted to a development event."""

    minimum_iou: float = 0.05
    minimum_ios: float = 0.10
    maximum_center_distance: float = 0.35

    def __post_init__(self) -> None:
        for name in ("minimum_iou", "minimum_ios"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.maximum_center_distance) or self.maximum_center_distance < 0:
            raise ValueError("maximum_center_distance must be non-negative and finite")

    def to_record(self) -> dict[str, float]:
        return {
            "minimum_iou": float(self.minimum_iou),
            "minimum_ios": float(self.minimum_ios),
            "maximum_center_distance": float(self.maximum_center_distance),
        }


@dataclass(frozen=True)
class FrameAssociationResult:
    """Structured result retaining both accepted pose records and diagnostics."""

    status: str
    observations: tuple[PersonPoseObservation, ...]
    diagnostics: tuple[dict[str, Any], ...]


def _area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def _intersection(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    overlap = _intersection(first, second)
    union = _area(first) + _area(second) - overlap
    return overlap / union if union else 0.0


def _ios(first: BoundingBox, second: BoundingBox) -> float:
    overlap = _intersection(first, second)
    denominator = min(_area(first), _area(second))
    return overlap / denominator if denominator else 0.0


def _center_distance(
    first: BoundingBox,
    second: BoundingBox,
    *,
    frame_width: float,
    frame_height: float,
) -> float:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    first_center = ((first.x1 + first.x2) / 2.0, (first.y1 + first.y2) / 2.0)
    second_center = ((second.x1 + second.x2) / 2.0, (second.y1 + second.y2) / 2.0)
    return math.hypot(
        (first_center[0] - second_center[0]) / frame_width,
        (first_center[1] - second_center[1]) / frame_height,
    )


def _observation_ref(observation: Observation) -> str:
    return (
        f"{observation.video_id}#frame={observation.frame_index}"
        f"#persistent={observation.persistent_id}#track={observation.track_id}"
    )


def _candidate(
    pose: PoseDetection,
    person: Observation,
    *,
    frame_width: float,
    frame_height: float,
    config: AssociationConfig,
) -> dict[str, Any]:
    iou = _iou(pose.pose_bbox, person.bbox)
    ios = _ios(pose.pose_bbox, person.bbox)
    center_distance = _center_distance(
        pose.pose_bbox,
        person.bbox,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    accepted = (
        iou >= config.minimum_iou
        or ios >= config.minimum_ios
        or center_distance <= config.maximum_center_distance
    )
    return {
        "persistent_person_id": person.persistent_id,
        "source_person_observation_ref": _observation_ref(person),
        "iou": iou,
        "ios": ios,
        "center_distance": center_distance,
        "candidate": accepted,
    }


def associate_frame_poses(
    pose_detections: Iterable[PoseDetection],
    person_observations: Iterable[Observation],
    *,
    frame_width: float,
    frame_height: float,
    pose_backend: str,
    config: AssociationConfig | None = None,
) -> FrameAssociationResult:
    """Associate pose detections to same-frame consolidated person observations.

    No pose detection is linked across frames. A detection with zero or more
    than one geometric candidate is not emitted as a person pose observation.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    config = config or AssociationConfig()
    poses = tuple(sorted(pose_detections, key=lambda item: item.detection_index))
    persons = tuple(
        sorted(
            (
                item
                for item in person_observations
                if item.entity_type == "person" and item.persistent_id
            ),
            key=lambda item: (str(item.persistent_id), item.track_id),
        )
    )
    diagnostics: list[dict[str, Any]] = []
    provisional: list[tuple[PoseDetection, Observation, dict[str, Any]]] = []
    for pose in poses:
        same_frame_persons = [
            person
            for person in persons
            if person.video_id == pose.video_id and person.frame_index == pose.frame_index
        ]
        candidates = [
            _candidate(
                pose,
                person,
                frame_width=frame_width,
                frame_height=frame_height,
                config=config,
            )
            for person in same_frame_persons
        ]
        accepted = [item for item in candidates if item["candidate"]]
        diagnostic = {
            "video_id": pose.video_id,
            "frame_index": pose.frame_index,
            "detection_index": pose.detection_index,
            "candidate_count": len(accepted),
            "candidates": candidates,
            "association_config": config.to_record(),
        }
        if len(accepted) != 1:
            diagnostic["status"] = "NO_MATCH" if not accepted else "AMBIGUOUS"
            diagnostics.append(diagnostic)
            continue
        person = next(item for item in same_frame_persons if item.persistent_id == accepted[0]["persistent_person_id"])
        provisional.append((pose, person, diagnostic))

    persistent_to_items: dict[str, list[int]] = {}
    for index, (_, person, _) in enumerate(provisional):
        assert person.persistent_id is not None
        persistent_to_items.setdefault(person.persistent_id, []).append(index)
    ambiguous_indices = {
        index
        for indices in persistent_to_items.values()
        if len(indices) > 1
        for index in indices
    }
    accepted_observations: list[PersonPoseObservation] = []
    for index, (pose, person, diagnostic) in enumerate(provisional):
        if index in ambiguous_indices:
            diagnostic["status"] = "AMBIGUOUS"
            diagnostic["ambiguity_reason"] = "multiple_pose_detections_match_one_persistent_person"
            diagnostics.append(diagnostic)
            continue
        diagnostic["status"] = "MATCHED"
        accepted_observations.append(
            PersonPoseObservation(
                video_id=pose.video_id,
                frame_index=pose.frame_index,
                timestamp=pose.timestamp,
                persistent_person_id=str(person.persistent_id),
                source_person_observation_ref=_observation_ref(person),
                pose_backend=pose_backend,
                pose_bbox=pose.pose_bbox,
                keypoints=pose.keypoints,
                association_diagnostics=dict(diagnostic),
                inference_policy=pose.inference_policy,
                roi_provenance=pose.roi_provenance,
            )
        )
        diagnostics.append(diagnostic)
    diagnostics.sort(key=lambda item: (item["frame_index"], item["detection_index"], item["status"]))
    if ambiguous_indices or any(item["status"] == "AMBIGUOUS" for item in diagnostics):
        status = "AMBIGUOUS"
    elif accepted_observations:
        status = "MATCHED"
    else:
        status = "NO_MATCH"
    return FrameAssociationResult(
        status=status,
        observations=tuple(
            sorted(accepted_observations, key=lambda item: (item.frame_index, item.persistent_person_id))
        ),
        diagnostics=tuple(diagnostics),
    )
