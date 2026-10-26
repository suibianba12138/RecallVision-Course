"""Stable, backend-independent 2D pose evidence records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from recallvision.memory.schema import BoundingBox


def _finite(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _bbox_record(bbox: BoundingBox) -> dict[str, float]:
    return {
        "x1": float(bbox.x1),
        "y1": float(bbox.y1),
        "x2": float(bbox.x2),
        "y2": float(bbox.y2),
    }


@dataclass(frozen=True)
class KeypointObservation:
    """One named keypoint in image pixels plus model-provided scores."""

    name: str
    x: float
    y: float
    confidence: float | None = None
    visibility: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("keypoint name must be a non-empty string")
        _finite(self.x, "keypoint x")
        _finite(self.y, "keypoint y")
        if self.confidence is not None:
            _finite(self.confidence, "keypoint confidence")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("keypoint confidence must be in [0, 1]")
        if self.visibility is not None:
            _finite(self.visibility, "keypoint visibility")
            if not 0.0 <= self.visibility <= 1.0:
                raise ValueError("keypoint visibility must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
            "confidence": None if self.confidence is None else float(self.confidence),
            "visibility": None if self.visibility is None else float(self.visibility),
        }


@dataclass(frozen=True)
class PoseDetection:
    """One raw pose detection converted out of a backend-specific object."""

    video_id: str
    frame_index: int
    timestamp: float
    detection_index: int
    pose_bbox: BoundingBox
    keypoints: tuple[KeypointObservation, ...]
    pose_confidence: float | None = None
    inference_policy: str = "FULL_FRAME"
    roi_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id is required")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        _finite(self.timestamp, "timestamp")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if isinstance(self.detection_index, bool) or self.detection_index < 0:
            raise ValueError("detection_index must be non-negative")
        if self.pose_bbox.x2 < self.pose_bbox.x1 or self.pose_bbox.y2 < self.pose_bbox.y1:
            raise ValueError("pose_bbox must have non-negative width and height")
        names = [item.name for item in self.keypoints]
        if len(names) != len(set(names)):
            raise ValueError("keypoint names must be unique per pose detection")
        if self.pose_confidence is not None:
            _finite(self.pose_confidence, "pose confidence")
            if not 0.0 <= self.pose_confidence <= 1.0:
                raise ValueError("pose confidence must be in [0, 1]")
        if not self.inference_policy:
            raise ValueError("inference_policy is required")

    def keypoint(self, name: str) -> KeypointObservation | None:
        return next((item for item in self.keypoints if item.name == name), None)

    def to_record(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
            "detection_index": self.detection_index,
            "pose_bbox": _bbox_record(self.pose_bbox),
            "keypoints": [item.to_record() for item in sorted(self.keypoints, key=lambda item: item.name)],
            "pose_confidence": None if self.pose_confidence is None else float(self.pose_confidence),
            "inference_policy": self.inference_policy,
            "roi_provenance": None if self.roi_provenance is None else dict(self.roi_provenance),
        }


@dataclass(frozen=True)
class PersonPoseObservation:
    """A pose detection attached only to a same-frame persistent person observation."""

    video_id: str
    frame_index: int
    timestamp: float
    persistent_person_id: str
    source_person_observation_ref: str
    pose_backend: str
    pose_bbox: BoundingBox
    keypoints: tuple[KeypointObservation, ...]
    association_diagnostics: Mapping[str, Any]
    inference_policy: str = "FULL_FRAME"
    roi_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.video_id or not self.persistent_person_id:
            raise ValueError("video_id and persistent_person_id are required")
        if not self.source_person_observation_ref:
            raise ValueError("source_person_observation_ref is required")
        if not self.pose_backend:
            raise ValueError("pose_backend is required")
        if not self.inference_policy:
            raise ValueError("inference_policy is required")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        _finite(self.timestamp, "timestamp")

    def keypoint(self, name: str) -> KeypointObservation | None:
        return next((item for item in self.keypoints if item.name == name), None)

    def to_record(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
            "persistent_person_id": self.persistent_person_id,
            "source_person_observation_ref": self.source_person_observation_ref,
            "pose_backend": self.pose_backend,
            "pose_bbox": _bbox_record(self.pose_bbox),
            "keypoints": [item.to_record() for item in sorted(self.keypoints, key=lambda item: item.name)],
            "association_diagnostics": dict(self.association_diagnostics),
            "inference_policy": self.inference_policy,
            "roi_provenance": None if self.roi_provenance is None else dict(self.roi_provenance),
        }
