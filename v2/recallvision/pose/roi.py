"""Persistent-person ROI geometry and backend-independent remapping helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.pose.schema import PoseDetection


@dataclass(frozen=True)
class PersonROI:
    """A clipped image-space crop derived from one canonical person bbox."""

    canonical_bbox: BoundingBox
    crop_bbox: tuple[int, int, int, int]
    crop_width: int
    crop_height: int
    margin: float
    clipped: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "canonical_person_bbox": {
                "x1": float(self.canonical_bbox.x1),
                "y1": float(self.canonical_bbox.y1),
                "x2": float(self.canonical_bbox.x2),
                "y2": float(self.canonical_bbox.y2),
            },
            "crop_bbox": list(self.crop_bbox),
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "roi_margin": float(self.margin),
            "clipped": self.clipped,
        }


def compute_person_roi(
    canonical_bbox: BoundingBox,
    frame_width: float,
    frame_height: float,
    *,
    margin: float = 0.0,
) -> PersonROI:
    """Compute a deterministic crop from a canonical person bbox.

    ``margin`` is a fraction of the bbox width/height. Production R1 uses
    zero; nonzero values remain available to diagnostic tests only.
    """

    frame_width = float(frame_width)
    frame_height = float(frame_height)
    margin = float(margin)
    if not math.isfinite(frame_width) or frame_width <= 0:
        raise ValueError("frame_width must be positive and finite")
    if not math.isfinite(frame_height) or frame_height <= 0:
        raise ValueError("frame_height must be positive and finite")
    if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be finite and in [0, 1]")
    width = float(canonical_bbox.x2) - float(canonical_bbox.x1)
    height = float(canonical_bbox.y2) - float(canonical_bbox.y1)
    if width <= 0 or height <= 0:
        raise ValueError("canonical person bbox must have positive width and height")
    raw = (
        float(canonical_bbox.x1) - width * margin,
        float(canonical_bbox.y1) - height * margin,
        float(canonical_bbox.x2) + width * margin,
        float(canonical_bbox.y2) + height * margin,
    )
    clipped = raw[0] < 0.0 or raw[1] < 0.0 or raw[2] > frame_width or raw[3] > frame_height
    x0 = max(0, math.floor(raw[0]))
    y0 = max(0, math.floor(raw[1]))
    x1 = min(int(frame_width), math.ceil(raw[2]))
    y1 = min(int(frame_height), math.ceil(raw[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("canonical person bbox produces an empty clipped ROI")
    return PersonROI(
        canonical_bbox=canonical_bbox,
        crop_bbox=(x0, y0, x1, y1),
        crop_width=x1 - x0,
        crop_height=y1 - y0,
        margin=margin,
        clipped=clipped,
    )


def remap_pose_detection(
    detection: PoseDetection,
    roi: PersonROI,
    *,
    persistent_person_id: str | None = None,
) -> PoseDetection:
    """Map local ROI pose coordinates into global full-frame coordinates."""

    x0, y0, _, _ = roi.crop_bbox
    global_bbox = BoundingBox(
        detection.pose_bbox.x1 + x0,
        detection.pose_bbox.y1 + y0,
        detection.pose_bbox.x2 + x0,
        detection.pose_bbox.y2 + y0,
    )
    global_keypoints = tuple(
        type(keypoint)(
            keypoint.name,
            keypoint.x + x0,
            keypoint.y + y0,
            keypoint.confidence,
            keypoint.visibility,
        )
        for keypoint in detection.keypoints
    )
    provenance = {
        "pose_inference_policy": "PERSISTENT_PERSON_ROI",
        **roi.to_record(),
        "local_pose_bbox": {
            "x1": float(detection.pose_bbox.x1),
            "y1": float(detection.pose_bbox.y1),
            "x2": float(detection.pose_bbox.x2),
            "y2": float(detection.pose_bbox.y2),
        },
        "local_keypoints": [item.to_record() for item in sorted(detection.keypoints, key=lambda item: item.name)],
    }
    if persistent_person_id is not None:
        provenance["persistent_person_id"] = persistent_person_id
    return type(detection)(
        video_id=detection.video_id,
        frame_index=detection.frame_index,
        timestamp=detection.timestamp,
        detection_index=detection.detection_index,
        pose_bbox=global_bbox,
        keypoints=global_keypoints,
        pose_confidence=detection.pose_confidence,
        inference_policy="PERSISTENT_PERSON_ROI",
        roi_provenance=provenance,
    )


def select_person_roi_observations(
    observations: Iterable[Observation],
    *,
    selected_frames: Iterable[int],
) -> tuple[Observation, ...]:
    """Return only canonical person observations in the event-local frame set."""

    selected = {int(frame) for frame in selected_frames}
    return tuple(
        sorted(
            (
                item
                for item in observations
                if item.entity_type == "person"
                and item.persistent_id is not None
                and item.frame_index in selected
            ),
            key=lambda item: (item.frame_index, str(item.persistent_id), item.track_id),
        )
    )
