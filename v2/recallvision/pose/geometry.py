"""Deterministic wrist-to-object geometry and fail-closed validity rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from recallvision.memory.schema import BoundingBox
from recallvision.pose.schema import KeypointObservation


def _validate_dimensions(frame_width: float, frame_height: float) -> tuple[float, float]:
    width = float(frame_width)
    height = float(frame_height)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive and finite")
    return width, height


def _validate_bbox(bbox: BoundingBox) -> None:
    if bbox.x2 < bbox.x1 or bbox.y2 < bbox.y1:
        raise ValueError("bbox must have non-negative width and height")


def point_to_bbox_distance(
    point: tuple[float, float],
    bbox: BoundingBox,
    *,
    frame_width: float,
    frame_height: float,
) -> float:
    """Return normalized shortest point-to-rectangle distance."""

    width, height = _validate_dimensions(frame_width, frame_height)
    _validate_bbox(bbox)
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("point coordinates must be finite")
    dx = max(bbox.x1 - x, 0.0, x - bbox.x2)
    dy = max(bbox.y1 - y, 0.0, y - bbox.y2)
    return math.hypot(dx / width, dy / height)


def point_to_bbox_center_distance(
    point: tuple[float, float],
    bbox: BoundingBox,
    *,
    frame_width: float,
    frame_height: float,
) -> float:
    """Return normalized Euclidean distance from a point to a bbox center."""

    width, height = _validate_dimensions(frame_width, frame_height)
    _validate_bbox(bbox)
    center = ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)
    return math.hypot((float(point[0]) - center[0]) / width, (float(point[1]) - center[1]) / height)


def _inside(point: tuple[float, float], bbox: BoundingBox) -> bool:
    return bbox.x1 <= point[0] <= bbox.x2 and bbox.y1 <= point[1] <= bbox.y2


@dataclass(frozen=True)
class WristObjectMeasurement:
    """One valid wrist-object spatial proximity measurement."""

    wrist_name: str
    point: tuple[float, float]
    confidence: float
    visibility: float | None
    distance_to_bbox: float
    distance_to_center: float
    inside_object_bbox: bool
    distance_over_bbox_diagonal: float | None
    semantic_label: str = "WRIST_OBJECT_PROXIMITY"

    def to_record(self) -> dict[str, object]:
        return {
            "wrist": self.wrist_name,
            "point": [float(self.point[0]), float(self.point[1])],
            "confidence": float(self.confidence),
            "visibility": None if self.visibility is None else float(self.visibility),
            "distance_to_bbox": float(self.distance_to_bbox),
            "distance_to_center": float(self.distance_to_center),
            "inside_object_bbox": self.inside_object_bbox,
            "distance_over_bbox_diagonal": (
                None
                if self.distance_over_bbox_diagonal is None
                else float(self.distance_over_bbox_diagonal)
            ),
            "semantic_label": self.semantic_label,
        }


def measure_wrist_proximity(
    wrist: KeypointObservation,
    object_bbox: BoundingBox,
    *,
    frame_width: float,
    frame_height: float,
    minimum_confidence: float,
    minimum_visibility: float | None = None,
) -> WristObjectMeasurement | None:
    """Measure one wrist only when the backend score policy accepts it."""

    if wrist.confidence is None or wrist.confidence < minimum_confidence:
        return None
    if minimum_visibility is not None and (
        wrist.visibility is None or wrist.visibility < minimum_visibility
    ):
        return None
    _validate_bbox(object_bbox)
    distance = point_to_bbox_distance(
        (wrist.x, wrist.y), object_bbox, frame_width=frame_width, frame_height=frame_height
    )
    center_distance = point_to_bbox_center_distance(
        (wrist.x, wrist.y), object_bbox, frame_width=frame_width, frame_height=frame_height
    )
    diagonal = math.hypot(
        (object_bbox.x2 - object_bbox.x1) / float(frame_width),
        (object_bbox.y2 - object_bbox.y1) / float(frame_height),
    )
    return WristObjectMeasurement(
        wrist_name=wrist.name,
        point=(float(wrist.x), float(wrist.y)),
        confidence=float(wrist.confidence),
        visibility=None if wrist.visibility is None else float(wrist.visibility),
        distance_to_bbox=distance,
        distance_to_center=center_distance,
        inside_object_bbox=_inside((wrist.x, wrist.y), object_bbox),
        distance_over_bbox_diagonal=None if diagonal == 0 else distance / diagonal,
        semantic_label=(
            "WRIST_OBJECT_OVERLAP_CANDIDATE"
            if _inside((wrist.x, wrist.y), object_bbox)
            else "WRIST_OBJECT_PROXIMITY"
        ),
    )


def select_closest_wrist(
    measurements: Iterable[WristObjectMeasurement],
) -> WristObjectMeasurement | None:
    """Select the closest valid wrist with a stable left-before-right tie break."""

    valid = tuple(measurements)
    if not valid:
        return None
    return min(valid, key=lambda item: (item.distance_to_bbox, item.wrist_name))
