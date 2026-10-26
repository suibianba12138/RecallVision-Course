"""Dependency-free geometric zone assignment for P1B."""

from __future__ import annotations

from recallvision.memory.schema import BoundingBox


def assign_horizontal_zone(bbox: BoundingBox, *, frame_width: int | float) -> str:
    """Assign a bbox to left, center, or right by normalized center x."""

    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    center_x = (bbox.x1 + bbox.x2) / 2.0
    normalized_center_x = center_x / frame_width
    if normalized_center_x < 1.0 / 3.0:
        return "left_region"
    if normalized_center_x < 2.0 / 3.0:
        return "center_region"
    return "right_region"
