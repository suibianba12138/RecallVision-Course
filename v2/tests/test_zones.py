"""Tests for the P1B horizontal geometric zone baseline."""

import pytest

from recallvision.memory.schema import BoundingBox
from recallvision.memory.zones import assign_horizontal_zone


@pytest.mark.parametrize(
    ("center_x", "expected"),
    [
        (0.0, "left_region"),
        (319.9, "left_region"),
        (320.0, "center_region"),
        (639.9, "center_region"),
        (640.0, "right_region"),
        (960.0, "right_region"),
    ],
)
def test_horizontal_zone_boundaries(center_x: float, expected: str) -> None:
    bbox = BoundingBox(x1=center_x - 1.0, y1=10.0, x2=center_x + 1.0, y2=20.0)

    assert assign_horizontal_zone(bbox, frame_width=960) == expected


def test_horizontal_zone_rejects_non_positive_frame_width() -> None:
    with pytest.raises(ValueError, match="frame_width"):
        assign_horizontal_zone(
            BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
            frame_width=0,
        )
