"""Optional tiny-video test for frame seeking and evidence rendering."""

from pathlib import Path

import pytest

from recallvision.memory.schema import BoundingBox, Observation


cv2 = pytest.importorskip("cv2")
from recallvision.visualization.evidence import render_evidence_frame  # noqa: E402


def test_evidence_renderer_seeks_requested_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    for _ in range(3):
        writer.write(__import__("numpy").zeros((48, 64, 3), dtype="uint8"))
    writer.release()

    observation = Observation(
        video_id="tiny",
        frame_index=1,
        timestamp=0.1,
        track_id="7",
        persistent_id="track:7",
        entity_type="object",
        class_name="bottle",
        bbox=BoundingBox(x1=5.0, y1=5.0, x2=20.0, y2=20.0),
        confidence=0.8,
        zone="left_region",
    )
    output_path = tmp_path / "evidence.jpg"

    result = render_evidence_frame(video_path, observation, output_path)

    assert result == output_path
    assert output_path.exists()
    rendered = cv2.imread(str(output_path))
    assert rendered is not None
    assert rendered.shape[:2] == (48, 64)
