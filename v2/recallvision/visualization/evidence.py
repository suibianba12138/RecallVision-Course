"""Detector-free rendering of one observation's source video frame."""

from __future__ import annotations

from pathlib import Path

from recallvision.memory.schema import Observation


def render_evidence_frame(
    video_path: str | Path,
    observation: Observation,
    output_path: str | Path,
) -> Path:
    """Seek one frame, annotate the supplied observation, and save a JPEG."""

    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_index)
        success, frame = capture.read()
    finally:
        capture.release()
    if not success:
        raise RuntimeError(
            f"Could not seek frame {observation.frame_index} in video: {video_path}"
        )

    top_left = (int(round(observation.bbox.x1)), int(round(observation.bbox.y1)))
    bottom_right = (int(round(observation.bbox.x2)), int(round(observation.bbox.y2)))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
    confidence = (
        f"{observation.confidence:.2f}" if observation.confidence is not None else "n/a"
    )
    label = (
        f"{observation.class_name} {confidence} "
        f"track={observation.track_id} zone={observation.zone} "
        f"t={observation.timestamp:.2f}s"
    )
    cv2.putText(
        frame,
        label,
        (max(0, top_left[0]), max(24, top_left[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write evidence image: {output}")
    return output
