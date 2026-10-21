"""Optional Ultralytics/ByteTrack adapter for the P1B analysis phase."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.zones import assign_horizontal_zone


def observation_from_detection(
    *,
    video_id: str,
    frame_index: int,
    fps: float,
    frame_width: int | float,
    bbox: Sequence[float],
    class_name: str,
    confidence: float,
    track_id: int | str | None,
    detection_index: int,
) -> Observation:
    """Map one detector/tracker result into the existing Observation schema.

    ``persistent_id`` is deliberately provisional in P1B. A ByteTrack ID is
    not a long-term RecallVision persistent object identity.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(bbox) != 4:
        raise ValueError("bbox must contain xyxy coordinates")
    box = BoundingBox(
        x1=float(bbox[0]),
        y1=float(bbox[1]),
        x2=float(bbox[2]),
        y2=float(bbox[3]),
    )
    if track_id is None:
        mapped_track_id = f"untracked:{frame_index}:{detection_index}"
        provisional_id = None
    else:
        mapped_track_id = str(track_id)
        provisional_id = f"track:{mapped_track_id}"
    return Observation(
        video_id=video_id,
        frame_index=frame_index,
        timestamp=frame_index / fps,
        track_id=mapped_track_id,
        persistent_id=provisional_id,
        entity_type="person" if class_name == "person" else "object",
        class_name=class_name,
        bbox=box,
        confidence=float(confidence),
        zone=assign_horizontal_zone(box, frame_width=frame_width),
    )


def read_video_metadata(video_path: str | Path) -> dict[str, float | int]:
    """Read video dimensions and timing using OpenCV without decoding output frames."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count < 0:
        raise RuntimeError(f"Invalid video metadata for: {video_path}")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": frame_count / fps,
    }


class UltralyticsPerceptionBackend:
    """Run a pretrained Ultralytics detector with the official ByteTrack config."""

    def __init__(
        self,
        *,
        model_name: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = device
        self.model_weights_path: Path | None = None

    def process_video(self, video_path: str | Path, *, video_id: str) -> Iterator[Observation]:
        """Yield RecallVision observations for each tracked video detection."""

        import cv2
        from ultralytics import YOLO

        metadata = read_video_metadata(video_path)
        model = YOLO(self.model_name)
        self.model_weights_path = _resolve_model_weights(model, self.model_name)
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        for frame_index, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_indices = boxes.cls.cpu().tolist()
            if boxes.id is None:
                track_ids: list[int | None] = [None] * len(coordinates)
            else:
                track_ids = [int(value) for value in boxes.id.int().cpu().tolist()]
            for detection_index, coordinates_for_detection in enumerate(coordinates):
                class_index = int(class_indices[detection_index])
                class_name = _class_name(result.names, class_index)
                yield observation_from_detection(
                    video_id=video_id,
                    frame_index=frame_index,
                    fps=float(metadata["fps"]),
                    frame_width=int(metadata["width"]),
                    bbox=coordinates_for_detection,
                    class_name=class_name,
                    confidence=float(confidences[detection_index]),
                    track_id=track_ids[detection_index],
                    detection_index=detection_index,
                )


def _class_name(names: Any, class_index: int) -> str:
    if isinstance(names, dict):
        return str(names[class_index])
    return str(names[class_index])


def _resolve_model_weights(model: Any, model_name: str) -> Path | None:
    candidate = getattr(model, "ckpt_path", None)
    if candidate:
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    named_path = Path(model_name)
    if named_path.exists():
        return named_path.resolve()
    return None


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
