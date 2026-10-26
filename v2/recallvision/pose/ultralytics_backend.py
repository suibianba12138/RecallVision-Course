"""Optional local-weight Ultralytics pose adapter.

The module imports no optional package at import time. A local model path is
required so constructing the backend can never trigger an implicit download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from recallvision.memory.schema import BoundingBox
from recallvision.memory.schema import Observation
from recallvision.pose.roi import compute_person_roi, remap_pose_detection
from recallvision.pose.schema import KeypointObservation, PoseDetection


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseBackendInfo:
    backend: str
    version: str
    model_identifier: str
    model_path: str
    model_sha256: str
    keypoint_confidence_source: str
    visibility_available: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "version": self.version,
            "model_identifier": self.model_identifier,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "keypoint_confidence_source": self.keypoint_confidence_source,
            "visibility_available": self.visibility_available,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _bbox_from_keypoints(keypoints: list[KeypointObservation]) -> BoundingBox | None:
    if not keypoints:
        return None
    return BoundingBox(
        x1=min(item.x for item in keypoints),
        y1=min(item.y for item in keypoints),
        x2=max(item.x for item in keypoints),
        y2=max(item.y for item in keypoints),
    )


class UltralyticsPoseBackend:
    """Run local Ultralytics pose inference on explicitly selected frames."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"pose model weights must already exist locally; refusing download: {path}"
            )
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.model_path = path.resolve()
        self.device = device
        self.confidence_threshold = confidence_threshold

    def info(self) -> PoseBackendInfo:
        """Return backend provenance without constructing the model."""

        try:
            import ultralytics
        except ImportError as error:
            raise RuntimeError("Ultralytics is required only in the selected pose runtime") from error
        return PoseBackendInfo(
            backend="ultralytics",
            version=str(getattr(ultralytics, "__version__", "UNKNOWN")),
            model_identifier=self.model_path.name,
            model_path=str(self.model_path),
            model_sha256=_sha256_file(self.model_path),
            keypoint_confidence_source="Ultralytics Keypoints.conf",
            visibility_available=False,
        )

    def infer(
        self,
        *,
        video_path: str | Path,
        video_id: str,
        frame_indices: Iterable[int],
    ) -> tuple[PoseDetection, ...]:
        """Infer only the explicitly listed frames; never invoke tracking."""

        import cv2
        from ultralytics import YOLO

        from recallvision.perception.ultralytics_backend import read_video_metadata

        metadata = read_video_metadata(video_path)
        frames = tuple(sorted(set(int(item) for item in frame_indices)))
        if any(item < 0 or item >= int(metadata["frame_count"]) for item in frames):
            raise ValueError("selected pose frame is outside the video")
        model = YOLO(str(self.model_path))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        detections: list[PoseDetection] = []
        try:
            for frame_index in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(f"Could not read selected frame {frame_index}: {video_path}")
                results = model.predict(
                    source=frame,
                    device=self.device,
                    conf=self.confidence_threshold,
                    verbose=False,
                )
                result = results[0] if results else None
                if result is None or result.keypoints is None:
                    continue
                detections.extend(
                    _convert_result(
                        result,
                        video_id=video_id,
                        frame_index=frame_index,
                        timestamp=frame_index / float(metadata["fps"]),
                    )
                )
        finally:
            capture.release()
        return tuple(sorted(detections, key=lambda item: (item.frame_index, item.detection_index)))

    def infer_persistent_person_rois(
        self,
        *,
        video_path: str | Path,
        video_id: str,
        person_observations: Iterable[Observation],
        frame_width: float,
        frame_height: float,
        roi_margin: float = 0.0,
    ) -> tuple[PoseDetection, ...]:
        """Infer pose only on event-local crops from canonical person bboxes."""

        import cv2
        from ultralytics import YOLO

        from recallvision.perception.ultralytics_backend import read_video_metadata

        metadata = read_video_metadata(video_path)
        frames_by_index: dict[int, list[Observation]] = {}
        for observation in person_observations:
            if observation.entity_type != "person" or observation.persistent_id is None:
                continue
            frames_by_index.setdefault(observation.frame_index, []).append(observation)
        frames = tuple(sorted(frames_by_index))
        if any(item < 0 or item >= int(metadata["frame_count"]) for item in frames):
            raise ValueError("selected pose frame is outside the video")
        model = YOLO(str(self.model_path))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        detections: list[PoseDetection] = []
        try:
            for frame_index in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(f"Could not read selected frame {frame_index}: {video_path}")
                for person in sorted(
                    frames_by_index[frame_index],
                    key=lambda item: (str(item.persistent_id), item.track_id),
                ):
                    roi = compute_person_roi(
                        person.bbox,
                        frame_width,
                        frame_height,
                        margin=roi_margin,
                    )
                    x0, y0, x1, y1 = roi.crop_bbox
                    crop = frame[y0:y1, x0:x1]
                    results = model.predict(
                        source=crop,
                        device=self.device,
                        conf=self.confidence_threshold,
                        verbose=False,
                    )
                    result = results[0] if results else None
                    if result is None or result.keypoints is None:
                        continue
                    local_detections = _convert_result(
                        result,
                        video_id=video_id,
                        frame_index=frame_index,
                        timestamp=frame_index / float(metadata["fps"]),
                    )
                    detections.extend(
                        remap_pose_detection(
                            detection,
                            roi,
                            persistent_person_id=str(person.persistent_id),
                        )
                        for detection in local_detections
                    )
        finally:
            capture.release()
        return tuple(
            sorted(
                detections,
                key=lambda item: (
                    item.frame_index,
                    str((item.roi_provenance or {}).get("persistent_person_id", "")),
                    item.detection_index,
                ),
            )
        )


def _convert_result(
    result: Any,
    *,
    video_id: str,
    frame_index: int,
    timestamp: float,
) -> list[PoseDetection]:
    keypoints = result.keypoints
    xy_rows = _tolist(getattr(keypoints, "xy", None))
    conf_rows = _tolist(getattr(keypoints, "conf", None))
    data_rows = _tolist(getattr(keypoints, "data", None))
    boxes = getattr(result, "boxes", None)
    bbox_rows = _tolist(getattr(boxes, "xyxy", None)) if boxes is not None else []
    box_conf_rows = _tolist(getattr(boxes, "conf", None)) if boxes is not None else []
    count = max(len(xy_rows), len(data_rows), len(bbox_rows))
    converted: list[PoseDetection] = []
    for detection_index in range(count):
        row = xy_rows[detection_index] if detection_index < len(xy_rows) else []
        data_row = data_rows[detection_index] if detection_index < len(data_rows) else []
        if not row and data_row:
            row = [item[:2] for item in data_row]
        confidence_row = conf_rows[detection_index] if detection_index < len(conf_rows) else []
        points: list[KeypointObservation] = []
        for keypoint_index, coordinates in enumerate(row):
            if len(coordinates) < 2:
                continue
            confidence = None
            if keypoint_index < len(confidence_row):
                confidence = float(confidence_row[keypoint_index])
            elif keypoint_index < len(data_row) and len(data_row[keypoint_index]) >= 3:
                confidence = float(data_row[keypoint_index][2])
            name = (
                COCO_KEYPOINT_NAMES[keypoint_index]
                if keypoint_index < len(COCO_KEYPOINT_NAMES)
                else f"keypoint_{keypoint_index}"
            )
            points.append(
                KeypointObservation(
                    name=name,
                    x=float(coordinates[0]),
                    y=float(coordinates[1]),
                    confidence=confidence,
                    visibility=None,
                )
            )
        if detection_index < len(bbox_rows) and len(bbox_rows[detection_index]) == 4:
            bbox = BoundingBox(*[float(item) for item in bbox_rows[detection_index]])
        else:
            bbox = _bbox_from_keypoints(points)
        if bbox is None:
            continue
        pose_confidence = (
            None
            if detection_index >= len(box_conf_rows)
            else float(box_conf_rows[detection_index])
        )
        converted.append(
            PoseDetection(
                video_id=video_id,
                frame_index=frame_index,
                timestamp=timestamp,
                detection_index=detection_index,
                pose_bbox=bbox,
                keypoints=tuple(points),
                pose_confidence=pose_confidence,
            )
        )
    return converted
