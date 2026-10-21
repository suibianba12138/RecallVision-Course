"""Typed, deterministic summaries of short-term tracker tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from recallvision.memory.schema import BoundingBox, Observation

TrackletKey = tuple[str, str, str]


@dataclass(frozen=True)
class RepresentativeFrame:
    """A frame selected for audit or appearance extraction."""

    frame_index: int
    timestamp: float
    bbox: BoundingBox


@dataclass(frozen=True)
class TrackletSummary:
    """A deterministic summary for one ``(video, class, track_id)`` group."""

    video_id: str
    class_name: str
    track_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    observation_count: int
    mean_confidence: float | None
    first_bbox: BoundingBox
    last_bbox: BoundingBox
    median_bbox: BoundingBox
    first_zone: str | None
    last_zone: str | None
    representative_frames: tuple[RepresentativeFrame, ...]
    first_center_normalized: tuple[float, float]
    last_center_normalized: tuple[float, float]
    median_area: float

    @property
    def key(self) -> TrackletKey:
        return (self.video_id, self.class_name, self.track_id)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_record(self) -> dict[str, Any]:
        """Serialize without exposing dataclass internals to artifact writers."""

        return {
            "video_id": self.video_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "observation_count": self.observation_count,
            "mean_confidence": self.mean_confidence,
            "first_bbox": _bbox_record(self.first_bbox),
            "last_bbox": _bbox_record(self.last_bbox),
            "median_bbox": _bbox_record(self.median_bbox),
            "first_center_normalized": list(self.first_center_normalized),
            "last_center_normalized": list(self.last_center_normalized),
            "median_area": self.median_area,
            "first_zone": self.first_zone,
            "last_zone": self.last_zone,
            "representative_frames": [
                {
                    "frame_index": item.frame_index,
                    "timestamp": item.timestamp,
                    "bbox": _bbox_record(item.bbox),
                }
                for item in self.representative_frames
            ],
        }


def build_tracklet_summaries(
    observations: list[Observation] | tuple[Observation, ...],
    *,
    frame_width: int,
    frame_height: int,
) -> list[TrackletSummary]:
    """Group observations by explicit tracklet key and summarize each group."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    groups: dict[TrackletKey, list[Observation]] = {}
    for observation in observations:
        groups.setdefault(
            (observation.video_id, observation.class_name, observation.track_id), []
        ).append(observation)

    summaries: list[TrackletSummary] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key], key=lambda item: (item.frame_index, item.timestamp, item.track_id)
        )
        confidences = [item.confidence for item in ordered if item.confidence is not None]
        first = ordered[0]
        last = ordered[-1]
        median_box = BoundingBox(
            x1=float(median(item.bbox.x1 for item in ordered)),
            y1=float(median(item.bbox.y1 for item in ordered)),
            x2=float(median(item.bbox.x2 for item in ordered)),
            y2=float(median(item.bbox.y2 for item in ordered)),
        )
        representatives = tuple(
            RepresentativeFrame(
                frame_index=item.frame_index,
                timestamp=item.timestamp,
                bbox=item.bbox,
            )
            for item in _select_representatives(ordered)
        )
        summaries.append(
            TrackletSummary(
                video_id=first.video_id,
                class_name=first.class_name,
                track_id=first.track_id,
                start_frame=first.frame_index,
                end_frame=last.frame_index,
                start_time=first.timestamp,
                end_time=last.timestamp,
                observation_count=len(ordered),
                mean_confidence=(
                    sum(confidences) / len(confidences) if confidences else None
                ),
                first_bbox=first.bbox,
                last_bbox=last.bbox,
                median_bbox=median_box,
                first_zone=first.zone,
                last_zone=last.zone,
                representative_frames=representatives,
                first_center_normalized=_normalized_center(
                    first.bbox, frame_width, frame_height
                ),
                last_center_normalized=_normalized_center(
                    last.bbox, frame_width, frame_height
                ),
                median_area=_bbox_area(median_box),
            )
        )
    return summaries


def _select_representatives(ordered: list[Observation]) -> list[Observation]:
    # Keep three labelled audit slots even for a one- or two-observation
    # tracklet; repeated frames make the lack of temporal extent explicit.
    indices = [0, len(ordered) // 2, len(ordered) - 1]
    return [ordered[index] for index in indices]


def _normalized_center(
    bbox: BoundingBox, frame_width: int, frame_height: int
) -> tuple[float, float]:
    return (
        ((bbox.x1 + bbox.x2) / 2.0) / frame_width,
        ((bbox.y1 + bbox.y2) / 2.0) / frame_height,
    )


def bbox_center_normalized(bbox: BoundingBox, *, frame_width: int, frame_height: int) -> tuple[float, float]:
    """Return an ``(x, y)`` center normalized by frame dimensions."""

    return _normalized_center(bbox, frame_width, frame_height)


def _bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def bbox_area(bbox: BoundingBox) -> float:
    """Return a non-negative xyxy area."""

    return _bbox_area(bbox)


def _bbox_record(bbox: BoundingBox) -> dict[str, float]:
    return {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}
