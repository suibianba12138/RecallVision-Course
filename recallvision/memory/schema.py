"""Stable, serialization-friendly contracts for object memory."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    """An image-space bounding box in ``xyxy`` pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Observation:
    """One entity observation at a frame and timestamp."""

    video_id: str
    frame_index: int
    timestamp: float
    track_id: str
    persistent_id: str | None
    entity_type: str
    class_name: str
    bbox: BoundingBox
    confidence: float | None = None
    zone: str | None = None
    source_track_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectState:
    """A time-indexed semantic state for a persistent object."""

    timestamp: float
    zone: str | None
    state: str


@dataclass
class ObjectMemory:
    """Persistent object identity and its time-varying history."""

    object_id: str
    class_name: str
    observations: list[Observation] = field(default_factory=list)
    trajectory: list[BoundingBox] = field(default_factory=list)
    zone_history: list[tuple[float, str | None]] = field(default_factory=list)
    state_history: list[ObjectState] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceRef:
    """A reference to supporting frames, timestamps, or a video segment."""

    video_id: str
    timestamp: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame_indices: list[int] = field(default_factory=list)
    segment_ref: str | None = None
