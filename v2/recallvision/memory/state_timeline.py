"""Observed object-state timelines built from consolidated memory.

This module deliberately represents only what was observed in the supplied
memory. A zone segment is not a claim that the object stayed there between
observations, and the module emits no event records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from recallvision.memory.schema import EvidenceRef, Observation


@dataclass(frozen=True)
class StateTimelineConfig:
    """Fixed segmentation controls for one P2B run."""

    max_observation_gap_seconds: float = 0.50
    minimum_observations_per_segment: int = 1

    def __post_init__(self) -> None:
        if self.max_observation_gap_seconds < 0:
            raise ValueError("max_observation_gap_seconds must be non-negative")
        if self.minimum_observations_per_segment < 1:
            raise ValueError("minimum_observations_per_segment must be at least 1")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "StateTimelineConfig":
        return cls(
            max_observation_gap_seconds=float(mapping["max_observation_gap_seconds"]),
            minimum_observations_per_segment=int(mapping["minimum_observations_per_segment"]),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "StateTimelineConfig":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("state timeline config must be a JSON object")
        return cls.from_mapping(record)

    def to_record(self) -> dict[str, Any]:
        return {
            "max_observation_gap_seconds": self.max_observation_gap_seconds,
            "minimum_observations_per_segment": self.minimum_observations_per_segment,
        }


@dataclass(frozen=True)
class ObjectStateRecord:
    """One contiguous observed-zone segment for one persistent entity."""

    persistent_id: str
    class_name: str
    video_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    zone: str | None
    observation_count: int
    first_evidence: EvidenceRef
    last_evidence: EvidenceRef
    source_track_ids: tuple[str, ...]
    source_frame_indices: tuple[int, ...]
    last_confidence: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "video_id": self.video_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "zone": self.zone,
            "observation_count": self.observation_count,
            "first_evidence": _evidence_to_record(self.first_evidence),
            "last_evidence": _evidence_to_record(self.last_evidence),
            "source_track_ids": list(self.source_track_ids),
            "source_frame_indices": list(self.source_frame_indices),
            "last_confidence": self.last_confidence,
        }


@dataclass(frozen=True)
class ObjectStateTimeline:
    """The ordered observed-state segments for one persistent entity."""

    persistent_id: str
    class_name: str
    video_id: str
    segments: tuple[ObjectStateRecord, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "video_id": self.video_id,
            "segments": [segment.to_record() for segment in self.segments],
        }


@dataclass(frozen=True)
class LastSeenState:
    """The latest observed state and its explicit distance from video end."""

    persistent_id: str
    class_name: str
    zone: str | None
    timestamp: float
    frame_index: int
    confidence: float | None
    evidence_ref: EvidenceRef
    video_end_timestamp: float
    staleness_seconds: float

    def to_record(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "zone": self.zone,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "confidence": self.confidence,
            "evidence_ref": _evidence_to_record(self.evidence_ref),
            "video_end_timestamp": self.video_end_timestamp,
            "staleness_seconds": self.staleness_seconds,
        }


def build_state_timeline(
    observations: Iterable[Observation],
    config: StateTimelineConfig,
) -> ObjectStateTimeline:
    """Order observations and segment them by zone and temporal continuity."""

    ordered = sorted(
        observations,
        key=lambda item: (item.timestamp, item.frame_index, item.track_id),
    )
    if not ordered:
        raise ValueError("at least one observation is required")
    _validate_identity(ordered)

    segment_observations: list[list[Observation]] = []
    current: list[Observation] = []
    for observation in ordered:
        if current:
            previous = current[-1]
            gap = observation.timestamp - previous.timestamp
            same_zone = observation.zone == previous.zone
            if not same_zone or gap > config.max_observation_gap_seconds:
                segment_observations.append(current)
                current = []
        current.append(observation)
    segment_observations.append(current)

    segments = tuple(
        _build_segment(items)
        for items in segment_observations
        if len(items) >= config.minimum_observations_per_segment
    )
    first = ordered[0]
    return ObjectStateTimeline(
        persistent_id=_required_persistent_id(first),
        class_name=first.class_name,
        video_id=first.video_id,
        segments=segments,
    )


def build_state_timelines(
    observations: Iterable[Observation],
    config: StateTimelineConfig,
) -> list[ObjectStateTimeline]:
    """Build independent timelines keyed by video and persistent identity."""

    grouped: dict[tuple[str, str], list[Observation]] = {}
    for observation in observations:
        grouped.setdefault((observation.video_id, _required_persistent_id(observation)), []).append(
            observation
        )
    return [
        build_state_timeline(grouped[key], config)
        for key in sorted(grouped, key=lambda item: (item[0], item[1]))
    ]


def select_last_seen_state(
    timeline: ObjectStateTimeline,
    *,
    video_end_timestamp: float,
) -> LastSeenState:
    """Select the latest segment endpoint and compute non-negative staleness."""

    if video_end_timestamp < 0:
        raise ValueError("video_end_timestamp must be non-negative")
    if not timeline.segments:
        raise ValueError("timeline contains no retained state segments")
    segment = max(timeline.segments, key=lambda item: (item.end_time, item.end_frame))
    last_timestamp = segment.last_evidence.timestamp
    if last_timestamp is None:
        raise ValueError("last evidence must contain a timestamp")
    if video_end_timestamp < last_timestamp:
        raise ValueError("video_end_timestamp cannot precede the last observation")
    return LastSeenState(
        persistent_id=timeline.persistent_id,
        class_name=timeline.class_name,
        zone=segment.zone,
        timestamp=last_timestamp,
        frame_index=segment.end_frame,
        confidence=segment.last_confidence,
        evidence_ref=segment.last_evidence,
        video_end_timestamp=video_end_timestamp,
        staleness_seconds=video_end_timestamp - last_timestamp,
    )


def _validate_identity(observations: Sequence[Observation]) -> None:
    first = observations[0]
    _required_persistent_id(first)
    if any(observation.video_id != first.video_id for observation in observations):
        raise ValueError("one timeline cannot mix video IDs")
    if any(observation.class_name != first.class_name for observation in observations):
        raise ValueError("one timeline cannot mix class names")
    if any(_required_persistent_id(observation) != first.persistent_id for observation in observations):
        raise ValueError("one timeline cannot mix persistent IDs")


def _required_persistent_id(observation: Observation) -> str:
    if not observation.persistent_id:
        raise ValueError("state timeline observations require persistent_id")
    return observation.persistent_id


def _build_segment(observations: Sequence[Observation]) -> ObjectStateRecord:
    first = observations[0]
    last = observations[-1]
    source_track_ids = {
        track_id
        for observation in observations
        for track_id in (observation.source_track_ids or (observation.track_id,))
    }
    source_frame_indices = tuple(sorted({observation.frame_index for observation in observations}))
    return ObjectStateRecord(
        persistent_id=_required_persistent_id(first),
        class_name=first.class_name,
        video_id=first.video_id,
        start_frame=first.frame_index,
        end_frame=last.frame_index,
        start_time=first.timestamp,
        end_time=last.timestamp,
        zone=first.zone,
        observation_count=len(observations),
        first_evidence=_evidence_for(first),
        last_evidence=_evidence_for(last),
        source_track_ids=tuple(sorted(source_track_ids, key=_stable_text_key)),
        source_frame_indices=source_frame_indices,
        last_confidence=last.confidence,
    )


def _evidence_for(observation: Observation) -> EvidenceRef:
    return EvidenceRef(
        video_id=observation.video_id,
        timestamp=observation.timestamp,
        frame_indices=[observation.frame_index],
        segment_ref=f"{observation.video_id}#frame={observation.frame_index}",
    )


def _evidence_to_record(evidence: EvidenceRef) -> dict[str, Any]:
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }


def _stable_text_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)
