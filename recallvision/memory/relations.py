"""Typed duplicate relations and canonical same-frame observation handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from recallvision.memory.appearance import histogram_intersection_similarity
from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.tracklets import TrackletKey, TrackletSummary, bbox_area


class TrackletRelationType(str, Enum):
    """Relation semantics used by the R1 entity graph."""

    DUPLICATE = "DUPLICATE"
    CONTINUATION = "CONTINUATION"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class DuplicateCandidate:
    """One undirected same-class overlap candidate with all audit features."""

    source_key: TrackletKey
    target_key: TrackletKey
    source_track_id: str
    target_track_id: str
    video_id: str
    class_name: str
    shared_frame_count: int
    overlap_ratio: float
    median_iou: float
    median_ios: float
    median_center_distance: float
    median_area_ratio: float
    appearance_similarity: float
    hard_gate_pass: bool
    accepted: bool = False
    rejection_reason: str | None = None

    @property
    def relation_type(self) -> TrackletRelationType:
        return TrackletRelationType.DUPLICATE

    def to_record(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "source_track_id": self.source_track_id,
            "target_track_id": self.target_track_id,
            "source_key": list(self.source_key),
            "target_key": list(self.target_key),
            "video_id": self.video_id,
            "class_name": self.class_name,
            "shared_frame_count": self.shared_frame_count,
            "overlap_ratio": self.overlap_ratio,
            "median_iou": self.median_iou,
            "median_ios": self.median_ios,
            "median_center_distance": self.median_center_distance,
            "median_area_ratio": self.median_area_ratio,
            "appearance_similarity": self.appearance_similarity,
            "hard_gate_pass": self.hard_gate_pass,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class DuplicateGroup:
    """An equivalence component of simultaneous duplicate tracklets."""

    group_id: str
    video_id: str
    class_name: str
    tracklet_keys: tuple[TrackletKey, ...]
    track_ids: tuple[str, ...]
    start_frame: int
    end_frame: int
    accepted_duplicate_edges: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "video_id": self.video_id,
            "class_name": self.class_name,
            "tracklet_keys": [list(key) for key in self.tracklet_keys],
            "track_ids": list(self.track_ids),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "accepted_duplicate_edges": list(self.accepted_duplicate_edges),
        }


@dataclass(frozen=True)
class CanonicalObservation:
    """One per-group/per-frame observation with raw-track provenance."""

    observation: Observation
    group_id: str
    source_track_ids: tuple[str, ...]
    source_tracklet_keys: tuple[TrackletKey, ...]

    def to_record(self, *, persistent_id: str | None = None) -> dict[str, Any]:
        observation = self.observation
        return {
            "video_id": observation.video_id,
            "frame_index": observation.frame_index,
            "timestamp": observation.timestamp,
            "track_id": observation.track_id,
            "persistent_id": persistent_id if persistent_id is not None else observation.persistent_id,
            "entity_type": observation.entity_type,
            "class_name": observation.class_name,
            "bbox": {
                "x1": observation.bbox.x1,
                "y1": observation.bbox.y1,
                "x2": observation.bbox.x2,
                "y2": observation.bbox.y2,
            },
            "confidence": observation.confidence,
            "zone": observation.zone,
            "canonical_group_id": self.group_id,
            "source_track_ids": list(self.source_track_ids),
        }


def find_duplicate_candidates(
    summaries: list[TrackletSummary] | tuple[TrackletSummary, ...],
    observations: list[Observation] | tuple[Observation, ...],
    descriptors: dict[TrackletKey, tuple[float, ...]],
    *,
    frame_width: int,
    frame_height: int,
    config: Any,
) -> list[DuplicateCandidate]:
    """Compute one deterministic candidate for every overlapping same-class pair."""

    observations_by_key: dict[TrackletKey, list[Observation]] = {}
    for observation in observations:
        key = (observation.video_id, observation.class_name, observation.track_id)
        observations_by_key.setdefault(key, []).append(observation)
    ordered = sorted(summaries, key=lambda item: item.key)
    candidates: list[DuplicateCandidate] = []
    for index, source in enumerate(ordered):
        for target in ordered[index + 1 :]:
            if source.video_id != target.video_id or source.class_name != target.class_name:
                continue
            if not set(item.frame_index for item in observations_by_key[source.key]).intersection(
                item.frame_index for item in observations_by_key[target.key]
            ):
                continue
            if source.key not in descriptors or target.key not in descriptors:
                raise KeyError(f"missing appearance descriptor for {source.key} or {target.key}")
            candidates.append(
                compute_duplicate_features(
                    source,
                    target,
                    observations_by_key[source.key],
                    observations_by_key[target.key],
                    appearance_similarity=histogram_intersection_similarity(
                        descriptors[source.key], descriptors[target.key]
                    ),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    config=config,
                )
            )
    return candidates


def compute_duplicate_features(
    source: TrackletSummary,
    target: TrackletSummary,
    source_observations: list[Observation],
    target_observations: list[Observation],
    *,
    appearance_similarity: float,
    frame_width: int,
    frame_height: int,
    config: Any,
) -> DuplicateCandidate:
    """Compute median shared-frame geometry and apply fixed duplicate gates."""

    source_by_frame = {item.frame_index: item for item in source_observations}
    target_by_frame = {item.frame_index: item for item in target_observations}
    shared_frames = sorted(set(source_by_frame).intersection(target_by_frame))
    ious: list[float] = []
    ioss: list[float] = []
    centers: list[float] = []
    area_ratios: list[float] = []
    for frame_index in shared_frames:
        left = source_by_frame[frame_index].bbox
        right = target_by_frame[frame_index].bbox
        ious.append(_bbox_iou(left, right))
        ioss.append(_bbox_ios(left, right))
        left_center = _normalized_center(left, frame_width, frame_height)
        right_center = _normalized_center(right, frame_width, frame_height)
        centers.append(math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
        left_area = bbox_area(left)
        right_area = bbox_area(right)
        area_ratios.append(right_area / left_area if left_area > 0.0 else math.inf)

    shared_frame_count = len(shared_frames)
    overlap_ratio = shared_frame_count / min(
        len(source_by_frame), len(target_by_frame)
    ) if source_by_frame and target_by_frame else 0.0
    median_iou = float(median(ious)) if ious else 0.0
    median_ios = float(median(ioss)) if ioss else 0.0
    median_center = float(median(centers)) if centers else math.inf
    median_area_ratio = float(median(area_ratios)) if area_ratios else math.inf
    rejection_reason = _duplicate_gate_failure(
        source,
        target,
        shared_frame_count=shared_frame_count,
        overlap_ratio=overlap_ratio,
        median_iou=median_iou,
        median_ios=median_ios,
        median_center_distance=median_center,
        median_area_ratio=median_area_ratio,
        appearance_similarity=appearance_similarity,
        config=config,
    )
    return DuplicateCandidate(
        source_key=source.key,
        target_key=target.key,
        source_track_id=source.track_id,
        target_track_id=target.track_id,
        video_id=source.video_id,
        class_name=source.class_name,
        shared_frame_count=shared_frame_count,
        overlap_ratio=overlap_ratio,
        median_iou=median_iou,
        median_ios=median_ios,
        median_center_distance=median_center,
        median_area_ratio=median_area_ratio,
        appearance_similarity=appearance_similarity,
        hard_gate_pass=rejection_reason is None,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def build_duplicate_groups(
    summaries: list[TrackletSummary] | tuple[TrackletSummary, ...],
    candidates: list[DuplicateCandidate] | tuple[DuplicateCandidate, ...],
) -> list[DuplicateGroup]:
    """Build deterministic union-find components from accepted duplicate edges."""

    keys = {summary.key for summary in summaries}
    parent = {key: key for key in keys}

    def find(key: TrackletKey) -> TrackletKey:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: TrackletKey, right: TrackletKey) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    accepted = [candidate for candidate in candidates if candidate.accepted]
    for candidate in sorted(accepted, key=lambda item: (item.source_key, item.target_key)):
        if candidate.source_key not in parent or candidate.target_key not in parent:
            raise ValueError("duplicate edge references an unknown tracklet")
        union(candidate.source_key, candidate.target_key)

    by_root: dict[TrackletKey, list[TrackletSummary]] = {}
    summary_by_key = {summary.key: summary for summary in summaries}
    for key in sorted(keys):
        by_root.setdefault(find(key), []).append(summary_by_key[key])
    components = sorted(
        by_root.values(),
        key=lambda items: (
            items[0].class_name,
            min(item.start_frame for item in items),
            min(_track_id_sort_key(item.track_id) for item in items),
            items[0].video_id,
        ),
    )
    counters: dict[str, int] = {}
    groups: list[DuplicateGroup] = []
    for items in components:
        items = sorted(items, key=lambda item: (_track_id_sort_key(item.track_id), item.key))
        class_name = items[0].class_name
        counters[class_name] = counters.get(class_name, 0) + 1
        group_id = f"dup:{class_name}:{counters[class_name]:04d}"
        item_keys = {item.key for item in items}
        edge_labels = sorted(
            {
                _edge_label(candidate.source_track_id, candidate.target_track_id)
                for candidate in accepted
                if candidate.source_key in item_keys and candidate.target_key in item_keys
            }
        )
        groups.append(
            DuplicateGroup(
                group_id=group_id,
                video_id=items[0].video_id,
                class_name=class_name,
                tracklet_keys=tuple(item.key for item in items),
                track_ids=tuple(item.track_id for item in items),
                start_frame=min(item.start_frame for item in items),
                end_frame=max(item.end_frame for item in items),
                accepted_duplicate_edges=tuple(edge_labels),
            )
        )
    return groups


def canonicalize_observations(
    observations: list[Observation] | tuple[Observation, ...],
    groups: list[DuplicateGroup] | tuple[DuplicateGroup, ...],
) -> list[CanonicalObservation]:
    """Select one deterministic observation per duplicate group and frame."""

    group_by_key: dict[TrackletKey, DuplicateGroup] = {}
    for group in groups:
        for key in group.tracklet_keys:
            group_by_key[key] = group
    grouped: dict[tuple[str, int], list[Observation]] = {}
    for observation in observations:
        key = (observation.video_id, observation.class_name, observation.track_id)
        group = group_by_key.get(key)
        group_id = group.group_id if group else f"singleton:{observation.video_id}:{observation.class_name}:{observation.track_id}"
        grouped.setdefault((group_id, observation.frame_index), []).append(observation)
    ordered: list[CanonicalObservation] = []
    for (group_id, _frame), candidates in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        chosen = sorted(candidates, key=_canonical_selection_key)[0]
        source_ids = tuple(sorted({item.track_id for item in candidates}, key=_track_id_sort_key))
        source_keys = tuple(
            sorted(
                {
                    (item.video_id, item.class_name, item.track_id) for item in candidates
                },
                key=lambda item: (item[:2], _track_id_sort_key(item[2])),
            )
        )
        ordered.append(
            CanonicalObservation(
                observation=chosen,
                group_id=group_id,
                source_track_ids=source_ids,
                source_tracklet_keys=source_keys,
            )
        )
    return sorted(
        ordered,
        key=lambda item: (
            item.observation.video_id,
            item.observation.frame_index,
            item.group_id,
            _track_id_sort_key(item.observation.track_id),
        ),
    )


def _duplicate_gate_failure(
    source: TrackletSummary,
    target: TrackletSummary,
    *,
    shared_frame_count: int,
    overlap_ratio: float,
    median_iou: float,
    median_ios: float,
    median_center_distance: float,
    median_area_ratio: float,
    appearance_similarity: float,
    config: Any,
) -> str | None:
    if source.video_id != target.video_id:
        return "video_mismatch"
    if source.class_name != target.class_name:
        return "class_mismatch"
    if shared_frame_count < config.duplicate_min_shared_frames:
        return "shared_frame_count"
    if overlap_ratio < config.duplicate_min_overlap_ratio:
        return "overlap_ratio"
    if appearance_similarity < config.duplicate_min_appearance_similarity:
        return "appearance_similarity"
    if not config.duplicate_min_area_ratio <= median_area_ratio <= config.duplicate_max_area_ratio:
        return "area_ratio"
    spatial_evidence = (
        median_iou >= config.duplicate_min_median_iou
        or median_ios >= config.duplicate_min_median_ios
        or median_center_distance <= config.duplicate_max_median_center_distance
    )
    if not spatial_evidence:
        return "spatial_evidence"
    return None


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _intersection_area(left, right)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_ios(left: BoundingBox, right: BoundingBox) -> float:
    smaller = min(bbox_area(left), bbox_area(right))
    return _intersection_area(left, right) / smaller if smaller > 0.0 else 0.0


def _intersection_area(left: BoundingBox, right: BoundingBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def _normalized_center(bbox: BoundingBox, frame_width: int, frame_height: int) -> tuple[float, float]:
    return (((bbox.x1 + bbox.x2) / 2.0) / frame_width, ((bbox.y1 + bbox.y2) / 2.0) / frame_height)


def _canonical_selection_key(observation: Observation) -> tuple[float, float, tuple[int, int | str]]:
    confidence = observation.confidence if observation.confidence is not None else float("-inf")
    return (-confidence, -bbox_area(observation.bbox), _track_id_sort_key(observation.track_id))


def _edge_label(source: str, target: str) -> str:
    return f"{source}<->{target}"


def _track_id_sort_key(track_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)
