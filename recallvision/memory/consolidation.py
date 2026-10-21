"""Duplicate-aware fragment consolidation for the P2A-R1 baseline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import median
from pathlib import Path
from typing import Any

from recallvision.memory.appearance import histogram_intersection_similarity
from recallvision.memory.relations import (
    CanonicalObservation,
    DuplicateCandidate,
    DuplicateGroup,
    TrackletRelationType,
    canonicalize_observations,
)
from recallvision.memory.schema import Observation
from recallvision.memory.tracklets import TrackletKey, TrackletSummary, bbox_area


@dataclass(frozen=True)
class EntityConsolidationConfig:
    duplicate_min_shared_frames: int = 2
    duplicate_min_overlap_ratio: float = 0.25
    duplicate_min_median_iou: float = 0.20
    duplicate_min_median_ios: float = 0.60
    duplicate_max_median_center_distance: float = 0.08
    duplicate_min_appearance_similarity: float = 0.35
    duplicate_min_area_ratio: float = 0.20
    duplicate_max_area_ratio: float = 5.00
    continuation_max_temporal_gap_seconds: float = 1.25
    continuation_max_center_distance_normalized: float = 0.35
    continuation_min_area_ratio: float = 0.40
    continuation_max_area_ratio: float = 2.50
    continuation_min_appearance_similarity: float = 0.35
    continuation_sigma_d: float = 0.25
    continuation_sigma_s: float = 0.70
    continuation_threshold: float = 0.65
    boundary_window_size: int = 3
    appearance_h_bins: int = 32
    appearance_s_bins: int = 32
    conflict_max_center_distance: float = 0.08
    conflict_max_iou: float = 0.05

    def to_record(self) -> dict[str, Any]:
        return {
            "duplicate_min_shared_frames": self.duplicate_min_shared_frames,
            "duplicate_min_overlap_ratio": self.duplicate_min_overlap_ratio,
            "duplicate_min_median_iou": self.duplicate_min_median_iou,
            "duplicate_min_median_ios": self.duplicate_min_median_ios,
            "duplicate_max_median_center_distance": self.duplicate_max_median_center_distance,
            "duplicate_min_appearance_similarity": self.duplicate_min_appearance_similarity,
            "duplicate_min_area_ratio": self.duplicate_min_area_ratio,
            "duplicate_max_area_ratio": self.duplicate_max_area_ratio,
            "continuation_max_temporal_gap_seconds": self.continuation_max_temporal_gap_seconds,
            "continuation_max_center_distance_normalized": self.continuation_max_center_distance_normalized,
            "continuation_min_area_ratio": self.continuation_min_area_ratio,
            "continuation_max_area_ratio": self.continuation_max_area_ratio,
            "continuation_min_appearance_similarity": self.continuation_min_appearance_similarity,
            "continuation_sigma_d": self.continuation_sigma_d,
            "continuation_sigma_s": self.continuation_sigma_s,
            "continuation_threshold": self.continuation_threshold,
            "boundary_window_size": self.boundary_window_size,
            "appearance_h_bins": self.appearance_h_bins,
            "appearance_s_bins": self.appearance_s_bins,
            "conflict_max_center_distance": self.conflict_max_center_distance,
            "conflict_max_iou": self.conflict_max_iou,
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "EntityConsolidationConfig":
        return cls(**mapping)


@dataclass(frozen=True)
class BoundarySummary:
    fragment_id: str
    video_id: str
    class_name: str
    track_ids: tuple[str, ...]
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    observation_count: int
    canonical_observation_count: int
    start_center: tuple[float, float]
    end_center: tuple[float, float]
    start_area: float
    end_area: float
    start_appearance: tuple[float, ...]
    end_appearance: tuple[float, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "video_id": self.video_id,
            "class_name": self.class_name,
            "track_ids": list(self.track_ids),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "observation_count": self.observation_count,
            "canonical_observation_count": self.canonical_observation_count,
            "start_center": list(self.start_center),
            "end_center": list(self.end_center),
            "start_area": self.start_area,
            "end_area": self.end_area,
            "start_appearance": list(self.start_appearance),
            "end_appearance": list(self.end_appearance),
        }


@dataclass(frozen=True)
class EntityFragment:
    fragment_id: str
    group_id: str
    boundary: BoundarySummary
    tracklet_keys: tuple[TrackletKey, ...] = ()
    canonical_observations: tuple[CanonicalObservation, ...] = ()

    @property
    def class_name(self) -> str:
        return self.boundary.class_name

    @property
    def video_id(self) -> str:
        return self.boundary.video_id


@dataclass(frozen=True)
class ContinuationCandidate:
    source_fragment_id: str
    target_fragment_id: str
    video_id: str
    class_name: str
    temporal_gap: float
    center_distance: float
    area_ratio: float
    appearance_similarity: float
    spatial_similarity: float
    scale_similarity: float
    score: float
    hard_gate_pass: bool
    accepted: bool = False
    rejection_reason: str | None = None

    @property
    def relation_type(self) -> TrackletRelationType:
        return TrackletRelationType.CONTINUATION

    def to_record(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "source_fragment_id": self.source_fragment_id,
            "target_fragment_id": self.target_fragment_id,
            "video_id": self.video_id,
            "class_name": self.class_name,
            "temporal_gap": self.temporal_gap,
            "center_distance": self.center_distance,
            "area_ratio": self.area_ratio,
            "appearance_similarity": self.appearance_similarity,
            "spatial_similarity": self.spatial_similarity,
            "scale_similarity": self.scale_similarity,
            "score": self.score,
            "hard_gate_pass": self.hard_gate_pass,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class ContinuationSelection:
    all_edges: tuple[ContinuationCandidate, ...]
    accepted_edges: tuple[ContinuationCandidate, ...]

    def to_records(self) -> list[dict[str, Any]]:
        return [edge.to_record() for edge in self.all_edges]


@dataclass(frozen=True)
class EntityConflict:
    fragment_a: str
    fragment_b: str
    video_id: str
    class_name: str
    frame_indices: tuple[int, ...]
    center_distance: float
    median_iou: float
    reason: str

    @property
    def relation_type(self) -> TrackletRelationType:
        return TrackletRelationType.CONFLICT

    def to_record(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "fragment_a": self.fragment_a,
            "fragment_b": self.fragment_b,
            "video_id": self.video_id,
            "class_name": self.class_name,
            "frame_indices": list(self.frame_indices),
            "center_distance": self.center_distance,
            "median_iou": self.median_iou,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PersistentEntity:
    persistent_id: str
    class_name: str
    fragment_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    start_time: float
    end_time: float
    observation_count: int
    canonical_observation_count: int
    relation_provenance: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "fragment_ids": list(self.fragment_ids),
            "track_ids": list(self.track_ids),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "observation_count": self.observation_count,
            "canonical_observation_count": self.canonical_observation_count,
            "relation_provenance": list(self.relation_provenance),
        }


@dataclass(frozen=True)
class EntityBuild:
    entities: tuple[PersistentEntity, ...]
    track_to_persistent: dict[TrackletKey, str]
    fragment_to_persistent: dict[str, str]

    def to_records(self) -> dict[str, Any]:
        return {
            "entities": [entity.to_record() for entity in self.entities],
            "track_to_persistent": {
                "|".join(key): value for key, value in sorted(self.track_to_persistent.items())
            },
            "fragment_to_persistent": dict(sorted(self.fragment_to_persistent.items())),
        }


def build_boundary_summary(
    fragment_id: str,
    canonical_observations: list[CanonicalObservation] | tuple[CanonicalObservation, ...],
    appearances: dict[Any, tuple[float, ...]],
    *,
    frame_width: int,
    frame_height: int,
    window_size: int,
) -> BoundarySummary:
    """Summarize the first/last K canonical observations with elementwise medians."""

    ordered = sorted(
        canonical_observations,
        key=lambda item: (item.observation.frame_index, item.observation.timestamp),
    )
    if not ordered:
        raise ValueError(f"fragment {fragment_id} has no canonical observations")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    start = ordered[:window_size]
    end = ordered[-window_size:]
    first = ordered[0].observation
    last = ordered[-1].observation
    return BoundarySummary(
        fragment_id=fragment_id,
        video_id=first.video_id,
        class_name=first.class_name,
        track_ids=tuple(sorted({track_id for item in ordered for track_id in item.source_track_ids}, key=_track_id_sort_key)),
        start_frame=first.frame_index,
        end_frame=last.frame_index,
        start_time=first.timestamp,
        end_time=last.timestamp,
        observation_count=len(ordered),
        canonical_observation_count=len(ordered),
        start_center=_median_center(start, frame_width, frame_height),
        end_center=_median_center(end, frame_width, frame_height),
        start_area=float(median(bbox_area(item.observation.bbox) for item in start)),
        end_area=float(median(bbox_area(item.observation.bbox) for item in end)),
        start_appearance=_median_appearance(start, appearances),
        end_appearance=_median_appearance(end, appearances),
    )


def build_entity_fragments(
    summaries: list[TrackletSummary] | tuple[TrackletSummary, ...],
    groups: list[DuplicateGroup] | tuple[DuplicateGroup, ...],
    observations: list[Observation] | tuple[Observation, ...],
    appearances: dict[Any, tuple[float, ...]],
    *,
    frame_width: int,
    frame_height: int,
    window_size: int,
) -> list[EntityFragment]:
    """Create one fragment per duplicate-equivalence component."""

    canonical = canonicalize_observations(observations, groups)
    fragments: list[EntityFragment] = []
    by_group: dict[str, list[CanonicalObservation]] = {}
    for item in canonical:
        by_group.setdefault(item.group_id, []).append(item)
    summary_by_key = {summary.key: summary for summary in summaries}
    for group in groups:
        group_canonical = by_group.get(group.group_id, [])
        if not group_canonical:
            continue
        fragment_id = f"fragment:{group.group_id}"
        raw_count = sum(
            1
            for observation in observations
            if (observation.video_id, observation.class_name, observation.track_id)
            in set(group.tracklet_keys)
        )
        boundary = build_boundary_summary(
            fragment_id,
            group_canonical,
            appearances,
            frame_width=frame_width,
            frame_height=frame_height,
            window_size=window_size,
        )
        boundary = BoundarySummary(
            **{
                **boundary.__dict__,
                "observation_count": raw_count,
            }
        )
        fragments.append(
            EntityFragment(
                fragment_id=fragment_id,
                group_id=group.group_id,
                boundary=boundary,
                tracklet_keys=group.tracklet_keys,
                canonical_observations=tuple(group_canonical),
            )
        )
    return sorted(fragments, key=lambda item: _fragment_sort_key(item))


def find_continuation_candidates(
    fragments: list[EntityFragment] | tuple[EntityFragment, ...],
    *,
    config: EntityConsolidationConfig,
) -> list[ContinuationCandidate]:
    """Compute fixed-score forward candidates over duplicate-collapsed fragments."""

    ordered = sorted(fragments, key=_fragment_sort_key)
    candidates: list[ContinuationCandidate] = []
    for source_index, source in enumerate(ordered):
        for target in ordered[source_index + 1 :]:
            if source.video_id != target.video_id or source.class_name != target.class_name:
                continue
            if target.boundary.start_frame <= source.boundary.start_frame:
                continue
            candidates.append(compute_continuation_features(source, target, config=config))
    return candidates


def compute_continuation_features(
    source: EntityFragment,
    target: EntityFragment,
    *,
    config: EntityConsolidationConfig,
) -> ContinuationCandidate:
    source_boundary, target_boundary = source.boundary, target.boundary
    temporal_gap = target_boundary.start_time - source_boundary.end_time
    dx = source_boundary.end_center[0] - target_boundary.start_center[0]
    dy = source_boundary.end_center[1] - target_boundary.start_center[1]
    center_distance = math.hypot(dx, dy)
    area_ratio = target_boundary.start_area / source_boundary.end_area if source_boundary.end_area > 0 else math.inf
    appearance_similarity = histogram_intersection_similarity(
        source_boundary.end_appearance, target_boundary.start_appearance
    )
    spatial_similarity = math.exp(-center_distance / config.continuation_sigma_d)
    scale_similarity = (
        math.exp(-abs(math.log(area_ratio)) / config.continuation_sigma_s)
        if area_ratio > 0 and math.isfinite(area_ratio) else 0.0
    )
    score = 0.50 * appearance_similarity + 0.30 * spatial_similarity + 0.20 * scale_similarity
    rejection_reason = _continuation_gate_failure(
        source,
        target,
        temporal_gap=temporal_gap,
        center_distance=center_distance,
        area_ratio=area_ratio,
        appearance_similarity=appearance_similarity,
        config=config,
    )
    return ContinuationCandidate(
        source_fragment_id=source.fragment_id,
        target_fragment_id=target.fragment_id,
        video_id=source.video_id,
        class_name=source.class_name,
        temporal_gap=temporal_gap,
        center_distance=center_distance,
        area_ratio=area_ratio,
        appearance_similarity=appearance_similarity,
        spatial_similarity=spatial_similarity,
        scale_similarity=scale_similarity,
        score=score,
        hard_gate_pass=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def select_continuation_edges(
    candidates: list[ContinuationCandidate] | tuple[ContinuationCandidate, ...],
    *,
    conflicts: list[EntityConflict] | tuple[EntityConflict, ...],
    config: EntityConsolidationConfig,
) -> ContinuationSelection:
    """Select deterministic continuation edges while respecting conflict components."""

    eligible = [
        item for item in candidates
        if item.hard_gate_pass and item.score >= config.continuation_threshold
    ]
    eligible.sort(key=lambda item: (-item.score, _fragment_id_sort_key(item.source_fragment_id), _fragment_id_sort_key(item.target_fragment_id)))
    source_used: set[str] = set()
    target_used: set[str] = set()
    parent: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    conflict_pairs = {
        frozenset((conflict.fragment_a, conflict.fragment_b)) for conflict in conflicts
    }

    def ensure(node: str) -> None:
        if node not in parent:
            parent[node] = node
            members[node] = {node}

    def find(node: str) -> str:
        ensure(node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def can_union(source: str, target: str) -> bool:
        source_root, target_root = find(source), find(target)
        if source_root == target_root:
            return False
        return not any(
            frozenset((left, right)) in conflict_pairs
            for left in members[source_root]
            for right in members[target_root]
        )

    def union(source: str, target: str) -> None:
        source_root, target_root = find(source), find(target)
        if source_root == target_root:
            return
        if source_root > target_root:
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        members[source_root].update(members[target_root])
        del members[target_root]

    accepted_keys: set[tuple[str, str]] = set()
    reject_by_key: dict[tuple[str, str], str] = {}
    for candidate in eligible:
        key = (candidate.source_fragment_id, candidate.target_fragment_id)
        if candidate.source_fragment_id in source_used:
            reject_by_key[key] = "one_successor"
        elif candidate.target_fragment_id in target_used:
            reject_by_key[key] = "one_predecessor"
        elif not can_union(*key):
            reject_by_key[key] = "simultaneous_spatial_conflict" if any(
                frozenset((left, right)) in conflict_pairs
                for left in members[find(key[0])]
                for right in members[find(key[1])]
            ) else "cycle"
        else:
            source_used.add(candidate.source_fragment_id)
            target_used.add(candidate.target_fragment_id)
            union(*key)
            accepted_keys.add(key)

    finalized: list[ContinuationCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (_fragment_id_sort_key(item.source_fragment_id), _fragment_id_sort_key(item.target_fragment_id))):
        key = (candidate.source_fragment_id, candidate.target_fragment_id)
        if key in accepted_keys:
            finalized.append(_replace_continuation(candidate, accepted=True, rejection_reason=None))
        elif not candidate.hard_gate_pass:
            finalized.append(candidate)
        elif candidate.score < config.continuation_threshold:
            finalized.append(_replace_continuation(candidate, rejection_reason="score_below_threshold"))
        else:
            finalized.append(_replace_continuation(candidate, rejection_reason=reject_by_key.get(key, "not_selected")))
    accepted = tuple(item for item in finalized if item.accepted)
    return ContinuationSelection(all_edges=tuple(finalized), accepted_edges=accepted)


def detect_spatial_conflicts(
    fragments: list[EntityFragment] | tuple[EntityFragment, ...],
    *,
    config: EntityConsolidationConfig,
    frame_width: int,
    frame_height: int,
) -> list[EntityConflict]:
    """Detect distinct same-class fragments that coexist at separated locations."""

    conflicts: list[EntityConflict] = []
    ordered = sorted(fragments, key=_fragment_sort_key)
    for index, left in enumerate(ordered):
        left_by_frame = {item.observation.frame_index: item for item in left.canonical_observations}
        for right in ordered[index + 1 :]:
            if left.video_id != right.video_id or left.class_name != right.class_name:
                continue
            right_by_frame = {item.observation.frame_index: item for item in right.canonical_observations}
            shared = sorted(set(left_by_frame).intersection(right_by_frame))
            if not shared:
                continue
            centers: list[float] = []
            ious: list[float] = []
            for frame in shared:
                left_box = left_by_frame[frame].observation.bbox
                right_box = right_by_frame[frame].observation.bbox
                left_center = _center(left_box, frame_width, frame_height)
                right_center = _center(right_box, frame_width, frame_height)
                centers.append(math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
                ious.append(_iou(left_box, right_box))
            median_center = float(median(centers))
            median_iou = float(median(ious))
            if median_center > config.conflict_max_center_distance and median_iou <= config.conflict_max_iou:
                conflicts.append(
                    EntityConflict(
                        fragment_a=left.fragment_id,
                        fragment_b=right.fragment_id,
                        video_id=left.video_id,
                        class_name=left.class_name,
                        frame_indices=tuple(shared),
                        center_distance=median_center,
                        median_iou=median_iou,
                        reason="SIMULTANEOUS_SPATIAL_CONFLICT",
                    )
                )
    return conflicts


def build_persistent_entities(
    fragments: list[EntityFragment] | tuple[EntityFragment, ...],
    observations: list[Observation] | tuple[Observation, ...],
    canonical_observations: list[CanonicalObservation] | tuple[CanonicalObservation, ...],
    continuation_edges: list[ContinuationCandidate] | tuple[ContinuationCandidate, ...],
    duplicate_candidates: list[DuplicateCandidate] | tuple[DuplicateCandidate, ...],
) -> EntityBuild:
    """Build deterministic entity components and raw track-to-entity mapping."""

    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    parent = {fragment.fragment_id: fragment.fragment_id for fragment in fragments}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            raise ValueError("continuation graph contains a cycle")
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    accepted_continuations = [edge for edge in continuation_edges if edge.accepted]
    for edge in accepted_continuations:
        if edge.source_fragment_id not in by_id or edge.target_fragment_id not in by_id:
            raise ValueError("continuation references unknown fragment")
        union(edge.source_fragment_id, edge.target_fragment_id)
    components: dict[str, list[EntityFragment]] = {}
    for fragment in fragments:
        components.setdefault(find(fragment.fragment_id), []).append(fragment)
    ordered_components = sorted(
        components.values(),
        key=lambda items: (
            items[0].class_name,
            min(item.boundary.start_frame for item in items),
            min(_track_id_sort_key(track) for item in items for track in item.boundary.track_ids),
        ),
    )
    counters: dict[str, int] = {}
    entities: list[PersistentEntity] = []
    track_mapping: dict[TrackletKey, str] = {}
    fragment_mapping: dict[str, str] = {}
    canonical_by_fragment: dict[str, list[CanonicalObservation]] = {}
    for item in canonical_observations:
        canonical_by_fragment.setdefault(item.group_id, []).append(item)
    for component in ordered_components:
        component = sorted(component, key=_fragment_sort_key)
        class_name = component[0].class_name
        counters[class_name] = counters.get(class_name, 0) + 1
        persistent_id = f"obj:{class_name}:{counters[class_name]:04d}"
        for fragment in component:
            fragment_mapping[fragment.fragment_id] = persistent_id
            for key in fragment.tracklet_keys:
                track_mapping[key] = persistent_id
        provenance: list[dict[str, Any]] = []
        component_ids = {item.fragment_id for item in component}
        component_tracks = {track for item in component for track in item.boundary.track_ids}
        for duplicate in duplicate_candidates:
            if duplicate.accepted and duplicate.source_track_id in component_tracks and duplicate.target_track_id in component_tracks:
                provenance.append(duplicate.to_record())
        for continuation in accepted_continuations:
            if continuation.source_fragment_id in component_ids and continuation.target_fragment_id in component_ids:
                provenance.append(continuation.to_record())
        raw_count = sum(
            1
            for observation in observations
            if (observation.video_id, observation.class_name, observation.track_id)
            in {key for fragment in component for key in fragment.tracklet_keys}
        )
        canonical_count = sum(item.boundary.canonical_observation_count for item in component)
        entities.append(
            PersistentEntity(
                persistent_id=persistent_id,
                class_name=class_name,
                fragment_ids=tuple(item.fragment_id for item in component),
                track_ids=tuple(track for item in component for track in item.boundary.track_ids),
                start_time=min(item.boundary.start_time for item in component),
                end_time=max(item.boundary.end_time for item in component),
                observation_count=raw_count,
                canonical_observation_count=canonical_count,
                relation_provenance=tuple(sorted(provenance, key=lambda item: json.dumps(item, sort_keys=True))),
            )
        )
    return EntityBuild(
        entities=tuple(entities),
        track_to_persistent=track_mapping,
        fragment_to_persistent=fragment_mapping,
    )


def load_entity_consolidation_config(path: str | Path) -> EntityConsolidationConfig:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("R1 config must be a JSON object")
    return EntityConsolidationConfig.from_mapping(mapping)


def _continuation_gate_failure(
    source: EntityFragment,
    target: EntityFragment,
    *,
    temporal_gap: float,
    center_distance: float,
    area_ratio: float,
    appearance_similarity: float,
    config: EntityConsolidationConfig,
) -> str | None:
    if target.boundary.start_frame <= source.boundary.start_frame:
        return "not_forward_in_time"
    if temporal_gap > config.continuation_max_temporal_gap_seconds:
        return "temporal_gap"
    if center_distance > config.continuation_max_center_distance_normalized:
        return "center_distance"
    if not config.continuation_min_area_ratio <= area_ratio <= config.continuation_max_area_ratio:
        return "area_ratio"
    if appearance_similarity < config.continuation_min_appearance_similarity:
        return "appearance_similarity"
    return None


def _replace_continuation(candidate: ContinuationCandidate, **changes: Any) -> ContinuationCandidate:
    values = {**candidate.__dict__, **changes}
    return ContinuationCandidate(**values)


def _median_center(items: list[CanonicalObservation], width: int, height: int) -> tuple[float, float]:
    centers = [_center(item.observation.bbox, width, height) for item in items]
    return (float(median(item[0] for item in centers)), float(median(item[1] for item in centers)))


def _median_appearance(items: list[CanonicalObservation], appearances: dict[Any, tuple[float, ...]]) -> tuple[float, ...]:
    descriptors: list[tuple[float, ...]] = []
    for item in items:
        observation = item.observation
        key_candidates = [
            (item.group_id, observation.frame_index),
            (observation.video_id, observation.class_name, observation.frame_index),
        ]
        descriptor = next((appearances[key] for key in key_candidates if key in appearances), None)
        if descriptor is None:
            raise KeyError(f"missing appearance descriptor for {item.group_id} frame {observation.frame_index}")
        descriptors.append(descriptor)
    length = len(descriptors[0])
    if any(len(item) != length for item in descriptors):
        raise ValueError("boundary appearance descriptors have different lengths")
    return tuple(float(median(item[index] for item in descriptors)) for index in range(length))


def _center(bbox: Any, width: int, height: int) -> tuple[float, float]:
    return (((bbox.x1 + bbox.x2) / 2) / width, ((bbox.y1 + bbox.y2) / 2) / height)


def _iou(left: Any, right: Any) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _fragment_sort_key(fragment: EntityFragment) -> tuple[Any, ...]:
    return (
        fragment.class_name,
        fragment.boundary.start_frame,
        min((_track_id_sort_key(track) for track in fragment.boundary.track_ids), default=(1, "")),
        fragment.fragment_id,
    )


def _fragment_id_sort_key(fragment_id: str) -> tuple[int, int | str, str]:
    parts = fragment_id.rsplit(":", 1)
    if len(parts) == 2:
        try:
            return (0, int(parts[1]), fragment_id)
        except ValueError:
            pass
    return (1, fragment_id, fragment_id)


def _track_id_sort_key(track_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)
