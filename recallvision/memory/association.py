"""Deterministic, training-free association of compatible tracklets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from recallvision.memory.appearance import histogram_intersection_similarity
from recallvision.memory.tracklets import TrackletKey, TrackletSummary


@dataclass(frozen=True)
class AssociationConfig:
    max_temporal_gap_seconds: float = 1.25
    max_center_distance_normalized: float = 0.35
    min_area_ratio: float = 0.40
    max_area_ratio: float = 2.50
    min_appearance_similarity: float = 0.35
    max_temporal_overlap_frames: int = 2
    sigma_d: float = 0.25
    sigma_s: float = 0.70
    association_threshold: float = 0.65
    appearance_h_bins: int = 32
    appearance_s_bins: int = 32

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "AssociationConfig":
        return cls(**mapping)

    def to_record(self) -> dict[str, Any]:
        return {
            "max_temporal_gap_seconds": self.max_temporal_gap_seconds,
            "max_center_distance_normalized": self.max_center_distance_normalized,
            "min_area_ratio": self.min_area_ratio,
            "max_area_ratio": self.max_area_ratio,
            "min_appearance_similarity": self.min_appearance_similarity,
            "max_temporal_overlap_frames": self.max_temporal_overlap_frames,
            "sigma_d": self.sigma_d,
            "sigma_s": self.sigma_s,
            "association_threshold": self.association_threshold,
            "appearance_h_bins": self.appearance_h_bins,
            "appearance_s_bins": self.appearance_s_bins,
        }


@dataclass(frozen=True)
class AssociationCandidate:
    source_key: TrackletKey
    target_key: TrackletKey
    source_track_id: str
    target_track_id: str
    class_name: str
    temporal_gap: float
    center_distance: float
    area_ratio: float
    appearance_similarity: float
    spatial_similarity: float
    scale_similarity: float
    class_consistency: bool
    temporal_overlap_frames: int
    score: float
    hard_gate_pass: bool
    accepted: bool = False
    rejection_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "source_track_id": self.source_track_id,
            "target_track_id": self.target_track_id,
            "source_key": list(self.source_key),
            "target_key": list(self.target_key),
            "class_name": self.class_name,
            "temporal_gap": self.temporal_gap,
            "center_distance": self.center_distance,
            "area_ratio": self.area_ratio,
            "appearance_similarity": self.appearance_similarity,
            "spatial_similarity": self.spatial_similarity,
            "scale_similarity": self.scale_similarity,
            "class_consistency": self.class_consistency,
            "temporal_overlap_frames": self.temporal_overlap_frames,
            "score": self.score,
            "hard_gate_pass": self.hard_gate_pass,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class AssociationSelection:
    all_edges: tuple[AssociationCandidate, ...]
    accepted_edges: tuple[AssociationCandidate, ...]

    def to_records(self) -> list[dict[str, Any]]:
        return [edge.to_record() for edge in self.all_edges]


def compute_association_features(
    source: TrackletSummary,
    target: TrackletSummary,
    *,
    appearance_similarity: float,
    config: AssociationConfig,
) -> AssociationCandidate:
    """Compute all documented features and apply hard gates."""

    class_consistency = source.class_name == target.class_name
    temporal_gap = target.start_time - source.end_time
    temporal_overlap_frames = max(0, source.end_frame - target.start_frame + 1)
    dx = source.last_center_normalized[0] - target.first_center_normalized[0]
    dy = source.last_center_normalized[1] - target.first_center_normalized[1]
    center_distance = math.hypot(dx, dy)
    area_ratio = (
        target.median_area / source.median_area if source.median_area > 0.0 else math.inf
    )
    spatial_similarity = math.exp(-center_distance / config.sigma_d)
    scale_similarity = (
        math.exp(-abs(math.log(area_ratio)) / config.sigma_s)
        if area_ratio > 0.0 and math.isfinite(area_ratio)
        else 0.0
    )
    score = (
        0.50 * appearance_similarity
        + 0.30 * spatial_similarity
        + 0.20 * scale_similarity
    )
    rejection_reason = _first_gate_failure(
        source,
        target,
        class_consistency=class_consistency,
        temporal_gap=temporal_gap,
        temporal_overlap_frames=temporal_overlap_frames,
        center_distance=center_distance,
        area_ratio=area_ratio,
        appearance_similarity=appearance_similarity,
        config=config,
    )
    return AssociationCandidate(
        source_key=source.key,
        target_key=target.key,
        source_track_id=source.track_id,
        target_track_id=target.track_id,
        class_name=source.class_name,
        temporal_gap=temporal_gap,
        center_distance=center_distance,
        area_ratio=area_ratio,
        appearance_similarity=appearance_similarity,
        spatial_similarity=spatial_similarity,
        scale_similarity=scale_similarity,
        class_consistency=class_consistency,
        temporal_overlap_frames=temporal_overlap_frames,
        score=score,
        hard_gate_pass=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def associate_tracklets(
    summaries: list[TrackletSummary] | tuple[TrackletSummary, ...],
    descriptors: dict[TrackletKey, tuple[float, ...]],
    *,
    config: AssociationConfig = AssociationConfig(),
) -> AssociationSelection:
    """Build all same-class forward candidates and select valid edges."""

    ordered = sorted(summaries, key=lambda item: item.key)
    candidates: list[AssociationCandidate] = []
    for source in ordered:
        for target in ordered:
            if source.key == target.key or source.video_id != target.video_id:
                continue
            if source.class_name != target.class_name:
                continue
            if target.start_frame <= source.start_frame:
                continue
            if source.key not in descriptors or target.key not in descriptors:
                raise KeyError(f"missing appearance descriptor for {source.key} or {target.key}")
            similarity = histogram_intersection_similarity(
                descriptors[source.key], descriptors[target.key]
            )
            candidates.append(
                compute_association_features(
                    source,
                    target,
                    appearance_similarity=similarity,
                    config=config,
                )
            )
    return select_association_edges(candidates, config=config)


def select_association_edges(
    candidates: list[AssociationCandidate] | tuple[AssociationCandidate, ...],
    *,
    config: AssociationConfig,
) -> AssociationSelection:
    """Apply threshold, one-predecessor/successor, and deterministic tie breaks."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.hard_gate_pass and candidate.score >= config.association_threshold
    ]
    eligible.sort(
        key=lambda item: (
            -item.score,
            _track_id_sort_key(item.source_track_id),
            _track_id_sort_key(item.target_track_id),
            item.source_key,
            item.target_key,
        )
    )
    used_sources: set[TrackletKey] = set()
    used_targets: set[TrackletKey] = set()
    accepted_keys: set[tuple[TrackletKey, TrackletKey]] = set()
    for candidate in eligible:
        if candidate.source_key in used_sources or candidate.target_key in used_targets:
            continue
        used_sources.add(candidate.source_key)
        used_targets.add(candidate.target_key)
        accepted_keys.add((candidate.source_key, candidate.target_key))

    finalized: list[AssociationCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.source_key[:2],
            _track_id_sort_key(item.source_track_id),
            item.target_key[:2],
            _track_id_sort_key(item.target_track_id),
        ),
    ):
        key = (candidate.source_key, candidate.target_key)
        if key in accepted_keys:
            finalized.append(replace(candidate, accepted=True, rejection_reason=None))
        elif not candidate.hard_gate_pass:
            finalized.append(candidate)
        elif candidate.score < config.association_threshold:
            finalized.append(replace(candidate, rejection_reason="score_below_threshold"))
        elif candidate.source_key in used_sources:
            finalized.append(replace(candidate, rejection_reason="one_successor"))
        elif candidate.target_key in used_targets:
            finalized.append(replace(candidate, rejection_reason="one_predecessor"))
        else:  # pragma: no cover - defensive consistency branch
            finalized.append(replace(candidate, rejection_reason="not_selected"))
    accepted = tuple(edge for edge in finalized if edge.accepted)
    return AssociationSelection(all_edges=tuple(finalized), accepted_edges=accepted)


def load_association_config(path: str | Path) -> AssociationConfig:
    """Load JSON-compatible YAML using only the standard library."""

    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("association config must be a JSON object")
    return AssociationConfig.from_mapping(mapping)


def _first_gate_failure(
    source: TrackletSummary,
    target: TrackletSummary,
    *,
    class_consistency: bool,
    temporal_gap: float,
    temporal_overlap_frames: int,
    center_distance: float,
    area_ratio: float,
    appearance_similarity: float,
    config: AssociationConfig,
) -> str | None:
    if not class_consistency:
        return "class_mismatch"
    if target.start_frame <= source.start_frame or target.start_time < source.start_time:
        return "not_forward_in_time"
    if temporal_overlap_frames > config.max_temporal_overlap_frames:
        return "temporal_overlap"
    if temporal_gap > config.max_temporal_gap_seconds:
        return "temporal_gap"
    if center_distance > config.max_center_distance_normalized:
        return "center_distance"
    if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
        return "area_ratio"
    if appearance_similarity < config.min_appearance_similarity:
        return "appearance_similarity"
    return None


def _track_id_sort_key(track_id: str) -> tuple[int, int | str]:
    """Sort numeric tracker IDs numerically and other IDs lexicographically."""

    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)
