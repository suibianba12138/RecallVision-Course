"""Persistent object identities built from accepted P2A association edges."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from recallvision.memory.association import AssociationCandidate
from recallvision.memory.schema import Observation
from recallvision.memory.tracklets import TrackletKey, TrackletSummary


@dataclass(frozen=True)
class PersistentObject:
    """A deterministic identity chain; no event semantics are attached."""

    persistent_id: str
    class_name: str
    track_ids: tuple[str, ...]
    start_time: float
    end_time: float
    observation_count: int
    zone_history: tuple[tuple[float, str | None], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_name": self.class_name,
            "track_ids": list(self.track_ids),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "observation_count": self.observation_count,
            "zone_history": [
                {"timestamp": timestamp, "zone": zone}
                for timestamp, zone in self.zone_history
            ],
        }


@dataclass(frozen=True)
class PersistentObjectBuild:
    objects: tuple[PersistentObject, ...]
    mapping: dict[TrackletKey, str]


def build_persistent_objects(
    summaries: list[TrackletSummary] | tuple[TrackletSummary, ...],
    observations: list[Observation] | tuple[Observation, ...],
    accepted_edges: list[AssociationCandidate] | tuple[AssociationCandidate, ...],
) -> PersistentObjectBuild:
    """Construct acyclic chains and assign stable per-class object IDs."""

    summary_by_key = {summary.key: summary for summary in summaries}
    successors: dict[TrackletKey, TrackletKey] = {
        edge.source_key: edge.target_key for edge in accepted_edges if edge.accepted
    }
    predecessors: dict[TrackletKey, TrackletKey] = {
        edge.target_key: edge.source_key for edge in accepted_edges if edge.accepted
    }
    if len(successors) != len([edge for edge in accepted_edges if edge.accepted]):
        raise ValueError("association graph has more than one successor")
    if len(predecessors) != len([edge for edge in accepted_edges if edge.accepted]):
        raise ValueError("association graph has more than one predecessor")
    for edge in accepted_edges:
        if not edge.accepted:
            continue
        if edge.source_key[1] != edge.target_key[1] or edge.class_name != edge.source_key[1]:
            raise ValueError("cross-class association is not allowed")
        if (
            edge.source_key in summary_by_key
            and edge.target_key in summary_by_key
            and summary_by_key[edge.target_key].start_frame
            <= summary_by_key[edge.source_key].start_frame
        ):
            raise ValueError("association edge is not forward in time")

    roots = [key for key in summary_by_key if key not in predecessors]
    roots.sort(key=lambda key: (key[1], summary_by_key[key].start_frame, key[2], key[0]))
    chains: list[list[TrackletKey]] = []
    visited: set[TrackletKey] = set()
    for root in roots:
        chain: list[TrackletKey] = []
        current: TrackletKey | None = root
        while current is not None:
            if current in visited:
                raise ValueError("association graph contains a cycle")
            if current not in summary_by_key:
                raise ValueError(f"association references unknown tracklet {current}")
            visited.add(current)
            chain.append(current)
            current = successors.get(current)
        chains.append(chain)
    if visited != set(summary_by_key):
        raise ValueError("association graph contains a cycle or disconnected invalid edge")

    observations_by_key: dict[TrackletKey, list[Observation]] = {}
    for observation in observations:
        key = (observation.video_id, observation.class_name, observation.track_id)
        observations_by_key.setdefault(key, []).append(observation)
    class_counts: dict[str, int] = {}
    mapping: dict[TrackletKey, str] = {}
    objects: list[PersistentObject] = []
    for chain in chains:
        class_name = chain[0][1]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        persistent_id = f"obj:{class_name}:{class_counts[class_name]:04d}"
        for key in chain:
            mapping[key] = persistent_id
        chain_summaries = [summary_by_key[key] for key in chain]
        chain_observations = sorted(
            (item for key in chain for item in observations_by_key.get(key, [])),
            key=lambda item: (item.timestamp, item.frame_index, item.track_id),
        )
        objects.append(
            PersistentObject(
                persistent_id=persistent_id,
                class_name=class_name,
                track_ids=tuple(summary.track_id for summary in chain_summaries),
                start_time=min(summary.start_time for summary in chain_summaries),
                end_time=max(summary.end_time for summary in chain_summaries),
                observation_count=sum(summary.observation_count for summary in chain_summaries),
                zone_history=tuple(
                    (item.timestamp, item.zone) for item in chain_observations
                ),
            )
        )
    return PersistentObjectBuild(objects=tuple(objects), mapping=mapping)


def enrich_observations(
    observations: list[Observation] | tuple[Observation, ...],
    mapping: dict[TrackletKey, str],
) -> list[Observation]:
    """Copy observations while changing only ``persistent_id``."""

    enriched: list[Observation] = []
    for observation in observations:
        key = (observation.video_id, observation.class_name, observation.track_id)
        if key not in mapping:
            raise KeyError(f"missing persistent identity for {key}")
        enriched.append(replace(observation, persistent_id=mapping[key]))
    return enriched
