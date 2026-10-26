"""Run P2B state timelines and last-seen WHERE_OBJECT retrieval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from recallvision.memory.schema import Observation
from recallvision.memory.state_timeline import (
    LastSeenState,
    ObjectStateTimeline,
    StateTimelineConfig,
    build_state_timelines,
    select_last_seen_state,
)
from recallvision.memory.store import ObservationMemoryStore
from recallvision.query.object_state_location import (
    StateQueryResult,
    StateAnswerStatus,
    answer_where_object_state,
)
from recallvision.query.parser import parse_query
from recallvision.query.schema import Query, QueryType
from recallvision.query.target_resolution import build_target_reference


def run(
    *,
    consolidated_memory: Path,
    persistent_entities: Path,
    entity_summary: Path,
    config_path: Path,
    output_dir: Path,
    question: str | None = None,
    target_class: str | None = None,
    target_entity_id: str | None = None,
    video_end_timestamp: float | None = None,
) -> dict[str, Any]:
    """Build P2B outputs from frozen R1 artifacts without video processing."""

    observations = ObservationMemoryStore(consolidated_memory).load_all()
    if not observations:
        raise ValueError(f"consolidated memory is empty: {consolidated_memory}")
    entity_records = _load_json_list(persistent_entities, "persistent_entities")
    summary = _load_json_object(entity_summary, "entity_consolidation_summary")
    config = StateTimelineConfig.from_path(config_path)
    duration = (
        float(video_end_timestamp)
        if video_end_timestamp is not None
        else _video_duration(summary)
    )
    _validate_r1_inputs(observations, entity_records, summary)

    timelines = build_state_timelines(observations, config)
    last_seen_states = [
        select_last_seen_state(timeline, video_end_timestamp=duration)
        for timeline in timelines
        if timeline.segments
    ]
    if not last_seen_states:
        raise ValueError("no retained object state segments were produced")

    video_id = observations[0].video_id
    query_result = None
    if question is not None:
        query = parse_query(question, video_id=video_id, question_id="p2b-query")
        if query.query_type is not QueryType.WHERE_OBJECT:
            raise ValueError("P2B accepts WHERE_OBJECT questions only")
        if target_class is not None:
            reference = build_target_reference(target_class)
            query = replace(query, target_class=reference.canonical_label, target_reference=reference)
        if target_entity_id is not None:
            query = replace(query, target_entity_ids=[target_entity_id])
        query_result = answer_where_object_state(query, timelines, video_end_timestamp=duration)
    elif target_class is not None or target_entity_id is not None:
        reference = build_target_reference(target_class) if target_class is not None else None
        query = Query(
            question_id="p2b-query",
            video_id=video_id,
            question=f"Where is {target_class or target_entity_id}?",
            query_type=QueryType.WHERE_OBJECT,
            target_entity_ids=[target_entity_id] if target_entity_id else [],
            target_class=reference.canonical_label if reference else None,
            target_reference=reference,
        )
        query_result = answer_where_object_state(query, timelines, video_end_timestamp=duration)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write P2B artifacts into a non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "object_state_timelines.json", [item.to_record() for item in timelines])
    _write_json(output_dir / "last_seen_states.json", [item.to_record() for item in last_seen_states])
    if query_result is not None:
        _write_json(output_dir / "query_result.json", query_result.to_record())
    _write_text(output_dir / "state_timeline_audit.txt", _timeline_audit(timelines))

    summary_record = _build_summary(
        observations=observations,
        entity_records=entity_records,
        source_summary=summary,
        config=config,
        timelines=timelines,
        last_seen_states=last_seen_states,
        query_result=query_result,
        consolidated_memory=consolidated_memory,
        persistent_entities=persistent_entities,
        entity_summary=entity_summary,
        config_path=config_path,
        video_end_timestamp=duration,
    )
    _write_json(output_dir / "state_timeline_summary.json", summary_record)
    return summary_record


def _build_summary(
    *,
    observations: list[Observation],
    entity_records: list[dict[str, Any]],
    source_summary: dict[str, Any],
    config: StateTimelineConfig,
    timelines: list[ObjectStateTimeline],
    last_seen_states: list[LastSeenState],
    query_result: StateQueryResult,
    consolidated_memory: Path,
    persistent_entities: Path,
    entity_summary: Path,
    config_path: Path,
    video_end_timestamp: float,
) -> dict[str, Any]:
    timelines_by_class = dict(Counter(item.class_name for item in timelines))
    return {
        "p2b_status": "PASS" if timelines else "NO_TIMELINES",
        "pipeline": [
            "Persistent Entity",
            "Consolidated Observations",
            "Object State Timeline",
            "Last-Seen State",
            "Answer",
        ],
        "authoritative_inputs": {
            "consolidated_memory": consolidated_memory.name,
            "persistent_entities": persistent_entities.name,
            "entity_summary": entity_summary.name,
            "video_sha256": source_summary.get("video_sha256"),
        },
        "config_file": config_path.name,
        "config": config.to_record(),
        "video_id": observations[0].video_id,
        "identity_review_status": source_summary.get("p2a_r1_identity_validation_status"),
        "video_end_timestamp": video_end_timestamp,
        "consolidated_observation_count": len(observations),
        "persistent_entity_count": len(entity_records),
        "timeline_count": len(timelines),
        "timelines_by_class": timelines_by_class,
        "timelines": [item.to_record() for item in timelines],
        "segment_count": sum(len(item.segments) for item in timelines),
        "last_seen_state_count": len(last_seen_states),
        "query": query_result.to_record() if query_result is not None else {"status": "NOT_REQUESTED"},
        "events_generated": [],
        "training_used": False,
        "perception_rerun": False,
        "vlm_used": False,
    }


def _validate_r1_inputs(
    observations: list[Observation],
    entity_records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    entity_ids = {
        record.get("persistent_id")
        for record in entity_records
        if isinstance(record.get("persistent_id"), str)
    }
    observation_ids = {observation.persistent_id for observation in observations}
    if not observation_ids.issubset(entity_ids):
        raise ValueError("consolidated memory contains persistent IDs absent from persistent_entities.json")
    if summary.get("p2a_r1_pipeline_status") != "PASS":
        raise ValueError("R1 summary does not report pipeline PASS")
    if not summary.get("video_sha256"):
        raise ValueError("R1 summary is missing video_sha256")


def _video_duration(summary: dict[str, Any]) -> float:
    metadata = summary.get("video_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("duration"), (int, float)):
        raise ValueError("R1 summary is missing video_metadata.duration")
    return float(metadata["duration"])


def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _timeline_audit(timelines: list[ObjectStateTimeline]) -> str:
    lines = [
        "P2B observed object state timelines",
        "No event records are generated; segments describe only observed zones.",
    ]
    for timeline in timelines:
        lines.append(f"persistent_id={timeline.persistent_id},class_name={timeline.class_name}")
        for index, segment in enumerate(timeline.segments, start=1):
            lines.append(
                "segment="
                f"{index},start_frame={segment.start_frame},end_frame={segment.end_frame},"
                f"start_time={segment.start_time},end_time={segment.end_time},zone={segment.zone},"
                f"observation_count={segment.observation_count},"
                f"source_track_ids={','.join(segment.source_track_ids)},"
                f"source_frame_indices={','.join(str(item) for item in segment.source_frame_indices)}"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated-memory", type=Path, required=True)
    parser.add_argument("--persistent-entities", type=Path, required=True)
    parser.add_argument("--entity-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question")
    parser.add_argument("--target-class")
    parser.add_argument("--target-entity-id")
    parser.add_argument("--video-end-timestamp", type=float)
    arguments = parser.parse_args()
    result = run(
        consolidated_memory=arguments.consolidated_memory,
        persistent_entities=arguments.persistent_entities,
        entity_summary=arguments.entity_summary,
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        question=arguments.question,
        target_class=arguments.target_class,
        target_entity_id=arguments.target_entity_id,
        video_end_timestamp=arguments.video_end_timestamp,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["p2b_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
