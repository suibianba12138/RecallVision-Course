"""Strict, no-repair validation for EVAL-D0 manifests and annotations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    AGENT_ATTRIBUTION_STATUSES,
    ANSWERABILITY_STATUSES,
    ANNOTATION_SCHEMA_VERSION,
    EXPECTED_QUERY_STATUSES,
    HARD_NEGATIVE_TAGS,
    MANIFEST_SCHEMA_VERSION,
    MOVEMENT_SUPPORT_STATUSES,
    QUERY_TYPE_SET,
    QUERY_TYPES,
    SHA256_RE,
    SPLITS,
    TARGET_REFERENCE_STATUSES,
)


class EvaluationValidationError(ValueError):
    """Raised when an evaluation record violates its frozen contract."""


def validate_manifest(
    value: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    require_video_files: bool = False,
) -> None:
    """Validate one EVAL-D0 manifest without changing it or repairing fields."""

    _mapping(value, "manifest")
    _fields(
        value,
        {"schema_version", "pilot", "evaluation", "videos"},
        "manifest",
    )
    if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
        _error("manifest schema_version is not supported")

    pilot = _mapping(value["pilot"], "manifest pilot")
    _fields(
        pilot,
        {
            "target_video_count",
            "selected_eval_video_count",
            "annotation_dry_run_target_count",
            "annotation_dry_run_video_count",
            "fully_annotated_video_count",
            "data_collection_status",
            "annotation_schema_status",
        },
        "manifest pilot",
    )
    _positive_int(pilot["target_video_count"], "pilot.target_video_count", allow_zero=False)
    _positive_int(
        pilot["selected_eval_video_count"],
        "pilot.selected_eval_video_count",
        allow_zero=True,
    )
    _positive_int(
        pilot["annotation_dry_run_target_count"],
        "pilot.annotation_dry_run_target_count",
        allow_zero=False,
    )
    _positive_int(
        pilot["annotation_dry_run_video_count"],
        "pilot.annotation_dry_run_video_count",
        allow_zero=True,
    )
    _positive_int(
        pilot["fully_annotated_video_count"],
        "pilot.fully_annotated_video_count",
        allow_zero=True,
    )
    if pilot["data_collection_status"] not in {"INCOMPLETE", "READY"}:
        _error("pilot.data_collection_status is not allowed")
    if pilot["annotation_schema_status"] not in {"UNFROZEN", "FROZEN"}:
        _error("pilot.annotation_schema_status is not allowed")

    evaluation = _mapping(value["evaluation"], "manifest evaluation")
    _fields(
        evaluation,
        {"v1_method_head", "evaluation_protocol_head", "development_video_ids"},
        "manifest evaluation",
    )
    for field in ("v1_method_head", "evaluation_protocol_head"):
        _commit(evaluation[field], f"evaluation.{field}")
    development_ids = _string_list(evaluation["development_video_ids"], "development_video_ids")

    videos = value["videos"]
    if not isinstance(videos, list):
        _error("manifest videos must be a list")
    seen: set[str] = set()
    records: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(videos):
        path = f"videos[{index}]"
        record = _mapping(record, path)
        _validate_video_record(record, path, development_ids)
        video_id = record["video_id"]
        if video_id in seen:
            _error(f"duplicate video_id: {video_id}")
        seen.add(video_id)
        records[video_id] = record
        if require_video_files and repository_root is not None:
            video_path = Path(repository_root) / record["relative_video_path"]
            if not video_path.is_file():
                _error(f"video file is missing: {record['relative_video_path']}")

    selected_evals = [record for record in records.values() if record["split"] == "EVAL"]
    if len(selected_evals) > pilot["target_video_count"]:
        _error("manifest contains more EVAL videos than pilot.target_video_count")
    if len(selected_evals) != pilot["selected_eval_video_count"]:
        _error("pilot.selected_eval_video_count does not match manifest EVAL records")
    if pilot["fully_annotated_video_count"] > len(selected_evals):
        _error("pilot.fully_annotated_video_count exceeds selected EVAL videos")
    if pilot["annotation_dry_run_video_count"] > len(selected_evals):
        _error("pilot.annotation_dry_run_video_count exceeds selected EVAL videos")


def validate_eval_video_selection(
    manifest: Mapping[str, Any], video_ids: Sequence[str]
) -> None:
    """Reject any selection that is not explicitly marked EVAL."""

    validate_manifest(manifest)
    records = {record["video_id"]: record for record in manifest["videos"]}
    development_ids = set(manifest["evaluation"]["development_video_ids"])
    seen: set[str] = set()
    for video_id in video_ids:
        if not isinstance(video_id, str) or not video_id:
            _error("selected video IDs must be non-empty strings")
        if video_id in seen:
            _error(f"duplicate selected video_id: {video_id}")
        seen.add(video_id)
        if video_id in development_ids:
            _error(f"DEV_ONLY video cannot enter EVAL metrics: {video_id}")
        record = records.get(video_id)
        if record is None:
            _error(f"selected video is absent from manifest: {video_id}")
        if record["split"] != "EVAL":
            _error(f"selected video is not split EVAL: {video_id}")


def validate_annotation(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Validate one video annotation and all cross-references."""

    validate_manifest(manifest)
    _mapping(value, "annotation")
    _fields(
        value,
        {
            "schema_version",
            "annotator_metadata",
            "video",
            "objects",
            "agent_instances",
            "interaction_negatives",
            "queries",
            "ambiguity_notes",
        },
        "annotation",
    )
    if value["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        _error("annotation schema_version is not supported")
    _mapping(value["annotator_metadata"], "annotation annotator_metadata")

    video = _mapping(value["video"], "annotation video")
    _fields(
        video,
        {
            "video_id",
            "source_sha256",
            "duration_seconds",
            "fps",
            "frame_count",
            "width",
            "height",
        },
        "annotation video",
    )
    video_id = _nonempty_string(video["video_id"], "annotation video_id")
    record = _manifest_video(manifest, video_id)
    if video["source_sha256"] != record["sha256"]:
        _error("annotation source_sha256 does not match manifest")
    _finite_positive(video["duration_seconds"], "annotation duration_seconds")
    _finite_positive(video["fps"], "annotation fps")
    _positive_int(video["frame_count"], "annotation frame_count", allow_zero=False)
    _positive_int(video["width"], "annotation width", allow_zero=False)
    _positive_int(video["height"], "annotation height", allow_zero=False)
    if video["frame_count"] != record["frame_count"]:
        _error("annotation frame_count does not match manifest")

    agents = _validate_agents(value["agent_instances"])
    objects = _validate_objects(value["objects"], agents, video)
    events: dict[str, Mapping[str, Any]] = {}
    for obj in objects.values():
        for event in obj["movement_events"]:
            if event["event_id"] in events:
                _error(f"duplicate event_id: {event['event_id']}")
            events[event["event_id"]] = event
    _validate_interaction_negatives(value["interaction_negatives"], objects, agents, video)
    _validate_queries(value["queries"], video_id, objects, events, agents)
    _string_list(value["ambiguity_notes"], "annotation ambiguity_notes")


def _validate_video_record(
    record: Mapping[str, Any], path: str, development_ids: Sequence[str]
) -> None:
    _fields(
        record,
        {
            "video_id",
            "source_type",
            "source_provenance",
            "acquisition_date",
            "relative_video_path",
            "sha256",
            "duration_seconds",
            "fps",
            "frame_count",
            "width",
            "height",
            "split",
            "annotation_path",
            "eligible_query_types",
            "license_note",
            "notes",
        },
        path,
    )
    video_id = _nonempty_string(record["video_id"], f"{path}.video_id")
    _nonempty_string(record["source_type"], f"{path}.source_type")
    provenance = _mapping(record["source_provenance"], f"{path}.source_provenance")
    if not any(isinstance(item, str) and item.strip() for item in provenance.values()):
        _error(f"{path}.source_provenance must contain a non-empty source")
    acquisition = _nonempty_string(record["acquisition_date"], f"{path}.acquisition_date")
    relative = _nonempty_string(record["relative_video_path"], f"{path}.relative_video_path")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        _error(f"{path}.relative_video_path must be repository-relative")
    if not SHA256_RE.fullmatch(record["sha256"]):
        _error(f"{path}.sha256 must be a SHA-256 hex string")
    _finite_positive(record["duration_seconds"], f"{path}.duration_seconds")
    _finite_positive(record["fps"], f"{path}.fps")
    for field in ("frame_count", "width", "height"):
        _positive_int(record[field], f"{path}.{field}", allow_zero=False)
    if record["split"] not in SPLITS:
        _error(f"{path}.split is not allowed")
    if record["split"] == "EVAL" and not any(
        isinstance(provenance.get(field), str) and provenance[field].strip()
        for field in ("source_url", "source_uri")
    ):
        _error(f"{path}.source_provenance needs source_url or source_uri for EVAL")
    if (
        (len(acquisition) != 10 or acquisition[4] != "-" or acquisition[7] != "-")
        and not (record["split"] == "DEV" and acquisition == "UNKNOWN_HISTORICAL")
    ):
        _error(f"{path}.acquisition_date must be YYYY-MM-DD for EVAL records")
    if record["split"] == "EVAL" and video_id in development_ids:
        _error(f"DEV_ONLY video cannot be split EVAL: {video_id}")
    annotation_path = record["annotation_path"]
    if annotation_path is not None:
        annotation_path = _nonempty_string(annotation_path, f"{path}.annotation_path")
        if Path(annotation_path).is_absolute() or ".." in Path(annotation_path).parts:
            _error(f"{path}.annotation_path must be repository-relative")
    elif record["split"] == "EVAL":
        _error(f"{path}.annotation_path is required for EVAL videos")
    eligible = _string_list(record["eligible_query_types"], f"{path}.eligible_query_types")
    if not eligible or any(item not in QUERY_TYPE_SET for item in eligible):
        _error(f"{path}.eligible_query_types contains an unsupported query type")
    _nonempty_string(record["license_note"], f"{path}.license_note")
    _nonempty_string(record["notes"], f"{path}.notes")


def _validate_agents(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        _error("annotation agent_instances must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, agent in enumerate(value):
        agent = _mapping(agent, f"agent_instances[{index}]")
        _fields(agent, {"agent_instance_id", "agent_attribution_status"}, f"agent_instances[{index}]")
        agent_id = _nonempty_string(agent["agent_instance_id"], "agent_instance_id")
        if agent_id in result:
            _error(f"duplicate agent_instance_id: {agent_id}")
        if agent["agent_attribution_status"] not in AGENT_ATTRIBUTION_STATUSES:
            _error(f"agent attribution status is not allowed: {agent_id}")
        result[agent_id] = agent
    return result


def _validate_objects(
    value: Any,
    agents: Mapping[str, Mapping[str, Any]],
    video: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        _error("annotation objects must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    states: dict[str, Mapping[str, Any]] = {}
    for index, obj in enumerate(value):
        obj = _mapping(obj, f"objects[{index}]")
        _fields(
            obj,
            {
                "object_instance_id",
                "object_class",
                "description",
                "observation_status",
                "first_observed_frame",
                "first_observed_time",
                "last_observed_frame",
                "last_observed_time",
                "states",
                "movement_events",
            },
            f"objects[{index}]",
        )
        object_id = _nonempty_string(obj["object_instance_id"], "object_instance_id")
        if object_id in result:
            _error(f"duplicate object_instance_id: {object_id}")
        _nonempty_string(obj["object_class"], f"objects[{index}].object_class")
        _nonempty_string(obj["description"], f"objects[{index}].description")
        if obj["observation_status"] not in {"OBSERVED", "UNOBSERVED", "DISAPPEARED"}:
            _error(f"object observation_status is not allowed: {object_id}")
        _frame_time(obj["first_observed_frame"], obj["first_observed_time"], video, f"{object_id}.first_observed")
        _frame_time(obj["last_observed_frame"], obj["last_observed_time"], video, f"{object_id}.last_observed")
        if obj["first_observed_frame"] > obj["last_observed_frame"]:
            _error(f"object observation range is reversed: {object_id}")
        object_states: list[Mapping[str, Any]] = []
        for state_index, state in enumerate(obj["states"]):
            state = _mapping(state, f"{object_id}.states[{state_index}]")
            _fields(
                state,
                {
                    "state_id",
                    "zone",
                    "start_frame",
                    "end_frame",
                    "start_time",
                    "end_time",
                    "evidence_frames",
                    "ambiguity_status",
                    "notes",
                },
                f"{object_id}.states[{state_index}]",
            )
            state_id = _nonempty_string(state["state_id"], "state_id")
            if state_id in states:
                _error(f"duplicate state_id: {state_id}")
            _nonempty_string(state["zone"], f"{state_id}.zone")
            _range(state["start_frame"], state["end_frame"], video, state_id)
            _frame_time(state["start_frame"], state["start_time"], video, f"{state_id}.start")
            _frame_time(state["end_frame"], state["end_time"], video, f"{state_id}.end")
            evidence = _int_list(state["evidence_frames"], f"{state_id}.evidence_frames")
            if not evidence or any(item < state["start_frame"] or item > state["end_frame"] for item in evidence):
                _error(f"state evidence_frames must lie inside the state range: {state_id}")
            if state["ambiguity_status"] not in {"CLEAR", "AMBIGUOUS", "INSUFFICIENT"}:
                _error(f"state ambiguity_status is not allowed: {state_id}")
            _nonempty_string(state["notes"], f"{state_id}.notes")
            states[state_id] = state
            object_states.append(state)
        events = _validate_events(obj["movement_events"], object_id, object_states, states, agents, video)
        obj_copy = dict(obj)
        obj_copy["movement_events"] = events
        result[object_id] = obj_copy
    return result


def _validate_events(
    value: Any,
    object_id: str,
    object_states: Sequence[Mapping[str, Any]],
    all_states: Mapping[str, Mapping[str, Any]],
    agents: Mapping[str, Mapping[str, Any]],
    video: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        _error(f"movement_events must be a list: {object_id}")
    result: list[Mapping[str, Any]] = []
    local_ids: set[str] = set()
    for index, event in enumerate(value):
        event = _mapping(event, f"{object_id}.movement_events[{index}]")
        _fields(
            event,
            {
                "event_id",
                "event_type",
                "object_instance_id",
                "from_state_id",
                "to_state_id",
                "temporal_status",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "evidence_frames",
                "movement_support_status",
                "agent_instance_ids",
                "agent_attribution_status",
            },
            f"{object_id}.movement_events[{index}]",
        )
        event_id = _nonempty_string(event["event_id"], "event_id")
        if event_id in local_ids:
            _error(f"duplicate event_id: {event_id}")
        local_ids.add(event_id)
        if event["event_type"] != "MOVE_OBJECT":
            _error(f"movement event type is not MOVE_OBJECT: {event_id}")
        if event["object_instance_id"] != object_id:
            _error(f"movement event object reference does not match owner: {event_id}")
        from_id = _nonempty_string(event["from_state_id"], "from_state_id")
        to_id = _nonempty_string(event["to_state_id"], "to_state_id")
        if from_id not in all_states or to_id not in all_states:
            _error(f"movement event state reference is unresolved: {event_id}")
        _range(event["start_frame"], event["end_frame"], video, event_id)
        _frame_time(event["start_frame"], event["start_time"], video, f"{event_id}.start")
        _frame_time(event["end_frame"], event["end_time"], video, f"{event_id}.end")
        evidence = _int_list(event["evidence_frames"], f"{event_id}.evidence_frames")
        if not evidence or any(item < event["start_frame"] or item > event["end_frame"] for item in evidence):
            _error(f"movement evidence_frames must lie inside event range: {event_id}")
        if event["temporal_status"] not in {"BOUNDED_INTERVAL", "ANNOTATOR_SUPPORTED_BOUNDS"}:
            _error(f"movement temporal_status is not allowed: {event_id}")
        if event["movement_support_status"] not in MOVEMENT_SUPPORT_STATUSES:
            _error(f"movement_support_status is not allowed: {event_id}")
        if event["agent_attribution_status"] not in AGENT_ATTRIBUTION_STATUSES:
            _error(f"agent_attribution_status is not allowed: {event_id}")
        agent_ids = _string_list(event["agent_instance_ids"], f"{event_id}.agent_instance_ids")
        if any(agent_id not in agents for agent_id in agent_ids):
            _error(f"movement event has unresolved agent reference: {event_id}")
        status = event["agent_attribution_status"]
        if status == "SUPPORTED" and not agent_ids:
            _error(f"SUPPORTED movement event requires an agent: {event_id}")
        if status in {"NO_VISIBLE_AGENT", "INSUFFICIENT"} and agent_ids:
            _error(f"{status} movement event cannot contain agent IDs: {event_id}")
        from_state = all_states[from_id]
        to_state = all_states[to_id]
        if from_state["end_frame"] > event["start_frame"]:
            _error(f"pre-event state does not precede movement event: {event_id}")
        if to_state["start_frame"] < event["end_frame"]:
            _error(f"post-event state does not follow movement event: {event_id}")
        result.append(event)
    return result


def _validate_interaction_negatives(
    value: Any,
    objects: Mapping[str, Mapping[str, Any]],
    agents: Mapping[str, Mapping[str, Any]],
    video: Mapping[str, Any],
) -> None:
    if not isinstance(value, list):
        _error("interaction_negatives must be a list")
    seen: set[str] = set()
    for index, item in enumerate(value):
        item = _mapping(item, f"interaction_negatives[{index}]")
        _fields(
            item,
            {
                "interaction_id",
                "object_instance_id",
                "person_instance_ids",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "movement_present",
                "hard_negative_tags",
                "notes",
            },
            f"interaction_negatives[{index}]",
        )
        interaction_id = _nonempty_string(item["interaction_id"], "interaction_id")
        if interaction_id in seen:
            _error(f"duplicate interaction_id: {interaction_id}")
        seen.add(interaction_id)
        object_id = _nonempty_string(item["object_instance_id"], "interaction object_instance_id")
        if object_id not in objects:
            _error(f"interaction has unresolved object reference: {interaction_id}")
        people = _string_list(item["person_instance_ids"], f"{interaction_id}.person_instance_ids")
        if any(person_id not in agents for person_id in people):
            _error(f"interaction has unresolved person reference: {interaction_id}")
        _range(item["start_frame"], item["end_frame"], video, interaction_id)
        _frame_time(item["start_frame"], item["start_time"], video, f"{interaction_id}.start")
        _frame_time(item["end_frame"], item["end_time"], video, f"{interaction_id}.end")
        if item["movement_present"] is not False:
            _error(f"stationary interaction must set movement_present=false: {interaction_id}")
        tags = _string_list(item["hard_negative_tags"], f"{interaction_id}.hard_negative_tags")
        if "STATIONARY_INTERACTION" not in tags or any(tag not in HARD_NEGATIVE_TAGS for tag in tags):
            _error(f"interaction has invalid hard-negative tags: {interaction_id}")
        _nonempty_string(item["notes"], f"{interaction_id}.notes")


def _validate_queries(
    value: Any,
    video_id: str,
    objects: Mapping[str, Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
    agents: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list):
        _error("annotation queries must be a list")
    seen: set[str] = set()
    for index, query in enumerate(value):
        query = _mapping(query, f"queries[{index}]")
        _fields(
            query,
            {
                "query_id",
                "video_id",
                "query_type",
                "object_instance_id",
                "target_reference_status",
                "natural_language_question",
                "reference_event_id",
                "answerability",
                "expected_status",
                "ground_truth",
                "evidence_refs",
                "hard_negative_tags",
                "annotation_notes",
            },
            f"queries[{index}]",
        )
        query_id = _nonempty_string(query["query_id"], "query_id")
        if query_id in seen:
            _error(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if query["video_id"] != video_id:
            _error(f"query video_id does not match annotation video: {query_id}")
        if query["query_type"] not in QUERY_TYPE_SET:
            _error(f"query type is not supported: {query_id}")
        object_id = query["object_instance_id"]
        reference_status = query["target_reference_status"]
        if reference_status not in TARGET_REFERENCE_STATUSES:
            _error(f"query target_reference_status is not allowed: {query_id}")
        if reference_status == "KNOWN_OBJECT":
            if not isinstance(object_id, str) or object_id not in objects:
                _error(f"query has unresolved object reference: {query_id}")
        elif object_id is not None:
            _error(f"UNKNOWN_OBJECT query must have null object_instance_id: {query_id}")
        _nonempty_string(query["natural_language_question"], f"{query_id}.question")
        reference_event = query["reference_event_id"]
        if reference_event is not None:
            if not isinstance(reference_event, str) or reference_event not in events:
                _error(f"query has unresolved event reference: {query_id}")
        if query["answerability"] not in ANSWERABILITY_STATUSES:
            _error(f"query answerability is not allowed: {query_id}")
        if query["expected_status"] not in EXPECTED_QUERY_STATUSES:
            _error(f"query expected_status is not allowed: {query_id}")
        if query["answerability"] == "ANSWERABLE" and query["expected_status"] != "ANSWER":
            _error(f"ANSWERABLE query must expect ANSWER: {query_id}")
        if query["answerability"] != "ANSWERABLE" and query["expected_status"] != "ABSTAIN":
            _error(f"non-answerable query must expect ABSTAIN: {query_id}")
        ground_truth = _mapping(query["ground_truth"], f"{query_id}.ground_truth")
        evidence = query["evidence_refs"]
        if not isinstance(evidence, list):
            _error(f"query evidence_refs must be a list: {query_id}")
        if query["answerability"] == "ANSWERABLE" and not evidence:
            _error(f"answerable query requires evidence_refs: {query_id}")
        tags = _string_list(query["hard_negative_tags"], f"{query_id}.hard_negative_tags")
        if any(tag not in HARD_NEGATIVE_TAGS for tag in tags):
            _error(f"query has unsupported hard-negative tag: {query_id}")
        _nonempty_string(query["annotation_notes"], f"{query_id}.annotation_notes")
        if query["query_type"] == "WHO_MOVED_OBJECT" and reference_event in events:
            event = events[reference_event]
            for agent_id in event["agent_instance_ids"]:
                if agent_id not in agents:
                    _error(f"WHO query event has unresolved agent: {query_id}")
            ground_truth_agents = ground_truth.get("agent_instance_ids", [])
            if not isinstance(ground_truth_agents, list) or any(
                agent_id not in agents for agent_id in ground_truth_agents
            ):
                _error(f"WHO query ground truth has unresolved agent: {query_id}")
            if query["answerability"] == "ANSWERABLE" and set(ground_truth_agents) != set(
                event["agent_instance_ids"]
            ):
                _error(f"WHO query ground truth agents do not match event attribution: {query_id}")
        if query["answerability"] == "ANSWERABLE" and reference_event is not None:
            if ground_truth.get("event_id") != reference_event:
                _error(f"query ground_truth event_id does not match reference_event_id: {query_id}")
        if query["answerability"] == "ANSWERABLE" and query["query_type"] in {
            "WHEN_OBJECT_MOVED",
            "WHERE_OBJECT_BEFORE",
            "WHO_MOVED_OBJECT",
        } and reference_event is None:
            _error(f"answerable event query requires reference_event_id: {query_id}")
        if query["query_type"] == "WHERE_OBJECT" and object_id in objects:
            state_ids = {state["state_id"] for state in objects[object_id]["states"]}
            state_id = ground_truth.get("state_id")
            if query["answerability"] == "ANSWERABLE" and state_id not in state_ids:
                _error(f"WHERE query ground truth state is unresolved: {query_id}")
        if query["query_type"] == "WHERE_OBJECT_BEFORE" and object_id in objects:
            state_ids = {state["state_id"] for state in objects[object_id]["states"]}
            state_id = ground_truth.get("state_id")
            if query["answerability"] == "ANSWERABLE" and state_id not in state_ids:
                _error(f"WHERE_BEFORE ground truth state is unresolved: {query_id}")


def _manifest_video(manifest: Mapping[str, Any], video_id: str) -> Mapping[str, Any]:
    matches = [record for record in manifest["videos"] if record["video_id"] == video_id]
    if len(matches) != 1:
        _error(f"annotation video is absent or duplicated in manifest: {video_id}")
    return matches[0]


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        _error(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        _error(f"{label} has unknown fields: {sorted(unknown)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        _error(f"{label} must be a list of non-empty strings")
    return value


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _error(f"{label} must be a list of integers")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or (value < 0 if allow_zero else value <= 0):
        _error(f"{label} must be an integer in the allowed range")


def _finite_positive(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        _error(f"{label} must be finite and positive")


def _frame_time(frame: Any, timestamp: Any, video: Mapping[str, Any], label: str) -> None:
    _positive_int(frame, f"{label}_frame", allow_zero=True)
    if frame >= video["frame_count"]:
        _error(f"{label}_frame is outside video bounds")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        _error(f"{label}_time must be finite")
    if timestamp < 0 or timestamp > video["duration_seconds"]:
        _error(f"{label}_time is outside video bounds")
    expected_timestamp = frame / video["fps"]
    if not math.isclose(timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-6):
        _error(f"{label}_time does not match frame/fps")


def _range(start: Any, end: Any, video: Mapping[str, Any], label: str) -> None:
    _positive_int(start, f"{label}.start_frame", allow_zero=True)
    _positive_int(end, f"{label}.end_frame", allow_zero=True)
    if start > end:
        _error(f"{label} frame range is reversed")
    if end >= video["frame_count"]:
        _error(f"{label} frame range exceeds video bounds")


def _commit(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        _error(f"{label} must be a 40-character lowercase Git commit")


def _error(message: str) -> None:
    raise EvaluationValidationError(message)
