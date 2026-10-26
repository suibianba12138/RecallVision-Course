"""Deterministic structured query-ground-truth generation for EVAL-D0."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import QUERY_TYPES


def generate_query_ground_truth(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate controlled four-family query records in stable object order.

    This function only transforms human annotations. It never imports or runs
    the frozen RecallVision pipeline.
    """

    video = annotation["video"]
    video_id = video["video_id"]
    records: list[dict[str, Any]] = []
    for obj in sorted(annotation["objects"], key=lambda item: item["object_instance_id"]):
        object_id = obj["object_instance_id"]
        object_class = obj["object_class"]
        states = sorted(obj["states"], key=lambda item: (item["end_frame"], item["state_id"]))
        events = sorted(obj["movement_events"], key=lambda item: (item["start_frame"], item["event_id"]))
        records.extend(
            [
                _where_record(video_id, obj, states),
                _when_record(video_id, obj, events),
                _before_record(video_id, obj, events, states),
                _who_record(video_id, obj, events),
            ]
        )
    return records


def _where_record(video_id: str, obj: Mapping[str, Any], states: list[Mapping[str, Any]]) -> dict[str, Any]:
    object_id = obj["object_instance_id"]
    if obj["observation_status"] == "UNOBSERVED" or not states:
        return _record(
            video_id,
            obj,
            "WHERE_OBJECT",
            answerability="UNANSWERABLE",
            expected_status="ABSTAIN",
            ground_truth={"reason": "UNKNOWN_OBJECT"},
            hard_negative_tags=["OBJECT_DISAPPEARS"] if obj["observation_status"] == "DISAPPEARED" else [],
        ) | {"object_instance_id": None, "target_reference_status": "UNKNOWN_OBJECT"}
    state = states[-1]
    return _record(
        video_id,
        obj,
        "WHERE_OBJECT",
        answerability="ANSWERABLE",
        expected_status="ANSWER",
        ground_truth={
            "state_id": state["state_id"],
            "zone": state["zone"],
            "evidence_frames": [state["evidence_frames"][-1]],
        },
        evidence_refs=[{"frame": state["evidence_frames"][-1], "state_id": state["state_id"]}],
    )


def _when_record(video_id: str, obj: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(events) == 1:
        event = events[0]
        if event["movement_support_status"] == "SUPPORTED":
            return _event_answer_record(video_id, obj, "WHEN_OBJECT_MOVED", event)
        return _record(
            video_id,
            obj,
            "WHEN_OBJECT_MOVED",
            answerability="UNANSWERABLE",
            expected_status="ABSTAIN",
            ground_truth={"reason": "NO_SUPPORTED_EVENT"},
        )
    if len(events) > 1:
        return _record(
            video_id,
            obj,
            "WHEN_OBJECT_MOVED",
            answerability="AMBIGUOUS",
            expected_status="ABSTAIN",
            ground_truth={"reason": "MULTIPLE_EVENT_REFERENCE"},
            hard_negative_tags=["MULTIPLE_MOVE_EVENTS"],
        )
    return _record(
        video_id,
        obj,
        "WHEN_OBJECT_MOVED",
        answerability="UNANSWERABLE",
        expected_status="ABSTAIN",
        ground_truth={"reason": "NO_SUPPORTED_EVENT"},
    )


def _before_record(
    video_id: str,
    obj: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    states: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(events) != 1:
        reason = "MULTIPLE_EVENT_REFERENCE" if len(events) > 1 else "NO_SUPPORTED_EVENT"
        return _record(
            video_id,
            obj,
            "WHERE_OBJECT_BEFORE",
            answerability="AMBIGUOUS" if len(events) > 1 else "UNANSWERABLE",
            expected_status="ABSTAIN",
            ground_truth={"reason": reason},
            hard_negative_tags=["MULTIPLE_MOVE_EVENTS"] if len(events) > 1 else [],
        )
    event = events[0]
    source_state = next(
        (state for state in states if state["state_id"] == event["from_state_id"]),
        None,
    )
    if source_state is None or event["movement_support_status"] != "SUPPORTED":
        return _record(
            video_id,
            obj,
            "WHERE_OBJECT_BEFORE",
            answerability="UNANSWERABLE",
            expected_status="ABSTAIN",
            ground_truth={"reason": "NO_SUPPORTED_EVENT"},
        )
    return _event_answer_record(
        video_id,
        obj,
        "WHERE_OBJECT_BEFORE",
        event,
        ground_truth={
            "event_id": event["event_id"],
            "state_id": source_state["state_id"],
            "zone": source_state["zone"],
            "evidence_frames": [source_state["evidence_frames"][-1]],
        },
        evidence_refs=[{"frame": source_state["evidence_frames"][-1], "state_id": source_state["state_id"]}],
    )


def _who_record(video_id: str, obj: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(events) != 1:
        reason = "MULTIPLE_EVENT_REFERENCE" if len(events) > 1 else "NO_SUPPORTED_EVENT"
        return _record(
            video_id,
            obj,
            "WHO_MOVED_OBJECT",
            answerability="AMBIGUOUS" if len(events) > 1 else "UNANSWERABLE",
            expected_status="ABSTAIN",
            ground_truth={"reason": reason},
            hard_negative_tags=["MULTIPLE_MOVE_EVENTS"] if len(events) > 1 else [],
        )
    event = events[0]
    status = event["agent_attribution_status"]
    if event["movement_support_status"] != "SUPPORTED":
        answerability = "UNANSWERABLE"
        reason = "NO_SUPPORTED_EVENT"
    elif status == "SUPPORTED" and event["agent_instance_ids"]:
        return _event_answer_record(
            video_id,
            obj,
            "WHO_MOVED_OBJECT",
            event,
            ground_truth={
                "event_id": event["event_id"],
                "agent_instance_ids": list(event["agent_instance_ids"]),
                "agent_attribution_status": status,
            },
        )
    elif status == "AMBIGUOUS":
        answerability = "AMBIGUOUS"
        reason = "AMBIGUOUS_AGENT"
    else:
        answerability = "UNANSWERABLE"
        reason = "NO_VISIBLE_AGENT" if status == "NO_VISIBLE_AGENT" else "INSUFFICIENT"
    tags = (
        ["AMBIGUOUS_AGENT"]
        if reason == "AMBIGUOUS_AGENT"
        else ["NO_VISIBLE_AGENT"]
        if reason == "NO_VISIBLE_AGENT"
        else []
    )
    return _record(
        video_id,
        obj,
        "WHO_MOVED_OBJECT",
        answerability=answerability,
        expected_status="ABSTAIN",
        ground_truth={"reason": reason},
        hard_negative_tags=tags,
    )


def _event_answer_record(
    video_id: str,
    obj: Mapping[str, Any],
    query_type: str,
    event: Mapping[str, Any],
    *,
    ground_truth: Mapping[str, Any] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_ground_truth = {
        "event_id": event["event_id"],
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "evidence_frames": list(event["evidence_frames"]),
    }
    return _record(
        video_id,
        obj,
        query_type,
        answerability="ANSWERABLE",
        expected_status="ANSWER",
        reference_event_id=event["event_id"],
        ground_truth=dict(ground_truth or default_ground_truth),
        evidence_refs=evidence_refs
        if evidence_refs is not None
        else [{"frame": frame} for frame in event["evidence_frames"]],
    )


def _record(
    video_id: str,
    obj: Mapping[str, Any],
    query_type: str,
    *,
    answerability: str,
    expected_status: str,
    ground_truth: Mapping[str, Any],
    reference_event_id: str | None = None,
    hard_negative_tags: list[str] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    object_class = obj["object_class"]
    object_id = obj["object_instance_id"]
    return {
        "query_id": f"{video_id}:query:{object_id}:{query_type}",
        "video_id": video_id,
        "query_type": query_type,
        "object_instance_id": object_id,
        "target_reference_status": "KNOWN_OBJECT",
        "natural_language_question": _template(query_type, object_class),
        "reference_event_id": reference_event_id,
        "answerability": answerability,
        "expected_status": expected_status,
        "ground_truth": dict(ground_truth),
        "evidence_refs": list(evidence_refs or []),
        "hard_negative_tags": list(hard_negative_tags or []),
        "annotation_notes": "Generated by the deterministic EVAL-D0 template policy.",
    }


def _template(query_type: str, object_class: str) -> str:
    templates = {
        "WHERE_OBJECT": "Where is the {object_class}?",
        "WHEN_OBJECT_MOVED": "When did the {object_class} move?",
        "WHERE_OBJECT_BEFORE": "Where was the {object_class} before?",
        "WHO_MOVED_OBJECT": "Who moved the {object_class}?",
    }
    if query_type not in QUERY_TYPES:
        raise ValueError(f"unsupported query type: {query_type}")
    return templates[query_type].format(object_class=object_class)
