"""Strict independent-human review contracts and raw agreement comparison.

This module is deliberately separate from the frozen ``evaluation_annotation_v1``
contract.  It validates a reviewer-local, prediction-blind record and compares
it with primary annotations only after the blind pass has been returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from .validator import validate_manifest


SECOND_REVIEW_SCHEMA_VERSION = "eval_d0_second_review_v1"
REVIEW_AGENT_STATUSES = frozenset(
    {
        "SUPPORTED_SINGLE_AGENT",
        "SUPPORTED_MULTIPLE_AGENTS",
        "AMBIGUOUS_AGENT",
        "NO_VISIBLE_AGENT",
        "INSUFFICIENT_VISUAL_EVIDENCE",
    }
)
_REVIEW_PERSON_ID = re.compile(r"^review:person:[0-9]{4}$")
_REVIEW_EVENT_ID = re.compile(r"^review:event:[0-9]{4}$")
_REVIEW_INTERACTION_ID = re.compile(r"^review:interaction:[0-9]{4}$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class SecondReviewValidationError(ValueError):
    """Raised when a returned independent-review file violates its contract."""


def validate_second_review(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_catalog: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Validate one completed, prediction-blind human review without repair."""

    validate_manifest(manifest)
    _mapping(value, "second review")
    _fields(
        value,
        {
            "review_schema_version",
            "reviewer_id_or_alias",
            "review_date",
            "video_id",
            "video_metadata",
            "blindness_attestation",
            "target_reviews",
            "review_notes",
        },
        "second review",
    )
    if value["review_schema_version"] != SECOND_REVIEW_SCHEMA_VERSION:
        _error("review_schema_version is not supported")
    _nonempty_string(value["reviewer_id_or_alias"], "reviewer_id_or_alias")
    if not isinstance(value["review_date"], str) or _DATE.fullmatch(value["review_date"]) is None:
        _error("review_date must use YYYY-MM-DD")
    video_id = _nonempty_string(value["video_id"], "video_id")
    if video_id not in target_catalog:
        _error(f"video_id is not an R1 dry-run target: {video_id}")
    record = _manifest_record(manifest, video_id)
    if record["split"] != "EVAL":
        _error(f"review video must be EVAL: {video_id}")
    _validate_metadata(value["video_metadata"], record)
    _validate_blindness_attestation(value["blindness_attestation"])

    expected_targets = _target_map(target_catalog[video_id], video_id)
    target_reviews = value["target_reviews"]
    if not isinstance(target_reviews, list):
        _error("target_reviews must be a list")
    seen_targets: set[str] = set()
    seen_event_ids: set[str] = set()
    seen_interaction_ids: set[str] = set()
    for index, target_review in enumerate(target_reviews):
        path = f"target_reviews[{index}]"
        target_review = _mapping(target_review, path)
        _fields(
            target_review,
            {
                "target_object_id",
                "object_description",
                "last_observed_state",
                "last_observed_frame_or_time",
                "movement_events",
                "stationary_interactions",
                "notes",
            },
            path,
        )
        target_id = _nonempty_string(target_review["target_object_id"], f"{path}.target_object_id")
        if target_id in seen_targets:
            _error(f"duplicate target_object_id: {target_id}")
        seen_targets.add(target_id)
        if target_id not in expected_targets:
            _error(f"target_object_id is not in the predefined target list: {target_id}")
        if target_review["object_description"] != expected_targets[target_id]:
            _error(f"object_description does not match predefined target: {target_id}")
        _validate_last_state(
            target_review["last_observed_state"],
            target_review["last_observed_frame_or_time"],
            record,
            target_id,
        )
        _validate_events(
            target_review["movement_events"],
            record,
            target_id,
            seen_event_ids,
        )
        _validate_interactions(
            target_review["stationary_interactions"],
            record,
            target_id,
            seen_interaction_ids,
        )
        _nonempty_string(target_review["notes"], f"{path}.notes")
    if seen_targets != set(expected_targets):
        missing = sorted(set(expected_targets) - seen_targets)
        _error(f"target_reviews is missing target_object_id values: {missing}")
    _nonempty_string(value["review_notes"], "review_notes")


def match_review_events(
    primary_annotation: Mapping[str, Any],
    secondary_review: Mapping[str, Any],
    *,
    fps: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match primary and reviewer movement events by target and frame IoU.

    Matching is maximum-cardinality bipartite matching.  For each primary
    event, candidate reviewer events are considered by descending temporal IoU
    and then reviewer event ID.  The fixed threshold is never tuned from the
    observed agreement.
    """

    _validate_comparison_arguments(fps, iou_threshold)
    primary_events = _primary_events(primary_annotation)
    review_events = _review_events(secondary_review)
    edges: dict[int, list[tuple[int, float]]] = {}
    for primary_index, primary_event in enumerate(primary_events):
        candidates: list[tuple[int, float]] = []
        for review_index, review_event in enumerate(review_events):
            if primary_event["object_instance_id"] != review_event["target_object_id"]:
                continue
            score = _frame_interval_iou(
                (primary_event["start_frame"], primary_event["end_frame"]),
                (review_event["evidence_start_frame"], review_event["evidence_end_frame"]),
            )
            if score >= iou_threshold:
                candidates.append((review_index, score))
        edges[primary_index] = sorted(
            candidates,
            key=lambda item: (-item[1], review_events[item[0]]["review_event_id"]),
        )

    matched_review_to_primary: dict[int, int] = {}

    def augment(primary_index: int, visited: set[int]) -> bool:
        for review_index, _ in edges[primary_index]:
            if review_index in visited:
                continue
            visited.add(review_index)
            previous = matched_review_to_primary.get(review_index)
            if previous is None or augment(previous, visited):
                matched_review_to_primary[review_index] = primary_index
                return True
        return False

    for primary_index in range(len(primary_events)):
        augment(primary_index, set())

    matches: list[dict[str, Any]] = []
    for review_index, primary_index in sorted(
        matched_review_to_primary.items(),
        key=lambda item: primary_events[item[1]]["event_id"],
    ):
        primary_event = primary_events[primary_index]
        review_event = review_events[review_index]
        temporal_iou = _frame_interval_iou(
            (primary_event["start_frame"], primary_event["end_frame"]),
            (review_event["evidence_start_frame"], review_event["evidence_end_frame"]),
        )
        matches.append(
            {
                "target_object_id": primary_event["object_instance_id"],
                "primary_event_id": primary_event["event_id"],
                "review_event_id": review_event["review_event_id"],
                "primary_interval": [primary_event["start_frame"], primary_event["end_frame"]],
                "secondary_interval": [
                    review_event["evidence_start_frame"],
                    review_event["evidence_end_frame"],
                ],
                "temporal_iou": temporal_iou,
                "start_boundary_abs_error_seconds": abs(
                    primary_event["start_frame"] - review_event["evidence_start_frame"]
                )
                / fps,
                "end_boundary_abs_error_seconds": abs(
                    primary_event["end_frame"] - review_event["evidence_end_frame"]
                )
                / fps,
            }
        )
    matched_primary_ids = {item["primary_event_id"] for item in matches}
    matched_review_ids = {item["review_event_id"] for item in matches}
    return {
        "iou_threshold": iou_threshold,
        "matched_count": len(matches),
        "matches": matches,
        "unmatched_primary_event_ids": [
            event["event_id"] for event in primary_events if event["event_id"] not in matched_primary_ids
        ],
        "unmatched_review_event_ids": [
            event["review_event_id"]
            for event in review_events
            if event["review_event_id"] not in matched_review_ids
        ],
    }


def compare_second_review(
    primary_annotation: Mapping[str, Any],
    secondary_review: Mapping[str, Any],
    *,
    fps: float,
    person_correspondence: Mapping[str, str] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return deterministic raw agreement and case-level discrepancies."""

    _validate_comparison_arguments(fps, iou_threshold)
    correspondence = dict(person_correspondence or {})
    primary_objects = {
        obj["object_instance_id"]: obj for obj in primary_annotation.get("objects", [])
    }
    secondary_targets = {
        target["target_object_id"]: target for target in secondary_review.get("target_reviews", [])
    }
    disagreements: list[dict[str, Any]] = []
    last_state_correct = 0
    for object_id in sorted(primary_objects):
        primary_object = primary_objects[object_id]
        secondary_target = secondary_targets.get(object_id)
        primary_state = max(primary_object["states"], key=lambda state: state["end_frame"])
        if secondary_target is None:
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    object_id,
                    "LAST_OBSERVED_STATE",
                    {"zone": primary_state["zone"], "frame": primary_state["end_frame"]},
                    None,
                    {"primary_state_id": primary_state["state_id"]},
                    "STATE_LABEL",
                )
            )
            continue
        secondary_state = secondary_target["last_observed_state"]
        if _norm(primary_state["zone"]) == _norm(secondary_state["zone_or_location"]):
            last_state_correct += 1
        else:
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    object_id,
                    "LAST_OBSERVED_STATE",
                    {"zone": primary_state["zone"], "frame": primary_state["end_frame"]},
                    {
                        "zone_or_location": secondary_state["zone_or_location"],
                        "frame_or_time": secondary_target["last_observed_frame_or_time"],
                    },
                    {
                        "primary_state_id": primary_state["state_id"],
                        "secondary_note": secondary_state["ambiguity_note"],
                    },
                    "STATE_LABEL",
                )
            )

    matching = match_review_events(
        primary_annotation,
        secondary_review,
        fps=fps,
        iou_threshold=iou_threshold,
    )
    primary_events = {event["event_id"]: event for event in _primary_events(primary_annotation)}
    review_events = {
        event["review_event_id"]: event for event in _review_events(secondary_review)
    }
    primary_by_object = _group_by(primary_events.values(), "object_instance_id")
    review_by_object = _group_by(review_events.values(), "target_object_id")
    target_ids = sorted(set(primary_by_object) | set(review_by_object))
    event_presence_correct = sum(
        bool(primary_by_object.get(target_id)) == bool(review_by_object.get(target_id))
        for target_id in target_ids
    )

    pre_event_correct = 0
    agent_status_correct = 0
    supported_identity_correct = 0
    supported_identity_total = 0
    same_zone_correct = 0
    interval_summary: list[dict[str, Any]] = []
    for match in matching["matches"]:
        primary_event = primary_events[match["primary_event_id"]]
        review_event = review_events[match["review_event_id"]]
        primary_from = _state_by_id(primary_annotation, primary_event["from_state_id"])
        if _norm(primary_from["zone"]) == _norm(review_event["from_state"]["zone_or_location"]):
            pre_event_correct += 1
        else:
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    primary_event["object_instance_id"],
                    "PRE_EVENT_STATE",
                    {"zone": primary_from["zone"], "state_id": primary_from["state_id"]},
                    review_event["from_state"],
                    {"primary_event_id": primary_event["event_id"], "review_event_id": review_event["review_event_id"]},
                    "STATE_LABEL",
                )
            )
        expected_status = _review_status_for_primary(primary_event)
        if expected_status == review_event["agent_status"]:
            agent_status_correct += 1
        else:
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    primary_event["object_instance_id"],
                    "AGENT_STATUS",
                    {
                        "status": expected_status,
                        "person_ids": primary_event["agent_instance_ids"],
                    },
                    {
                        "status": review_event["agent_status"],
                        "person_ids": review_event["review_person_ids"],
                    },
                    {"primary_event_id": primary_event["event_id"], "review_event_id": review_event["review_event_id"]},
                    "AGENT_CAUSALITY",
                )
            )
        primary_same_zone = _state_by_id(primary_annotation, primary_event["from_state_id"])["zone"] == _state_by_id(
            primary_annotation, primary_event["to_state_id"]
        )["zone"]
        review_same_zone = _norm(review_event["from_state"]["zone_or_location"]) == _norm(
            review_event["to_state"]["zone_or_location"]
        )
        if primary_same_zone == review_same_zone:
            same_zone_correct += 1
        else:
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    primary_event["object_instance_id"],
                    "SAME_ZONE_MOVEMENT",
                    primary_same_zone,
                    review_same_zone,
                    {"primary_event_id": primary_event["event_id"], "review_event_id": review_event["review_event_id"]},
                    "STATE_LABEL",
                )
            )
        if expected_status in {"SUPPORTED_SINGLE_AGENT", "SUPPORTED_MULTIPLE_AGENTS"}:
            mapped_review_ids = {correspondence.get(person_id) for person_id in review_event["review_person_ids"]}
            if all(person_id in correspondence for person_id in review_event["review_person_ids"]):
                supported_identity_total += 1
                if mapped_review_ids == set(primary_event["agent_instance_ids"]):
                    supported_identity_correct += 1
                else:
                    disagreements.append(
                        _disagreement(
                            secondary_review["video_id"],
                            primary_event["object_instance_id"],
                            "SUPPORTED_AGENT_IDENTITY",
                            primary_event["agent_instance_ids"],
                            review_event["review_person_ids"],
                            {"primary_event_id": primary_event["event_id"], "review_event_id": review_event["review_event_id"]},
                            "PERSON_CORRESPONDENCE",
                        )
                    )
        if (
            match["primary_interval"] != match["secondary_interval"]
        ):
            disagreements.append(
                _disagreement(
                    secondary_review["video_id"],
                    primary_event["object_instance_id"],
                    "MOVEMENT_INTERVAL",
                    match["primary_interval"],
                    match["secondary_interval"],
                    {
                        "primary_event_id": primary_event["event_id"],
                        "review_event_id": review_event["review_event_id"],
                        "primary_evidence_frames": primary_event["evidence_frames"],
                    },
                    "BOUNDARY_TOLERANCE",
                )
            )
        interval_summary.append(match)

    for event_id in matching["unmatched_primary_event_ids"]:
        event = primary_events[event_id]
        disagreements.append(
            _disagreement(
                secondary_review["video_id"],
                event["object_instance_id"],
                "MOVEMENT_EVENT",
                {"event_id": event_id, "interval": [event["start_frame"], event["end_frame"]]},
                None,
                {"primary_evidence_frames": event["evidence_frames"]},
                "EVENT_EXISTENCE",
            )
        )
    for event_id in matching["unmatched_review_event_ids"]:
        event = review_events[event_id]
        disagreements.append(
            _disagreement(
                secondary_review["video_id"],
                event["target_object_id"],
                "MOVEMENT_EVENT",
                None,
                {"event_id": event_id, "interval": [event["evidence_start_frame"], event["evidence_end_frame"]]},
                {"secondary_interval": [event["evidence_start_frame"], event["evidence_end_frame"]]},
                "EVENT_EXISTENCE",
            )
        )

    stationary = _stationary_agreement(primary_annotation, secondary_review, fps=fps)
    disagreements.extend(stationary["disagreements"])
    disagreements.sort(key=_disagreement_sort_key)
    return {
        "video_id": secondary_review["video_id"],
        "primary_movement_event_count": len(primary_events),
        "secondary_movement_event_count": len(review_events),
        "target_last_state_agreement": _agreement(last_state_correct, len(primary_objects)),
        "movement_event_presence_agreement": _agreement(event_presence_correct, len(target_ids)),
        "matched_event_count": matching["matched_count"],
        "temporal_interval_agreement_summary": interval_summary,
        "pre_event_state_agreement": _agreement(pre_event_correct, matching["matched_count"]),
        "agent_status_agreement": _agreement(agent_status_correct, matching["matched_count"]),
        "supported_agent_identity_agreement": _identity_agreement(
            supported_identity_correct, supported_identity_total
        ),
        "stationary_interaction_agreement": stationary["agreement"],
        "same_zone_movement_agreement": _agreement(same_zone_correct, matching["matched_count"]),
        "unmatched_primary_events": matching["unmatched_primary_event_ids"],
        "unmatched_secondary_events": matching["unmatched_review_event_ids"],
        "disagreements": disagreements,
    }


def _validate_metadata(value: Any, record: Mapping[str, Any]) -> None:
    _mapping(value, "video_metadata")
    _fields(
        value,
        {"raw_video_path", "source_sha256", "duration_seconds", "fps", "frame_count", "width", "height"},
        "video_metadata",
    )
    if value["raw_video_path"] != record["relative_video_path"]:
        _error("video_metadata.raw_video_path does not match manifest")
    if value["source_sha256"] != record["sha256"]:
        _error("video_metadata.source_sha256 does not match manifest")
    for field in ("duration_seconds", "fps"):
        _positive_number(value[field], f"video_metadata.{field}")
        if not math.isclose(value[field], record[field], rel_tol=0.0, abs_tol=1e-9):
            _error(f"video_metadata.{field} does not match manifest")
    for field in ("frame_count", "width", "height"):
        _positive_int(value[field], f"video_metadata.{field}")
        if value[field] != record[field]:
            _error(f"video_metadata.{field} does not match manifest")


def _validate_blindness_attestation(value: Any) -> None:
    _mapping(value, "blindness_attestation")
    _fields(
        value,
        {
            "target_list_predefined",
            "primary_event_labels_visible_to_reviewer",
            "primary_agent_labels_visible_to_reviewer",
            "system_predictions_visible_to_reviewer",
        },
        "blindness_attestation",
    )
    if value["target_list_predefined"] is not True:
        _error("target_list_predefined must be true")
    for field in (
        "primary_event_labels_visible_to_reviewer",
        "primary_agent_labels_visible_to_reviewer",
        "system_predictions_visible_to_reviewer",
    ):
        if value[field] is not False:
            _error(f"{field} must be false for a blind pass")


def _validate_last_state(
    state_value: Any,
    frame_value: Any,
    record: Mapping[str, Any],
    target_id: str,
) -> None:
    state = _mapping(state_value, f"{target_id}.last_observed_state")
    _fields(
        state,
        {"zone_or_location", "state_description", "ambiguity_note"},
        f"{target_id}.last_observed_state",
    )
    _nonempty_string(state["zone_or_location"], f"{target_id}.last_observed_state.zone_or_location")
    _nonempty_string(state["state_description"], f"{target_id}.last_observed_state.state_description")
    _nonempty_string(state["ambiguity_note"], f"{target_id}.last_observed_state.ambiguity_note")
    _validate_frame_or_time(frame_value, record, f"{target_id}.last_observed_frame_or_time", allow_null=True)


def _validate_events(
    value: Any,
    record: Mapping[str, Any],
    target_id: str,
    seen_ids: set[str],
) -> None:
    if not isinstance(value, list):
        _error(f"{target_id}.movement_events must be a list")
    for index, event_value in enumerate(value):
        path = f"{target_id}.movement_events[{index}]"
        event = _mapping(event_value, path)
        _fields(
            event,
            {
                "review_event_id",
                "evidence_start_frame",
                "evidence_end_frame",
                "evidence_start_time_seconds",
                "evidence_end_time_seconds",
                "from_state",
                "to_state",
                "agent_status",
                "review_person_ids",
                "confidence_or_ambiguity_note",
            },
            path,
        )
        event_id = _nonempty_string(event["review_event_id"], f"{path}.review_event_id")
        if _REVIEW_EVENT_ID.fullmatch(event_id) is None:
            _error(f"{path}.review_event_id must use review:event:NNNN")
        if event_id in seen_ids:
            _error(f"duplicate review_event_id: {event_id}")
        seen_ids.add(event_id)
        _validate_interval(
            event["evidence_start_frame"],
            event["evidence_end_frame"],
            record,
            path,
        )
        _validate_frame_time_pair(
            event["evidence_start_frame"],
            event["evidence_start_time_seconds"],
            record,
            f"{path}.evidence_start",
        )
        _validate_frame_time_pair(
            event["evidence_end_frame"],
            event["evidence_end_time_seconds"],
            record,
            f"{path}.evidence_end",
        )
        _validate_state_ref(event["from_state"], f"{path}.from_state")
        _validate_state_ref(event["to_state"], f"{path}.to_state")
        status = event["agent_status"]
        if status not in REVIEW_AGENT_STATUSES:
            _error(f"{path}.agent_status is not allowed")
        person_ids = _validate_review_person_ids(event["review_person_ids"], f"{path}.review_person_ids")
        if status == "SUPPORTED_SINGLE_AGENT" and len(person_ids) != 1:
            _error(f"{path}.SUPPORTED_SINGLE_AGENT requires exactly one review person ID")
        if status == "SUPPORTED_MULTIPLE_AGENTS" and len(person_ids) < 2:
            _error(f"{path}.SUPPORTED_MULTIPLE_AGENTS requires at least two review person IDs")
        if status in {"NO_VISIBLE_AGENT", "INSUFFICIENT_VISUAL_EVIDENCE"} and person_ids:
            _error(f"{path}.{status} cannot contain review person IDs")
        _nonempty_string(event["confidence_or_ambiguity_note"], f"{path}.confidence_or_ambiguity_note")


def _validate_interactions(
    value: Any,
    record: Mapping[str, Any],
    target_id: str,
    seen_ids: set[str],
) -> None:
    if not isinstance(value, list):
        _error(f"{target_id}.stationary_interactions must be a list")
    for index, interaction_value in enumerate(value):
        path = f"{target_id}.stationary_interactions[{index}]"
        interaction = _mapping(interaction_value, path)
        _fields(
            interaction,
            {
                "review_interaction_id",
                "evidence_start_frame",
                "evidence_end_frame",
                "evidence_start_time_seconds",
                "evidence_end_time_seconds",
                "review_person_ids",
                "interaction_description",
                "confidence_or_ambiguity_note",
            },
            path,
        )
        interaction_id = _nonempty_string(
            interaction["review_interaction_id"], f"{path}.review_interaction_id"
        )
        if _REVIEW_INTERACTION_ID.fullmatch(interaction_id) is None:
            _error(f"{path}.review_interaction_id must use review:interaction:NNNN")
        if interaction_id in seen_ids:
            _error(f"duplicate review_interaction_id: {interaction_id}")
        seen_ids.add(interaction_id)
        _validate_interval(
            interaction["evidence_start_frame"],
            interaction["evidence_end_frame"],
            record,
            path,
        )
        _validate_frame_time_pair(
            interaction["evidence_start_frame"],
            interaction["evidence_start_time_seconds"],
            record,
            f"{path}.evidence_start",
        )
        _validate_frame_time_pair(
            interaction["evidence_end_frame"],
            interaction["evidence_end_time_seconds"],
            record,
            f"{path}.evidence_end",
        )
        person_ids = _validate_review_person_ids(interaction["review_person_ids"], f"{path}.review_person_ids")
        if not person_ids:
            _error(f"{path}.stationary interaction requires a review person ID")
        _nonempty_string(interaction["interaction_description"], f"{path}.interaction_description")
        _nonempty_string(
            interaction["confidence_or_ambiguity_note"],
            f"{path}.confidence_or_ambiguity_note",
        )


def _validate_state_ref(value: Any, label: str) -> None:
    state = _mapping(value, label)
    _fields(state, {"zone_or_location", "state_description"}, label)
    _nonempty_string(state["zone_or_location"], f"{label}.zone_or_location")
    _nonempty_string(state["state_description"], f"{label}.state_description")


def _validate_review_person_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _error(f"{label} must be a list of review person IDs")
    result = list(value)
    if any(_REVIEW_PERSON_ID.fullmatch(item) is None for item in result):
        _error(f"{label} contains an invalid reviewer-local person ID")
    if len(set(result)) != len(result):
        _error(f"{label} contains duplicate person IDs")
    return result


def _validate_interval(start: Any, end: Any, record: Mapping[str, Any], label: str) -> None:
    _positive_int(start, f"{label}.start_frame")
    _positive_int(end, f"{label}.end_frame")
    if start > end:
        _error(f"{label} interval is reversed")
    if end >= record["frame_count"]:
        _error(f"{label} interval exceeds video bounds")


def _validate_frame_or_time(value: Any, record: Mapping[str, Any], label: str, *, allow_null: bool) -> None:
    data = _mapping(value, label)
    _fields(data, {"frame", "time_seconds"}, label)
    frame = data["frame"]
    timestamp = data["time_seconds"]
    if frame is None and timestamp is None and allow_null:
        return
    if frame is None or timestamp is None:
        _error(f"{label} frame and time must both be present or both null")
    _validate_frame_time_pair(frame, timestamp, record, label)


def _validate_frame_time_pair(frame: Any, timestamp: Any, record: Mapping[str, Any], label: str) -> None:
    _positive_int(frame, f"{label}.frame")
    if frame >= record["frame_count"]:
        _error(f"{label}.frame is outside video bounds")
    _positive_number(timestamp, f"{label}.time_seconds", allow_zero=True)
    if timestamp > record["duration_seconds"]:
        _error(f"{label}.time_seconds is outside video bounds")
    if abs(timestamp - frame / record["fps"]) > (1.0 / record["fps"]) + 1e-6:
        _error(f"{label}.time_seconds is inconsistent with frame/fps")


def _manifest_record(manifest: Mapping[str, Any], video_id: str) -> Mapping[str, Any]:
    records = [record for record in manifest["videos"] if record["video_id"] == video_id]
    if len(records) != 1:
        _error(f"video_id is absent or duplicated in manifest: {video_id}")
    return records[0]


def _target_map(value: Sequence[Mapping[str, str]], video_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, target in enumerate(value):
        target = _mapping(target, f"target_catalog[{video_id}][{index}]")
        _fields(target, {"target_object_id", "object_description"}, "target catalog entry")
        target_id = _nonempty_string(target["target_object_id"], "target catalog target_object_id")
        description = _nonempty_string(target["object_description"], "target catalog object_description")
        if target_id in result:
            _error(f"duplicate target catalog target_object_id: {target_id}")
        result[target_id] = description
    return result


def _primary_events(annotation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = [event for obj in annotation.get("objects", []) for event in obj.get("movement_events", [])]
    return sorted(events, key=lambda event: (event["object_instance_id"], event["event_id"]))


def _review_events(review: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for target in review.get("target_reviews", []):
        for event in target.get("movement_events", []):
            event_with_target = dict(event)
            event_with_target["target_object_id"] = target["target_object_id"]
            events.append(event_with_target)
    return sorted(events, key=lambda event: event["review_event_id"])


def _frame_interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union if union else 1.0


def _state_by_id(annotation: Mapping[str, Any], state_id: str) -> Mapping[str, Any]:
    for obj in annotation.get("objects", []):
        for state in obj.get("states", []):
            if state["state_id"] == state_id:
                return state
    raise SecondReviewValidationError(f"primary state is unresolved: {state_id}")


def _group_by(values: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for value in values:
        result.setdefault(value[key], []).append(value)
    return result


def _review_status_for_primary(event: Mapping[str, Any]) -> str:
    status = event["agent_attribution_status"]
    if status == "SUPPORTED":
        return "SUPPORTED_SINGLE_AGENT" if len(event["agent_instance_ids"]) == 1 else "SUPPORTED_MULTIPLE_AGENTS"
    if status == "AMBIGUOUS":
        return "AMBIGUOUS_AGENT"
    if status == "NO_VISIBLE_AGENT":
        return "NO_VISIBLE_AGENT"
    return "INSUFFICIENT_VISUAL_EVIDENCE"


def _stationary_agreement(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    fps: float,
) -> dict[str, Any]:
    primary_items = [
        {
            "target_object_id": item["object_instance_id"],
            "start_frame": item["start_frame"],
            "end_frame": item["end_frame"],
            "id": item["interaction_id"],
        }
        for item in primary.get("interaction_negatives", [])
    ]
    secondary_items = [
        {
            "target_object_id": target["target_object_id"],
            "start_frame": item["evidence_start_frame"],
            "end_frame": item["evidence_end_frame"],
            "id": item["review_interaction_id"],
        }
        for target in secondary.get("target_reviews", [])
        for item in target.get("stationary_interactions", [])
    ]
    matched = _match_interval_records(primary_items, secondary_items)
    disagreements: list[dict[str, Any]] = []
    matched_primary = {primary_items[left]["id"] for left, _ in matched}
    matched_secondary = {secondary_items[right]["id"] for _, right in matched}
    for item in primary_items:
        if item["id"] not in matched_primary:
            disagreements.append(
                _disagreement(
                    secondary["video_id"],
                    item["target_object_id"],
                    "STATIONARY_INTERACTION",
                    item,
                    None,
                    {"primary_interaction_id": item["id"]},
                    "STATIONARY_INTERACTION",
                )
            )
    for item in secondary_items:
        if item["id"] not in matched_secondary:
            disagreements.append(
                _disagreement(
                    secondary["video_id"],
                    item["target_object_id"],
                    "STATIONARY_INTERACTION",
                    None,
                    item,
                    {"secondary_interaction_id": item["id"]},
                    "STATIONARY_INTERACTION",
                )
            )
    total = max(len(primary_items), len(secondary_items))
    return {
        "agreement": _agreement(len(matched), total),
        "disagreements": disagreements,
    }


def _match_interval_records(
    primary: Sequence[Mapping[str, Any]], secondary: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    candidates = [
        (left, right, _frame_interval_iou(
            (first["start_frame"], first["end_frame"]),
            (second["start_frame"], second["end_frame"]),
        ))
        for left, first in enumerate(primary)
        for right, second in enumerate(secondary)
        if first["target_object_id"] == second["target_object_id"]
    ]
    candidates.sort(key=lambda item: (-item[2], primary[item[0]]["id"], secondary[item[1]]["id"]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    result: list[tuple[int, int]] = []
    for left, right, _ in candidates:
        if left not in used_left and right not in used_right:
            used_left.add(left)
            used_right.add(right)
            result.append((left, right))
    return result


def _agreement(correct: int, total: int) -> dict[str, Any]:
    return {"agree": correct, "total": total, "display": f"{correct} / {total}" if total else "NOT_APPLICABLE"}


def _identity_agreement(correct: int, total: int) -> dict[str, Any]:
    if not total:
        return {"agree": 0, "total": 0, "display": "NOT_COMPARABLE (no correspondence supplied)"}
    return _agreement(correct, total)


def _disagreement(
    video_id: str,
    target_object: str,
    variable: str,
    primary_value: Any,
    secondary_value: Any,
    raw_evidence_reference: Mapping[str, Any],
    disagreement_type: str,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "target_object": target_object,
        "variable": variable,
        "primary_value": primary_value,
        "secondary_value": secondary_value,
        "raw_evidence_reference": dict(raw_evidence_reference),
        "disagreement_type": disagreement_type,
    }


def _disagreement_sort_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        value["video_id"],
        value["target_object"],
        value["variable"],
        value["disagreement_type"],
    )


def _validate_comparison_arguments(fps: float, iou_threshold: float) -> None:
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not isinstance(iou_threshold, (int, float)) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


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


def _positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _error(f"{label} must be a non-negative integer")


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        _error(f"{label} must be a finite {'non-negative' if allow_zero else 'positive'} number")


def _error(message: str) -> None:
    raise SecondReviewValidationError(message)
