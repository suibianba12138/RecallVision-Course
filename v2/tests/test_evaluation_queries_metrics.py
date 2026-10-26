"""Tests-first query-ground-truth and metric primitives for EVAL-D0."""

from __future__ import annotations

from copy import deepcopy

import pytest

from recallvision.evaluation.metrics import (
    NOT_APPLICABLE,
    attribution_metrics,
    boundary_errors_seconds,
    categorical_accuracy,
    event_interval_metrics,
    fail_closed_metrics,
    frame_interval_iou,
)
from recallvision.evaluation.query_ground_truth import generate_query_ground_truth
from tests.test_evaluation_contracts import _annotation


def test_query_templates_are_deterministic_and_cover_all_four_families() -> None:
    annotation = _annotation()

    first = generate_query_ground_truth(annotation)
    second = generate_query_ground_truth(annotation)

    assert first == second
    assert [item["query_type"] for item in first] == [
        "WHERE_OBJECT",
        "WHEN_OBJECT_MOVED",
        "WHERE_OBJECT_BEFORE",
        "WHO_MOVED_OBJECT",
    ]
    assert [item["natural_language_question"] for item in first] == [
        "Where is the cup?",
        "When did the cup move?",
        "Where was the cup before?",
        "Who moved the cup?",
    ]
    assert all(item["answerability"] == "ANSWERABLE" for item in first)
    assert all(item["evidence_refs"] for item in first)


def test_multiple_events_generate_fail_closed_reference_records() -> None:
    annotation = _annotation()
    second_event = deepcopy(annotation["objects"][0]["movement_events"][0])  # type: ignore[index]
    second_event["event_id"] = "gt:event:move:0002"
    annotation["objects"][0]["movement_events"].append(second_event)  # type: ignore[index]

    records = generate_query_ground_truth(annotation)
    by_type = {record["query_type"]: record for record in records}

    assert by_type["WHEN_OBJECT_MOVED"]["answerability"] == "AMBIGUOUS"
    assert by_type["WHEN_OBJECT_MOVED"]["expected_status"] == "ABSTAIN"
    assert by_type["WHEN_OBJECT_MOVED"]["ground_truth"]["reason"] == "MULTIPLE_EVENT_REFERENCE"
    assert by_type["WHERE_OBJECT_BEFORE"]["answerability"] == "AMBIGUOUS"


def test_unknown_target_and_no_visible_agent_generate_abstention_records() -> None:
    unknown = _annotation()
    unknown["objects"][0]["observation_status"] = "UNOBSERVED"  # type: ignore[index]
    unknown["objects"][0]["states"] = []  # type: ignore[index]
    unknown["objects"][0]["movement_events"] = []  # type: ignore[index]
    unknown_records = generate_query_ground_truth(unknown)
    where = next(record for record in unknown_records if record["query_type"] == "WHERE_OBJECT")
    assert where["object_instance_id"] is None
    assert where["target_reference_status"] == "UNKNOWN_OBJECT"
    assert where["expected_status"] == "ABSTAIN"

    no_agent = _annotation()
    event = no_agent["objects"][0]["movement_events"][0]  # type: ignore[index]
    event["agent_attribution_status"] = "NO_VISIBLE_AGENT"
    event["agent_instance_ids"] = []
    who = next(
        record for record in generate_query_ground_truth(no_agent) if record["query_type"] == "WHO_MOVED_OBJECT"
    )
    assert who["ground_truth"]["reason"] == "NO_VISIBLE_AGENT"
    assert who["expected_status"] == "ABSTAIN"


def test_categorical_accuracy_reports_not_applicable_for_empty_input() -> None:
    assert categorical_accuracy([], []) == {
        "value": NOT_APPLICABLE,
        "correct": 0,
        "denominator": 0,
    }
    assert categorical_accuracy(["left", "right"], ["left", "wrong"]) == {
        "value": 0.5,
        "correct": 1,
        "denominator": 2,
    }


def test_fail_closed_metrics_separate_false_success_and_correct_abstention() -> None:
    result = fail_closed_metrics(
        expected_answerability=["ANSWERABLE", "UNANSWERABLE", "AMBIGUOUS"],
        predicted_status=["ANSWER", "ANSWER", "ABSTAIN"],
        expected_reasons=[None, "UNKNOWN_OBJECT", "MULTIPLE_EVENT_REFERENCE"],
    )

    assert result["answerability_accuracy"]["value"] == pytest.approx(2 / 3)
    assert result["false_success_rate"]["value"] == pytest.approx(1 / 2)
    assert result["correct_abstention_rate"]["value"] == pytest.approx(1 / 2)
    assert result["ambiguous_reference_rejection_correctness"]["value"] == pytest.approx(1.0)


def test_temporal_iou_and_boundary_errors_use_inclusive_frames_and_half_open_time() -> None:
    assert frame_interval_iou((0, 2), (1, 3), fps=10.0) == pytest.approx(0.5)
    assert boundary_errors_seconds((0, 2), (1, 3), fps=10.0) == {
        "start_error_seconds": pytest.approx(0.1),
        "end_error_seconds": pytest.approx(0.1),
    }


def test_event_metrics_match_one_to_one_and_report_empty_case() -> None:
    predicted = [
        {"event_id": "pred-1", "object_instance_id": "obj-1", "event_type": "MOVE_OBJECT", "start_frame": 10, "end_frame": 20},
        {"event_id": "pred-duplicate", "object_instance_id": "obj-1", "event_type": "MOVE_OBJECT", "start_frame": 10, "end_frame": 20},
    ]
    ground_truth = [
        {"event_id": "gt-1", "object_instance_id": "obj-1", "event_type": "MOVE_OBJECT", "start_frame": 10, "end_frame": 20},
    ]

    result = event_interval_metrics(predicted, ground_truth, fps=30.0)

    assert result["matched_count"] == 1
    assert result["precision"]["value"] == pytest.approx(0.5)
    assert result["recall"]["value"] == pytest.approx(1.0)
    assert event_interval_metrics([], [], fps=30.0)["precision"]["value"] == NOT_APPLICABLE


def test_attribution_metrics_keep_abstention_and_identity_unresolved_separate() -> None:
    result = attribution_metrics(
        predicted_agent_ids=[["person-1"], [], ["unknown-person"]],
        expected_agent_ids=[["person-1"], [], []],
        expected_statuses=["SUPPORTED", "NO_VISIBLE_AGENT", "AMBIGUOUS"],
        known_predicted_agent_ids={"person-1"},
    )

    assert result["attributable_event_accuracy"]["value"] == pytest.approx(1.0)
    assert result["agent_abstention_accuracy"]["value"] == pytest.approx(0.5)
    assert result["false_attribution_rate"]["value"] == pytest.approx(0.5)
    assert result["identity_unresolved_rate"]["value"] == pytest.approx(0.5)
