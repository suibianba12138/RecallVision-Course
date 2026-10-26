"""Small, deterministic metric primitives for structured EVAL-D0 records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

NOT_APPLICABLE = "NOT_APPLICABLE"


def categorical_accuracy(predicted: Sequence[Any], expected: Sequence[Any]) -> dict[str, Any]:
    _same_length(predicted, expected)
    correct = sum(left == right for left, right in zip(predicted, expected))
    return _ratio(correct, len(expected), correct=correct)


def fail_closed_metrics(
    *,
    expected_answerability: Sequence[str],
    predicted_status: Sequence[str],
    expected_reasons: Sequence[str | None],
) -> dict[str, dict[str, Any]]:
    _same_length(expected_answerability, predicted_status, expected_reasons)
    answerability_correct = sum(
        (answerability == "ANSWERABLE") == (status == "ANSWER")
        for answerability, status in zip(expected_answerability, predicted_status)
    )
    fail_closed_indices = [
        index for index, answerability in enumerate(expected_answerability) if answerability != "ANSWERABLE"
    ]
    false_success = sum(predicted_status[index] == "ANSWER" for index in fail_closed_indices)
    correct_abstention = sum(predicted_status[index] == "ABSTAIN" for index in fail_closed_indices)
    ambiguous_indices = [
        index for index, reason in enumerate(expected_reasons) if reason == "MULTIPLE_EVENT_REFERENCE"
    ]
    ambiguous_correct = sum(predicted_status[index] == "ABSTAIN" for index in ambiguous_indices)
    unknown_indices = [
        index for index, reason in enumerate(expected_reasons) if reason == "UNKNOWN_OBJECT"
    ]
    unknown_correct = sum(predicted_status[index] == "ABSTAIN" for index in unknown_indices)
    return {
        "answerability_accuracy": _ratio(answerability_correct, len(expected_answerability), correct=answerability_correct),
        "false_success_rate": _ratio(false_success, len(fail_closed_indices), correct=false_success),
        "correct_abstention_rate": _ratio(correct_abstention, len(fail_closed_indices), correct=correct_abstention),
        "ambiguous_reference_rejection_correctness": _ratio(ambiguous_correct, len(ambiguous_indices), correct=ambiguous_correct),
        "unknown_target_rejection_correctness": _ratio(unknown_correct, len(unknown_indices), correct=unknown_correct),
    }


def frame_interval_iou(
    predicted: tuple[int, int],
    expected: tuple[int, int],
    *,
    fps: float,
) -> float:
    """Compute IoU after converting inclusive frame endpoints to half-open time."""

    _validate_interval(predicted, "predicted")
    _validate_interval(expected, "expected")
    if fps <= 0:
        raise ValueError("fps must be positive")
    predicted_start, predicted_end = predicted[0] / fps, (predicted[1] + 1) / fps
    expected_start, expected_end = expected[0] / fps, (expected[1] + 1) / fps
    intersection = max(0.0, min(predicted_end, expected_end) - max(predicted_start, expected_start))
    union = max(predicted_end, expected_end) - min(predicted_start, expected_start)
    return intersection / union if union else 1.0


def boundary_errors_seconds(
    predicted: tuple[int, int],
    expected: tuple[int, int],
    *,
    fps: float,
) -> dict[str, float]:
    _validate_interval(predicted, "predicted")
    _validate_interval(expected, "expected")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return {
        "start_error_seconds": abs(predicted[0] - expected[0]) / fps,
        "end_error_seconds": abs(predicted[1] - expected[1]) / fps,
    }


def event_interval_metrics(
    predicted_events: Sequence[Mapping[str, Any]],
    ground_truth_events: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    predicted = sorted(predicted_events, key=lambda event: event["event_id"])
    truth = sorted(ground_truth_events, key=lambda event: event["event_id"])
    edges: dict[int, list[tuple[int, float]]] = {}
    for truth_index, truth_event in enumerate(truth):
        candidates: list[tuple[int, float]] = []
        for predicted_index, predicted_event in enumerate(predicted):
            if predicted_event["object_instance_id"] != truth_event["object_instance_id"]:
                continue
            if predicted_event.get("event_type", "MOVE_OBJECT") != truth_event.get("event_type", "MOVE_OBJECT"):
                continue
            score = frame_interval_iou(
                (predicted_event["start_frame"], predicted_event["end_frame"]),
                (truth_event["start_frame"], truth_event["end_frame"]),
                fps=fps,
            )
            if score >= iou_threshold:
                candidates.append((predicted_index, score))
        edges[truth_index] = sorted(
            candidates,
            key=lambda item: (-item[1], predicted[item[0]]["event_id"]),
        )
    matched_predicted_to_truth: dict[int, int] = {}

    def augment(truth_index: int, visited: set[int]) -> bool:
        for predicted_index, _ in edges[truth_index]:
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            previous = matched_predicted_to_truth.get(predicted_index)
            if previous is None or augment(previous, visited):
                matched_predicted_to_truth[predicted_index] = truth_index
                return True
        return False

    for truth_index in range(len(truth)):
        augment(truth_index, set())
    matches = [
        (predicted_index, truth_index, next(score for index, score in edges[truth_index] if index == predicted_index))
        for predicted_index, truth_index in sorted(matched_predicted_to_truth.items())
    ]
    start_errors = []
    end_errors = []
    for predicted_index, truth_index, _ in matches:
        errors = boundary_errors_seconds(
            (predicted[predicted_index]["start_frame"], predicted[predicted_index]["end_frame"]),
            (truth[truth_index]["start_frame"], truth[truth_index]["end_frame"]),
            fps=fps,
        )
        start_errors.append(errors["start_error_seconds"])
        end_errors.append(errors["end_error_seconds"])
    ious = [score for _, _, score in matches]
    return {
        "matched_count": len(matches),
        "precision": _ratio(len(matches), len(predicted), correct=len(matches)),
        "recall": _ratio(len(matches), len(truth), correct=len(matches)),
        "mean_iou": _ratio(sum(ious), len(ious), correct=sum(ious)),
        "mean_start_boundary_error_seconds": _mean_metric(start_errors),
        "mean_end_boundary_error_seconds": _mean_metric(end_errors),
    }


def attribution_metrics(
    *,
    predicted_agent_ids: Sequence[Sequence[str]],
    expected_agent_ids: Sequence[Sequence[str]],
    expected_statuses: Sequence[str],
    known_predicted_agent_ids: set[str] | None = None,
    authorization_available: Sequence[bool] | None = None,
) -> dict[str, dict[str, Any]]:
    _same_length(predicted_agent_ids, expected_agent_ids, expected_statuses)
    if authorization_available is not None:
        _same_length(predicted_agent_ids, authorization_available)
    supported = [index for index, status in enumerate(expected_statuses) if status == "SUPPORTED"]
    non_attributable = [index for index, status in enumerate(expected_statuses) if status != "SUPPORTED"]
    exact = sum(
        set(predicted_agent_ids[index]) == set(expected_agent_ids[index]) for index in supported
    )
    abstained = sum(not predicted_agent_ids[index] for index in non_attributable)
    false_attribution = sum(bool(predicted_agent_ids[index]) for index in non_attributable)
    all_predicted_ids = [person_id for ids in predicted_agent_ids for person_id in ids]
    unresolved = (
        sum(person_id not in known_predicted_agent_ids for person_id in all_predicted_ids)
        if known_predicted_agent_ids is not None
        else 0
    )
    result = {
        "attributable_event_accuracy": _ratio(exact, len(supported), correct=exact),
        "agent_abstention_accuracy": _ratio(abstained, len(non_attributable), correct=abstained),
        "false_attribution_rate": _ratio(false_attribution, len(non_attributable), correct=false_attribution),
        "identity_unresolved_rate": (
            _ratio(unresolved, len(all_predicted_ids), correct=unresolved)
            if known_predicted_agent_ids is not None
            else {"value": NOT_APPLICABLE, "correct": 0, "denominator": 0}
        ),
    }
    available_count = sum(not available for available in authorization_available) if authorization_available is not None else 0
    result["authorization_unavailable_rate"] = _ratio(
        available_count,
        len(authorization_available) if authorization_available is not None else 0,
        correct=available_count,
    )
    return result


def _ratio(numerator: float, denominator: int, *, correct: int | float) -> dict[str, Any]:
    return {
        "value": numerator / denominator if denominator else NOT_APPLICABLE,
        "correct": correct,
        "denominator": denominator,
    }


def _mean_metric(values: Sequence[float]) -> Any:
    return sum(values) / len(values) if values else NOT_APPLICABLE


def _same_length(*values: Sequence[Any]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("metric inputs must have equal lengths")


def _validate_interval(interval: tuple[int, int], label: str) -> None:
    if (
        not isinstance(interval, tuple)
        or len(interval) != 2
        or isinstance(interval[0], bool)
        or isinstance(interval[1], bool)
        or not isinstance(interval[0], int)
        or not isinstance(interval[1], int)
        or interval[0] < 0
        or interval[1] < interval[0]
    ):
        raise ValueError(f"{label} interval must be a non-negative inclusive frame tuple")
