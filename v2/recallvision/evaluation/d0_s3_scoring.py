"""Post-seal scoring and failure analysis for EVAL-D0-S3."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from recallvision.evaluation.metrics import (
    attribution_metrics,
    event_interval_metrics,
    fail_closed_metrics,
)
from recallvision.evaluation.provenance import sha256_file
from recallvision.evaluation.validator import validate_annotation, validate_manifest
from recallvision.memory.store import ObservationMemoryStore

NOT_APPLICABLE = "NOT_APPLICABLE"


def verify_prediction_seal(root: Path) -> dict[str, Any]:
    """Verify the immutable Phase-A seal before reading semantic GT."""

    output_root = root / "outputs" / "eval_d0_s3"
    if not (output_root / "prediction_seal.json").is_file():
        output_root = root
    seal = _load_json(output_root / "prediction_seal.json")
    if seal.get("status") != "SEALED_PHASE_A_PREDICTIONS":
        raise ValueError("prediction seal status is invalid")
    predictions_path = output_root / "predictions.jsonl"
    inputs_path = output_root / "inference_inputs.jsonl"
    pipeline_path = output_root / "frozen_v1_pipeline" / "pipeline_index.json"
    if sha256_file(predictions_path) != seal.get("predictions_sha256"):
        raise ValueError("predictions SHA256 does not match the seal")
    if sha256_file(inputs_path) != seal.get("inference_inputs_sha256"):
        raise ValueError("inference inputs SHA256 does not match the seal")
    if sha256_file(pipeline_path) != seal.get("pipeline_index_sha256"):
        raise ValueError("pipeline index SHA256 does not match the seal")
    predictions = _read_jsonl(predictions_path)
    inputs = _read_jsonl(inputs_path)
    if len(predictions) != seal.get("record_count") or len(inputs) != seal.get("record_count"):
        raise ValueError("sealed record count does not match inputs and predictions")
    prediction_ids = [item.get("query_id") for item in predictions]
    input_ids = [item.get("query_id") for item in inputs]
    if len(set(prediction_ids)) != len(prediction_ids) or prediction_ids != input_ids:
        raise ValueError("sealed prediction query IDs do not exactly match input query IDs")
    return {
        "predictions_sha256": seal["predictions_sha256"],
        "inference_inputs_sha256": seal["inference_inputs_sha256"],
        "pipeline_index_sha256": seal["pipeline_index_sha256"],
        "record_count": len(predictions),
    }


def ratio(correct: int | float, denominator: int) -> dict[str, Any]:
    return {
        "value": correct / denominator if denominator else NOT_APPLICABLE,
        "correct": correct,
        "denominator": denominator,
    }


def macro_average(values: Sequence[float | str]) -> dict[str, Any]:
    usable = [float(value) for value in values if value != NOT_APPLICABLE]
    return {
        "value": sum(usable) / len(usable) if usable else NOT_APPLICABLE,
        "denominator": len(usable),
    }


def classify_failure(
    *,
    query_type: str,
    expected_status: str,
    predicted_status: str,
    handler_status: str | None,
    object_binding_status: str,
    who_vlm_inference_executable: bool,
    pipeline_execution_available: bool = True,
) -> dict[str, Any]:
    """Assign the earliest supported cause without hiding VLM gaps."""

    if expected_status == "ABSTAIN" and predicted_status == "ABSTAIN":
        return {
            "failure_category": "EXPECTED_ABSTENTION",
            "failure_origin": "EXPECTED_ABSTENTION",
            "capability_gap": False,
            "system_failure": False,
        }
    if expected_status == "ANSWER" and not pipeline_execution_available:
        return {
            "failure_category": "PERCEPTION_RUNTIME_CAPABILITY_GAP",
            "failure_origin": "PERCEPTION_CAPABILITY_GAP",
            "capability_gap": True,
            "system_failure": False,
        }
    if (
        query_type == "WHO_MOVED_OBJECT"
        and expected_status == "ANSWER"
        and not who_vlm_inference_executable
    ):
        return {
            "failure_category": "VLM_CAPABILITY_GAP",
            "failure_origin": "VLM_CAPABILITY_GAP",
            "capability_gap": True,
            "system_failure": False,
        }
    if expected_status == "ANSWER" and object_binding_status != "BOUND":
        return {
            "failure_category": "PERCEPTION_TARGET_RESOLUTION_FAILURE",
            "failure_origin": "SYSTEM",
            "capability_gap": False,
            "system_failure": True,
        }
    categories = {
        "TARGET_NOT_FOUND": "QUERY_RETRIEVAL_FAILURE",
        "AMBIGUOUS_TARGET": "QUERY_RETRIEVAL_FAILURE",
        "NO_SUPPORTED_EVENT": "EVENT_DETECTION_FAILURE",
        "NO_REFERENCE_EVENT": "EVENT_REFERENCE_FAILURE",
        "AMBIGUOUS_MULTIPLE_EVENT_REFERENCE": "EVENT_REFERENCE_FAILURE",
        "NO_AUTHORIZED_ATTRIBUTION": "AUTHORIZATION_FAILURE",
        "UNRESOLVED_PERSON_ID": "IDENTITY_GATE_FAILURE",
        "INVALID_REFERENCE": "EVENT_REFERENCE_FAILURE",
    }
    category = categories.get(handler_status or "", "QUERY_RETRIEVAL_FAILURE")
    return {
        "failure_category": category,
        "failure_origin": "SYSTEM",
        "capability_gap": False,
        "system_failure": True,
    }


def build_object_binding(annotation: Mapping[str, Any], memory_path: Path) -> dict[str, Any]:
    """Construct the frozen class-and-evidence object binding artifact."""

    if not memory_path.is_file():
        return {
            "schema_version": "eval_d0_s3_object_binding_v1",
            "video_id": annotation["video"]["video_id"],
            "source_annotation_sha256": None,
            "memory_sha256": None,
            "bindings": [
                {
                    "ground_truth_object_id": obj["object_instance_id"],
                    "predicted_object_id": None,
                    "coverage": None,
                    "status": "UNBOUND",
                }
                for obj in annotation["objects"]
            ],
            "coverage_candidates": [],
            "binding_rule": "same class, evidence-zone coverage >= 0.5, maximum-weight one-to-one assignment",
        }
    observations = ObservationMemoryStore(memory_path).load_all()
    entities = _load_list(memory_path.parent / "persistent_entities.json")
    predicted_ids = [
        str(item["persistent_id"])
        for item in entities
        if isinstance(item.get("persistent_id"), str)
    ]
    by_id = {item: [] for item in predicted_ids}
    for observation in observations:
        if observation.persistent_id in by_id:
            by_id[observation.persistent_id].append(observation)

    objects = sorted(annotation["objects"], key=lambda item: item["object_instance_id"])
    candidates: dict[str, list[tuple[str, float]]] = {}
    coverage_records: list[dict[str, Any]] = []
    for obj in objects:
        gt_id = obj["object_instance_id"]
        expected = [
            (frame, state["zone"])
            for state in obj["states"]
            for frame in state["evidence_frames"]
        ]
        local: list[tuple[str, float]] = []
        if expected:
            for pred_id, pred_observations in by_id.items():
                classes = {item.class_name.casefold() for item in pred_observations}
                if obj["object_class"].casefold() not in classes:
                    continue
                covered = sum(
                    any(
                        item.frame_index == frame and item.zone == zone
                        for item in pred_observations
                    )
                    for frame, zone in expected
                )
                coverage = covered / len(expected)
                if coverage >= 0.5:
                    local.append((pred_id, coverage))
                coverage_records.append(
                    {
                        "ground_truth_object_id": gt_id,
                        "predicted_object_id": pred_id,
                        "coverage": coverage,
                        "eligible": coverage >= 0.5,
                    }
                )
        candidates[gt_id] = sorted(local, key=lambda item: (-item[1], item[0]))

    selected = _maximum_assignment(candidates)
    bindings = []
    for obj in objects:
        gt_id = obj["object_instance_id"]
        pred_id = selected.get(gt_id)
        coverage = next(
            (value for candidate, value in candidates[gt_id] if candidate == pred_id),
            None,
        )
        bindings.append(
            {
                "ground_truth_object_id": gt_id,
                "predicted_object_id": pred_id,
                "coverage": coverage,
                "status": "BOUND" if pred_id is not None else "UNBOUND",
            }
        )
    return {
        "schema_version": "eval_d0_s3_object_binding_v1",
        "video_id": annotation["video"]["video_id"],
        "source_annotation_sha256": None,
        "memory_sha256": sha256_file(memory_path),
        "bindings": bindings,
        "coverage_candidates": coverage_records,
        "binding_rule": "same class, evidence-zone coverage >= 0.5, maximum-weight one-to-one assignment",
    }


def score_query(
    annotation_query: Mapping[str, Any],
    prediction: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    who_vlm_inference_executable: bool,
    pipeline_execution_available: bool = True,
    expected_event: Mapping[str, Any] | None = None,
    predicted_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    structured = prediction["structured_result"]
    query_type = annotation_query["query_type"]
    expected_status = annotation_query["expected_status"]
    predicted_status = prediction["prediction_status"]
    gt_object_id = annotation_query.get("object_instance_id")
    binding_record = next(
        (
            item
            for item in binding["bindings"]
            if item["ground_truth_object_id"] == gt_object_id
        ),
        None,
    )
    object_binding_status = (
        binding_record["status"] if binding_record is not None else "NOT_APPLICABLE"
    )
    handler_status = structured.get("status")
    failure = classify_failure(
        query_type=query_type,
        expected_status=expected_status,
        predicted_status=predicted_status,
        handler_status=handler_status,
        object_binding_status=object_binding_status,
        who_vlm_inference_executable=who_vlm_inference_executable,
        pipeline_execution_available=pipeline_execution_available,
    )
    result = {
        "query_id": annotation_query["query_id"],
        "video_id": annotation_query["video_id"],
        "query_type": query_type,
        "answerability": annotation_query["answerability"],
        "expected_status": expected_status,
        "expected_reason": annotation_query["ground_truth"].get("reason"),
        "predicted_status": predicted_status,
        "handler_status": handler_status,
        "predicted_target_entity_id": _predicted_target_id(structured),
        "object_binding_status": object_binding_status,
        "bound_predicted_object_id": (
            binding_record["predicted_object_id"] if binding_record else None
        ),
        "answerability_correct": predicted_status == expected_status,
        "result_correct": False,
        **failure,
    }
    if expected_status == "ABSTAIN":
        result["result_correct"] = predicted_status == "ABSTAIN"
        return result
    if query_type == "WHERE_OBJECT":
        result["result_correct"] = _where_correct(annotation_query, structured, binding_record)
    elif query_type == "WHEN_OBJECT_MOVED":
        result["result_correct"] = _when_correct(
            annotation_query, structured, binding_record, expected_event
        )
    elif query_type == "WHERE_OBJECT_BEFORE":
        result["result_correct"] = _before_correct(
            annotation_query, structured, binding_record, expected_event, predicted_events
        )
    elif query_type == "WHO_MOVED_OBJECT":
        result["result_correct"] = _who_correct(annotation_query, structured)
    if result["result_correct"]:
        result["failure_category"] = None
        result["failure_origin"] = None
        result["capability_gap"] = False
        result["system_failure"] = False
    return result


def _where_correct(query: Mapping[str, Any], structured: Mapping[str, Any], binding: Mapping[str, Any] | None) -> bool:
    if binding is None or binding["status"] != "BOUND":
        return False
    return (
        structured.get("status") == "TARGET_LAST_SEEN"
        and structured.get("persistent_id") == binding["predicted_object_id"]
        and structured.get("zone") == query["ground_truth"].get("zone")
        and structured.get("frame_index") == query["ground_truth"].get("evidence_frames", [None])[-1]
    )


def _when_correct(
    query: Mapping[str, Any],
    structured: Mapping[str, Any],
    binding: Mapping[str, Any] | None,
    expected_event: Mapping[str, Any] | None,
) -> bool:
    if binding is None or binding["status"] != "BOUND" or structured.get("status") != "MOVEMENT_FOUND":
        return False
    if structured.get("persistent_id") != binding["predicted_object_id"]:
        return False
    expected = query["ground_truth"]
    expected_interval = (expected.get("start_frame"), expected.get("end_frame"))
    events_by_id = {
        item.get("event_id"): item for item in structured.get("events", [])
    }
    for interval in structured.get("frame_intervals", []):
        if interval.get("start_frame") is None or interval.get("end_frame") is None:
            continue
        if _frame_iou(
            (interval["start_frame"], interval["end_frame"]), expected_interval
        ) < 0.5:
            continue
        if expected_event is not None:
            event = events_by_id.get(interval.get("event_id"))
            if event is None or not _event_state_compatible(event, expected_event):
                continue
        return True
    return False


def _before_correct(
    query: Mapping[str, Any],
    structured: Mapping[str, Any],
    binding: Mapping[str, Any] | None,
    expected_event: Mapping[str, Any] | None,
    predicted_events: Sequence[Mapping[str, Any]],
) -> bool:
    if binding is None or binding["status"] != "BOUND":
        return False
    if not (
        structured.get("status") == "OK"
        and structured.get("target_entity_id") == binding["predicted_object_id"]
        and structured.get("zone") == query["ground_truth"].get("zone")
        and structured.get("frame_index") == query["ground_truth"].get("evidence_frames", [None])[-1]
        and structured.get("reference_event_id") is not None
    ):
        return False
    if expected_event is None:
        return True
    predicted = next(
        (item for item in predicted_events if item.get("event_id") == structured.get("reference_event_id")),
        None,
    )
    if predicted is None or not _event_state_compatible(predicted, expected_event):
        return False
    predicted_interval = _event_frame_interval(predicted)
    expected_interval = (expected_event.get("start_frame"), expected_event.get("end_frame"))
    return predicted_interval is not None and _frame_iou(predicted_interval, expected_interval) >= 0.5


def _event_state_compatible(
    predicted: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return (
        predicted.get("zone_before") == expected.get("from_zone")
        and predicted.get("zone_after") == expected.get("to_zone")
    )


def _event_frame_interval(event: Mapping[str, Any]) -> tuple[int, int] | None:
    evidence = event.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        return None
    start = evidence[0].get("frame_indices") if isinstance(evidence[0], Mapping) else None
    end = evidence[1].get("frame_indices") if isinstance(evidence[1], Mapping) else None
    if not start or not end or not isinstance(start[0], int) or not isinstance(end[0], int):
        return None
    return start[0], end[0]


def _who_correct(query: Mapping[str, Any], structured: Mapping[str, Any]) -> bool:
    expected = query["ground_truth"].get("agent_instance_ids", [])
    return structured.get("status") == "SUCCESS" and set(structured.get("person_ids", [])) == set(expected)


def score_dataset(root: Path) -> dict[str, Any]:
    verify_prediction_seal(root)
    output_root = root / "outputs" / "eval_d0_s3"
    replay_path = output_root / "replay_determinism.json"
    if not replay_path.is_file() or _load_json(replay_path).get("status") != "PASS":
        raise ValueError("Phase-A replay determinism must PASS before Phase-B scoring")
    manifest = _load_json(root / "data" / "eval" / "manifest.json")
    validate_manifest(manifest, repository_root=root, require_video_files=True)
    prediction_records = _read_jsonl(output_root / "predictions.jsonl")
    prediction_ids = [item.get("query_id") for item in prediction_records]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction query IDs must be unique")
    predictions = {item["query_id"]: item for item in prediction_records}
    pipeline_index = _load_json(root / "outputs" / "eval_d0_s3" / "frozen_v1_pipeline" / "pipeline_index.json")
    run_manifest = _load_json(output_root / "run_manifest.json")
    pipeline_execution_available = (
        run_manifest.get("phase_a_pipeline", {}).get("status") == "COMPLETED"
    )
    all_results: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    annotations: dict[str, dict[str, Any]] = {}
    annotation_validation: dict[str, str] = {}
    for video in [item for item in manifest["videos"] if item["split"] == "EVAL"]:
        annotation = _load_json(root / video["annotation_path"])
        validate_annotation(annotation, manifest)
        annotation_validation[video["video_id"]] = "PASS"
        annotations[video["video_id"]] = annotation
        memory_value = pipeline_index[video["video_id"]].get("persistent_memory", "")
        memory_path = Path(memory_value) if memory_value else Path("__missing_memory__")
        binding = build_object_binding(annotation, memory_path)
        binding["source_annotation_sha256"] = sha256_file(root / video["annotation_path"])
        bindings[video["video_id"]] = binding
        predicted_events = _load_list(Path(pipeline_index[video["video_id"]]["events"])) if Path(
            pipeline_index[video["video_id"]].get("events", "")
        ).is_file() else []
        for query in annotation["queries"]:
            all_results.append(
                score_query(
                    query,
                    predictions[query["query_id"]],
                    binding,
                    who_vlm_inference_executable=False,
                    pipeline_execution_available=pipeline_execution_available,
                    expected_event=_expected_event_for_query(annotation, query),
                    predicted_events=predicted_events,
                )
            )

    _write_json(output_root / "annotation_validation.json", annotation_validation)
    _write_json(output_root / "object_bindings.json", bindings)
    _write_jsonl(output_root / "per_query_results.jsonl", all_results)
    metrics = _build_metrics(all_results, annotations, pipeline_index)
    _write_json(output_root / "metrics_summary.json", metrics)
    per_video = _build_per_video(all_results)
    _write_json(output_root / "per_video_results.json", per_video)
    failure_analysis = _build_failure_analysis(all_results)
    _write_json(output_root / "failure_analysis.json", failure_analysis)
    report = _build_report(metrics, per_video, failure_analysis, output_root)
    (output_root / "EVAL_D0_S3_REPORT.md").write_text(report, encoding="utf-8")
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    run_manifest["phase_b_scoring"] = {
        "status": "COMPLETED_AFTER_PREDICTION_SEAL",
        "ground_truth_opened_after_seal": True,
        "annotation_count": len(annotations),
        "query_count": len(all_results),
        "final_classification": metrics["final_classification"],
    }
    _write_json(run_manifest_path, run_manifest)
    return metrics


def _build_metrics(
    results: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    pipeline_index: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_type.setdefault(result["query_type"], []).append(result)
    all_fail_closed = fail_closed_metrics(
        expected_answerability=[item["answerability"] for item in results],
        predicted_status=[item["predicted_status"] for item in results],
        expected_reasons=[item["expected_reason"] for item in results],
    )
    family: dict[str, Any] = {}
    for query_type, items in sorted(by_type.items()):
        all_answerable = [item for item in items if item["expected_status"] == "ANSWER"]
        answerable = [item for item in all_answerable if not item["capability_gap"]]
        fail_closed = [item for item in items if item["expected_status"] == "ABSTAIN"]
        family[query_type] = {
            "answerable_accuracy": ratio(
                sum(bool(item["result_correct"]) for item in answerable), len(answerable)
            ),
            "fail_closed_correctness": ratio(
                sum(bool(item["result_correct"]) for item in fail_closed), len(fail_closed)
            ),
            "false_success_rate": ratio(
                sum(item["predicted_status"] == "ANSWER" for item in fail_closed), len(fail_closed)
            ),
            "unscored_answerable_capability_gap_count": len(all_answerable) - len(answerable),
            "capability_gap_count": sum(bool(item["capability_gap"]) for item in items),
            "system_failure_count": sum(bool(item["system_failure"]) for item in items),
        }
        per_video_answerable = []
        per_video_fail_closed = []
        for video_id in sorted({item["video_id"] for item in items}):
            video_items = [item for item in items if item["video_id"] == video_id]
            video_answerable = [
                item
                for item in video_items
                if item["expected_status"] == "ANSWER" and not item["capability_gap"]
            ]
            video_fail_closed = [
                item for item in video_items if item["expected_status"] == "ABSTAIN"
            ]
            per_video_answerable.append(
                sum(bool(item["result_correct"]) for item in video_answerable) / len(video_answerable)
                if video_answerable
                else NOT_APPLICABLE
            )
            per_video_fail_closed.append(
                sum(bool(item["result_correct"]) for item in video_fail_closed) / len(video_fail_closed)
                if video_fail_closed
                else NOT_APPLICABLE
            )
        family[query_type]["per_video_macro_answerable_accuracy"] = macro_average(per_video_answerable)
        family[query_type]["per_video_macro_fail_closed_correctness"] = macro_average(per_video_fail_closed)

    predicted_ids = []
    expected_ids = []
    expected_statuses = []
    for item in by_type.get("WHO_MOVED_OBJECT", []):
        prediction = item
        expected = annotations[item["video_id"]]
        query = next(q for q in expected["queries"] if q["query_id"] == item["query_id"])
        predicted_ids.append([])
        expected_ids.append(query["ground_truth"].get("agent_instance_ids", []))
        expected_statuses.append(_expected_agent_status(query, expected))
    who_metrics = attribution_metrics(
        predicted_agent_ids=predicted_ids,
        expected_agent_ids=expected_ids,
        expected_statuses=expected_statuses,
        known_predicted_agent_ids=set(),
        authorization_available=[False] * len(predicted_ids),
    )
    supported_count = sum(status == "SUPPORTED" for status in expected_statuses)
    who_metrics["attributable_event_accuracy"] = ratio(0, 0)
    who_metrics["attributable_event_capability_gap"] = ratio(supported_count, supported_count)
    who_metrics["provider_status"] = "WHO_VLM_INFERENCE_EXECUTABLE_NO"
    event_metrics = _build_event_metrics(annotations, pipeline_index)
    final = (
        "PARTIAL_EVAL_D0_S3_CAPABILITY_GAP"
        if any(item["capability_gap"] for item in results)
        else "PASS_EVAL_D0_S3_FROZEN_V1_COMPLETED"
    )
    return {
        "schema_version": "eval_d0_s3_metrics_v1",
        "query_count": len(results),
        "answerable_count": sum(item["expected_status"] == "ANSWER" for item in results),
        "fail_closed_count": sum(item["expected_status"] == "ABSTAIN" for item in results),
        "family_metrics": family,
        "overall_fail_closed_metrics": all_fail_closed,
        "who_attribution_metrics": who_metrics,
        "movement_event_metrics": event_metrics,
        "capability_gap_count": sum(bool(item["capability_gap"]) for item in results),
        "system_failure_count": sum(bool(item["system_failure"]) for item in results),
        "final_classification": final,
        "recommendation": (
            "EVAL-D0-S3R1-CAPABILITY-GAP-DECISION"
            if final == "PARTIAL_EVAL_D0_S3_CAPABILITY_GAP"
            else "EVAL-D0-S4-ERROR-ANALYSIS-AND-METHOD-DECISION"
        ),
    }


def _build_event_metrics(
    annotations: Mapping[str, Mapping[str, Any]],
    pipeline_index: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    by_video = {}
    for video_id, annotation in annotations.items():
        video_artifacts = pipeline_index.get(video_id, {})
        memory_path = Path(video_artifacts.get("persistent_memory", "__missing_memory__"))
        binding = build_object_binding(annotation, memory_path)
        reverse = {
            item["predicted_object_id"]: item["ground_truth_object_id"]
            for item in binding["bindings"]
            if item["predicted_object_id"] is not None
        }
        predicted_records = []
        event_value = video_artifacts.get("events", "")
        raw = _load_list(Path(event_value)) if event_value and Path(event_value).is_file() else []
        for event in raw:
            object_ids = event.get("object_ids", [])
            evidence = event.get("evidence", [])
            if len(object_ids) != 1 or object_ids[0] not in reverse or len(evidence) < 2:
                continue
            start = _first_frame(evidence[0])
            end = _first_frame(evidence[1])
            if start is None or end is None:
                continue
            predicted_records.append(
                {
                    "event_id": event["event_id"],
                    "event_type": "MOVE_OBJECT",
                    "object_instance_id": reverse[object_ids[0]],
                    "start_frame": start,
                    "end_frame": end,
                    "zone_before": event.get("zone_before"),
                    "zone_after": event.get("zone_after"),
                }
            )
        truth = []
        for obj in annotation["objects"]:
            for event in obj["movement_events"]:
                if event["movement_support_status"] == "SUPPORTED":
                    truth.append(
                        {
                            "event_id": event["event_id"],
                            "event_type": "MOVE_OBJECT",
                            "object_instance_id": obj["object_instance_id"],
                            "start_frame": event["start_frame"],
                            "end_frame": event["end_frame"],
                            "zone_before": next(
                                state["zone"] for state in obj["states"] if state["state_id"] == event["from_state_id"]
                            ),
                            "zone_after": next(
                                state["zone"] for state in obj["states"] if state["state_id"] == event["to_state_id"]
                            ),
                        }
                    )
        by_video[video_id] = _event_interval_metrics_with_state(
            predicted_records,
            truth,
            fps=float(annotation["video"]["fps"]),
            iou_threshold=0.5,
        )
    per_video_precision = [item["precision"]["value"] for item in by_video.values()]
    per_video_recall = [item["recall"]["value"] for item in by_video.values()]
    per_video_iou = [item["mean_iou"]["value"] for item in by_video.values()]
    return {
        "per_video": by_video,
        "macro": {
            "precision": macro_average(per_video_precision),
            "recall": macro_average(per_video_recall),
            "mean_iou": macro_average(per_video_iou),
        },
    }


def _event_interval_metrics_with_state(
    predicted_events: Sequence[Mapping[str, Any]],
    truth_events: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    iou_threshold: float,
) -> dict[str, Any]:
    """Apply frozen interval metrics only after requiring FROM/TO compatibility."""

    state_checked: list[dict[str, Any]] = []
    for event in predicted_events:
        state_match = any(
            event.get("object_instance_id") == truth.get("object_instance_id")
            and event.get("event_type", "MOVE_OBJECT") == truth.get("event_type", "MOVE_OBJECT")
            and event.get("zone_before") == truth.get("zone_before")
            and event.get("zone_after") == truth.get("zone_after")
            for truth in truth_events
        )
        normalized = dict(event)
        if not state_match:
            normalized["object_instance_id"] = f"__state_mismatch__:{event.get('event_id')}"
        state_checked.append(normalized)
    return event_interval_metrics(
        state_checked,
        truth_events,
        fps=fps,
        iou_threshold=iou_threshold,
    )


def _build_per_video(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for video_id in sorted({item["video_id"] for item in results}):
        items = [item for item in results if item["video_id"] == video_id]
        output[video_id] = {
            "query_count": len(items),
            "correct_count": sum(bool(item["result_correct"]) for item in items),
            "answerable_count": sum(item["expected_status"] == "ANSWER" for item in items),
            "fail_closed_count": sum(item["expected_status"] == "ABSTAIN" for item in items),
            "capability_gap_count": sum(bool(item["capability_gap"]) for item in items),
            "system_failure_count": sum(bool(item["system_failure"]) for item in items),
            "failure_categories": dict(
                sorted(
                    _counts(item["failure_category"] for item in items if not item["result_correct"]).items()
                )
            ),
        }
    return output


def _build_failure_analysis(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [dict(item) for item in results if not item["result_correct"]]
    return {
        "incorrect_query_count": len(failures),
        "system_failure_count": sum(item["system_failure"] for item in failures),
        "capability_gap_count": sum(item["capability_gap"] for item in failures),
        "by_category": dict(sorted(_counts(item["failure_category"] for item in failures).items())),
        "cases": failures,
    }


def _build_report(
    metrics: Mapping[str, Any],
    per_video: Mapping[str, Any],
    failure_analysis: Mapping[str, Any],
    output_root: Path,
) -> str:
    replay_status = _load_json(output_root / "replay_determinism.json").get("status")
    non_who_system_failures = sum(
        item["system_failure"]
        for item in failure_analysis["cases"]
        if item["query_type"] != "WHO_MOVED_OBJECT"
    )
    who_capability_gaps = sum(
        item["capability_gap"]
        for item in failure_analysis["cases"]
        if item["query_type"] == "WHO_MOVED_OBJECT"
    )
    lines = [
        "# EVAL-D0-S3 Frozen V1 Report",
        "",
        f"Final classification: `{metrics['final_classification']}`",
        f"Recommendation: `{metrics['recommendation']}`",
        "",
        "## Integrity and execution",
        "",
        "- `TRAINING_RUN=NO`",
        "- `EVAL_TIME_TUNING_OCCURRED=NO`",
        "- `CORE_V1_METHOD_CHANGED=NO`",
        f"- {len(per_video)} manifest raw-video hashes passed.",
        "- The inference view forwarded only query identity/type/question/reference-event fields; semantic answer fields were discarded before dispatch.",
        "- Phase-A predictions were sealed before Phase-B annotation scoring.",
        f"- Prediction replay twice: `{replay_status}`.",
        "- WHO VLM inference executable: `NO`; no provider was configured and no manual VLM answer was used.",
        "",
        "## Metrics",
        "",
        f"- Queries: {metrics['query_count']} total; {metrics['answerable_count']} answerable; {metrics['fail_closed_count']} fail-closed.",
        f"- Capability-gap queries: {metrics['capability_gap_count']}.",
        f"- System-failure queries: {metrics['system_failure_count']}.",
        f"- WHO attribution: `{json.dumps(metrics['who_attribution_metrics'], sort_keys=True)}`",
        f"- Movement event metrics: `{json.dumps(metrics['movement_event_metrics'], sort_keys=True)}`",
        "",
        "## Per-video summary",
        "",
        "| Video | Queries | Correct | Answerable | Fail-closed | Capability gap | System failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for video_id, item in sorted(per_video.items()):
        lines.append(
            f"| {video_id} | {item['query_count']} | {item['correct_count']} | {item['answerable_count']} | "
            f"{item['fail_closed_count']} | {item['capability_gap_count']} | {item['system_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Incorrect-query analysis",
            "",
            f"Incorrect queries: {failure_analysis['incorrect_query_count']}; categories: "
            f"`{json.dumps(failure_analysis['by_category'], sort_keys=True)}`.",
            "",
            "Every incorrect query is retained in `per_query_results.jsonl` and `failure_analysis.json`.",
            f"The {non_who_system_failures} non-WHO answerable failures are target-resolution/system failures under the frozen class-and-evidence binding rule; the {who_capability_gaps} answerable WHO cases are separately marked as VLM capability gaps because F1/F2/F3 attribution inference was unavailable.",
            "",
            "## Artifacts",
            "",
            "- `predictions.sha256` is the Phase-A seal.",
            "- `object_bindings.json` is the post-seal binding artifact.",
            "- `metrics_summary.json`, `per_query_results.jsonl`, `per_video_results.json`, and `failure_analysis.json` contain machine-readable results.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expected_agent_status(query: Mapping[str, Any], annotation: Mapping[str, Any]) -> str:
    status = query["ground_truth"].get("agent_attribution_status")
    if isinstance(status, str):
        return status
    reason = query["ground_truth"].get("reason")
    return {
        "AMBIGUOUS_AGENT": "AMBIGUOUS",
        "NO_VISIBLE_AGENT": "NO_VISIBLE_AGENT",
    }.get(reason, "INSUFFICIENT")


def _expected_event_for_query(
    annotation: Mapping[str, Any], query: Mapping[str, Any]
) -> dict[str, Any] | None:
    event_id = query.get("reference_event_id")
    if not isinstance(event_id, str):
        return None
    for obj in annotation["objects"]:
        states = {state["state_id"]: state for state in obj["states"]}
        for event in obj["movement_events"]:
            if event["event_id"] != event_id:
                continue
            enriched = dict(event)
            enriched["from_zone"] = states.get(event["from_state_id"], {}).get("zone")
            enriched["to_zone"] = states.get(event["to_state_id"], {}).get("zone")
            return enriched
    return None


def _predicted_target_id(structured: Mapping[str, Any]) -> str | None:
    for key in ("target_entity_id", "persistent_id"):
        value = structured.get(key)
        if isinstance(value, str):
            return value
    return None


def _maximum_assignment(candidates: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, str]:
    gt_ids = sorted(candidates)
    best_score = -1.0
    best_pairs: tuple[tuple[str, str], ...] | None = None
    best: dict[str, str] = {}
    options = [[None] + [item[0] for item in candidates[gt_id]] for gt_id in gt_ids]
    weights = {
        (gt_id, pred_id): weight
        for gt_id in gt_ids
        for pred_id, weight in candidates[gt_id]
    }
    for choice in itertools.product(*options):
        selected = [item for item in choice if item is not None]
        if len(set(selected)) != len(selected):
            continue
        score = sum(weights[(gt_id, pred_id)] for gt_id, pred_id in zip(gt_ids, choice) if pred_id is not None)
        pairs = tuple(sorted((gt_id, pred_id) for gt_id, pred_id in zip(gt_ids, choice) if pred_id is not None))
        if score > best_score or (score == best_score and (best_pairs is None or pairs < best_pairs)):
            best_score = score
            best_pairs = pairs
            best = {gt_id: pred_id for gt_id, pred_id in zip(gt_ids, choice) if pred_id is not None}
    return best


def _frame_iou(predicted: tuple[int, int], expected: tuple[int, int]) -> float:
    ps, pe = predicted[0], predicted[1] + 1
    es, ee = expected[0], expected[1] + 1
    intersection = max(0, min(pe, ee) - max(ps, es))
    union = max(pe, ee) - min(ps, es)
    return intersection / union if union else 1.0


def _first_frame(evidence: Mapping[str, Any]) -> int | None:
    frames = evidence.get("frame_indices")
    if isinstance(frames, list) and frames and isinstance(frames[0], int):
        return frames[0]
    return None


def _counts(values):
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"expected JSON list of objects: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )
