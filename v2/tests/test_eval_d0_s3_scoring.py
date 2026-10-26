"""Small deterministic contracts for the post-seal D0-S3 scorer."""

from __future__ import annotations

from recallvision.evaluation.d0_s3_scoring import (
    classify_failure,
    _event_interval_metrics_with_state,
    macro_average,
    ratio,
    verify_prediction_seal,
)


def test_ratio_reports_not_applicable_for_empty_denominator() -> None:
    assert ratio(0, 0) == {"value": "NOT_APPLICABLE", "correct": 0, "denominator": 0}


def test_macro_average_excludes_empty_video_denominators() -> None:
    assert macro_average([1.0, "NOT_APPLICABLE", 0.5]) == {
        "value": 0.75,
        "denominator": 2,
    }


def test_answerable_who_without_provider_is_a_capability_gap() -> None:
    result = classify_failure(
        query_type="WHO_MOVED_OBJECT",
        expected_status="ANSWER",
        predicted_status="ABSTAIN",
        handler_status="NO_AUTHORIZED_ATTRIBUTION",
        object_binding_status="BOUND",
        who_vlm_inference_executable=False,
    )

    assert result["failure_category"] == "VLM_CAPABILITY_GAP"
    assert result["failure_origin"] == "VLM_CAPABILITY_GAP"
    assert result["capability_gap"] is True


def test_unavailable_perception_runtime_is_not_counted_as_system_accuracy() -> None:
    result = classify_failure(
        query_type="WHERE_OBJECT",
        expected_status="ANSWER",
        predicted_status="ABSTAIN",
        handler_status="PIPELINE_NOT_EXECUTABLE",
        object_binding_status="UNBOUND",
        who_vlm_inference_executable=False,
        pipeline_execution_available=False,
    )

    assert result["failure_category"] == "PERCEPTION_RUNTIME_CAPABILITY_GAP"
    assert result["capability_gap"] is True
    assert result["system_failure"] is False


def test_prediction_seal_rejects_tampered_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"query_id":"q"}\n', encoding="utf-8")
    (tmp_path / "inference_inputs.jsonl").write_text(
        '{"query_id":"q"}\n', encoding="utf-8"
    )
    (tmp_path / "pipeline_index.json").write_text("{}\n", encoding="utf-8")
    import hashlib
    import json

    seal = {
        "status": "SEALED_PHASE_A_PREDICTIONS",
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "inference_inputs_sha256": hashlib.sha256(
            (tmp_path / "inference_inputs.jsonl").read_bytes()
        ).hexdigest(),
        "pipeline_index_sha256": hashlib.sha256(
            (tmp_path / "pipeline_index.json").read_bytes()
        ).hexdigest(),
        "record_count": 1,
    }
    (tmp_path / "prediction_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    predictions.write_text('{"query_id":"tampered"}\n', encoding="utf-8")

    try:
        verify_prediction_seal(tmp_path)
    except ValueError as error:
        assert "predictions SHA256" in str(error)
    else:
        raise AssertionError("tampered predictions were accepted")


def test_event_metrics_reject_zone_state_mismatch() -> None:
    result = _event_interval_metrics_with_state(
        [
            {
                "event_id": "pred-1",
                "event_type": "MOVE_OBJECT",
                "object_instance_id": "obj:cup:1",
                "start_frame": 10,
                "end_frame": 20,
                "zone_before": "left_region",
                "zone_after": "right_region",
            }
        ],
        [
            {
                "event_id": "gt-1",
                "event_type": "MOVE_OBJECT",
                "object_instance_id": "obj:cup:1",
                "start_frame": 10,
                "end_frame": 20,
                "zone_before": "left_region",
                "zone_after": "center_region",
            }
        ],
        fps=30.0,
        iou_threshold=0.5,
    )

    assert result["matched_count"] == 0
    assert result["precision"] == {"correct": 0, "denominator": 1, "value": 0.0}
    assert result["recall"] == {"correct": 0, "denominator": 1, "value": 0.0}
