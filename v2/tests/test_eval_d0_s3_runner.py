"""Leakage guards for the evaluation-only D0-S3 runner."""

from __future__ import annotations

import pytest

from recallvision.evaluation.d0_s3_runner import (
    build_inference_input,
    evaluate_runtime_preflight,
    run_p2c_for_evaluation,
    run_p2a_for_evaluation,
)


def test_inference_input_contains_only_legitimate_query_fields() -> None:
    query = {
        "query_id": "eval_d0_001:query:object:WHERE_OBJECT",
        "video_id": "eval_d0_001",
        "query_type": "WHERE_OBJECT",
        "natural_language_question": "Where is the duffel bag?",
        "reference_event_id": None,
        "answerability": "ANSWERABLE",
        "expected_status": "ANSWER",
        "ground_truth": {"zone": "center_region"},
    }

    result = build_inference_input(query)

    assert result == {
        "query_id": "eval_d0_001:query:object:WHERE_OBJECT",
        "video_id": "eval_d0_001",
        "query_type": "WHERE_OBJECT",
        "question": "Where is the duffel bag?",
        "reference_event_id": None,
    }


def test_inference_input_rejects_missing_question() -> None:
    with pytest.raises(ValueError, match="natural_language_question"):
        build_inference_input(
            {
                "query_id": "q",
                "video_id": "eval_d0_001",
                "query_type": "WHERE_OBJECT",
                "reference_event_id": None,
            }
        )


def test_annotation_derived_event_reference_is_not_forwarded() -> None:
    result = build_inference_input(
        {
            "query_id": "q",
            "video_id": "eval_d0_001",
            "query_type": "WHERE_OBJECT_BEFORE",
            "natural_language_question": "Where was the object before?",
            "reference_event_id": "gt:event:001:object:move",
        }
    )

    assert result["reference_event_id"] is None


def test_runtime_preflight_fails_closed_without_local_perception_stack() -> None:
    result = evaluate_runtime_preflight(
        torch_available=False,
        cv2_available=False,
        ultralytics_available=False,
        model_exists=False,
    )

    assert result["status"] == "CAPABILITY_GAP"
    assert result["download_allowed"] is False
    assert "model" in result["reason"]


def test_p2a_evaluation_adapter_delegates_to_shared_generic_entrypoint(monkeypatch, tmp_path) -> None:
    import recallvision.cli.p2a_r1 as p2a

    expected = object()
    calls = {}

    def shared_entrypoint(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(p2a, "run_p2a_r1", shared_entrypoint)
    inputs = {
        "input_memory": tmp_path / "observations.jsonl",
        "input_video": tmp_path / "video.mp4",
        "output_dir": tmp_path / "p2a",
        "config_path": tmp_path / "config.json",
    }

    result = run_p2a_for_evaluation(**inputs)

    assert result is expected
    assert calls == {**inputs, "render_diagnostics": False}


def test_p2c_evaluation_adapter_does_not_rewrite_p2b_status(monkeypatch, tmp_path) -> None:
    import json
    import recallvision.cli.p2c as p2c

    timeline_path = tmp_path / "timelines.json"
    summary_path = tmp_path / "p2b_summary.json"
    config_path = tmp_path / "p2c.json"
    output_dir = tmp_path / "p2c"
    timeline_path.write_text("[]\n", encoding="utf-8")
    summary_path.write_text(json.dumps({"p2b_status": "TARGET_NOT_FOUND"}), encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")

    class FakeTimeline:
        video_id = "eval_d0_001"

    class FakeBuilder:
        def build_all(self, _timelines):
            return []

    monkeypatch.setattr(p2c, "_load_config", lambda _path: {"source": "test", "exact_transition_time_claimed": False})
    monkeypatch.setattr(p2c, "_load_object", lambda _path, _label: {"p2b_status": "TARGET_NOT_FOUND"})
    monkeypatch.setattr(p2c, "_load_timelines", lambda _path: [FakeTimeline()])
    monkeypatch.setattr(p2c, "ZoneTransitionEventBuilder", lambda: FakeBuilder())

    result = run_p2c_for_evaluation(
        timeline_path=timeline_path,
        summary_path=summary_path,
        config_path=config_path,
        output_dir=output_dir,
    )

    assert result["input_p2b_status"] == "TARGET_NOT_FOUND"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["p2b_status"] == "TARGET_NOT_FOUND"
