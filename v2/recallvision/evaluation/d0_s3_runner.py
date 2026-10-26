"""Evaluation-only helpers for the frozen EVAL-D0-S3 run.

This module deliberately keeps annotation-to-input conversion separate from
the frozen V1 method.  Only the mechanically allowed query fields are
returned to the inference path; semantic answer fields are never propagated.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


def build_inference_input(query: Mapping[str, Any]) -> dict[str, Any]:
    """Return the leak-free subset of one annotated query record."""

    required = ("query_id", "video_id", "query_type", "natural_language_question")
    for field in required:
        value = query.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    # Annotation-generated event IDs are ground-truth identifiers, not
    # legitimate user input for these template questions.  Keep the field in
    # the output schema for explicit-query compatibility, but fail closed to
    # null unless a future input manifest marks it as user-supplied.
    reference_event_id = None
    return {
        "query_id": query["query_id"],
        "video_id": query["video_id"],
        "query_type": query["query_type"],
        "question": query["natural_language_question"],
        "reference_event_id": reference_event_id,
    }


def evaluate_runtime_preflight(
    *,
    torch_available: bool,
    cv2_available: bool,
    ultralytics_available: bool,
    model_exists: bool,
) -> dict[str, Any]:
    """Return a no-download capability decision for the perception stage."""

    missing = [
        name
        for name, available in (
            ("torch", torch_available),
            ("cv2", cv2_available),
            ("ultralytics", ultralytics_available),
            ("model", model_exists),
        )
        if not available
    ]
    return {
        "status": "READY" if not missing else "CAPABILITY_GAP",
        "missing": missing,
        "reason": None if not missing else "missing local perception capability: " + ", ".join(missing),
        "download_allowed": False,
    }


def run_p2a_for_evaluation(**kwargs: Any) -> Any:
    """Reuse the generic P2A-R1 computation for evaluation inputs."""

    from recallvision.cli.p2a_r1 import run_p2a_r1

    return run_p2a_r1(**kwargs, render_diagnostics=False)


def run_p2c_for_evaluation(
    *,
    timeline_path: Any,
    summary_path: Any,
    config_path: Any,
    output_dir: Any,
) -> dict[str, Any]:
    """Run the frozen P2C builder against all supplied class timelines."""

    import recallvision.cli.p2c as p2c

    config = p2c._load_config(config_path)
    summary = p2c._load_object(summary_path, "P2B summary")
    timelines = p2c._load_timelines(timeline_path)
    if not timelines:
        raise ValueError("P2C evaluation input timeline is empty")
    events = p2c.ZoneTransitionEventBuilder().build_all(timelines)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_records = [event.to_record() for event in events]
    p2c._write_json(output_dir / "zone_change_events.json", event_records)
    p2c._write_text(output_dir / "zone_change_events.txt", p2c._event_audit(events))
    result = p2c._build_summary(
        config=config,
        summary=summary,
        timeline_path=timeline_path,
        timelines=timelines,
        events=events,
    )
    result["evaluation_gate"] = {
        "adapter_only": True,
        "frozen_zone_transition_builder": True,
        "development_p2b_query_status": summary.get("p2b_status"),
        "development_p2b_query_gate_used": False,
        "development_gate_bypass_reason": "P2B query status is independent of the generic P2C timeline input contract",
        "timeline_nonempty": True,
        "p2b_summary_rewritten": False,
        "semantic_method_changes": False,
    }
    p2c._write_json(output_dir / "zone_change_event_summary.json", result)
    return result
