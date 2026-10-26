"""Regression checks for the committed EVAL-D0-S1 data-collection gate."""

from __future__ import annotations

import json
from pathlib import Path

from recallvision.evaluation.contracts import QUERY_TYPES
from recallvision.evaluation.validator import validate_annotation, validate_manifest


ROOT = Path(__file__).parents[1]
EXPECTED_VIDEO_IDS = [
    "eval_d0_001",
    "eval_d0_002",
    "eval_d0_003",
    "eval_d0_004",
    "eval_d0_005",
    "eval_d0_006",
]
REQUIRED_LEDGER_FIELDS = {
    "candidate_id",
    "source_page",
    "provider",
    "license_note",
    "inspection_date",
    "fixed_camera",
    "indoor",
    "continuous_sequence",
    "usable_resolution",
    "queryable_objects",
    "temporal_object_activity",
    "selection_status",
    "selection_reason",
}
ALLOWED_SELECTION_STATUSES = {
    "SELECTED",
    "REJECTED_PROTOCOL_INELIGIBLE",
    "REJECTED_PROVENANCE_INSUFFICIENT",
    "REJECTED_DUPLICATE_CONTENT",
    "REJECTED_TECHNICAL_QUALITY",
}


def _json(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text())


def test_committed_s1_manifest_contains_six_ordered_eval_videos_and_two_annotations() -> None:
    manifest = _json("data/eval/manifest.json")
    validate_manifest(manifest)

    selected = [video for video in manifest["videos"] if video["split"] == "EVAL"]
    assert [video["video_id"] for video in selected] == EXPECTED_VIDEO_IDS
    assert manifest["pilot"]["selected_eval_video_count"] == 6
    assert manifest["pilot"]["annotation_dry_run_video_count"] == 2
    assert manifest["pilot"]["fully_annotated_video_count"] == 2
    assert manifest["pilot"]["annotation_schema_status"] == "FROZEN"
    assert manifest["evaluation"]["development_video_ids"] == ["p1b_pixabay_bottle"]

    for video in selected[:2]:
        annotation = _json(video["annotation_path"])
        validate_annotation(annotation, manifest)
        assert annotation["schema_version"] == "evaluation_annotation_v1"
        assert annotation["video"]["video_id"] == video["video_id"]
        assert set(video["eligible_query_types"]) == set(QUERY_TYPES)


def test_selection_ledger_records_every_candidate_without_model_failure_status() -> None:
    ledger = _json("data/eval/video_selection_ledger.json")
    candidates = ledger["candidates"]

    assert ledger["schema_version"] == "eval_d0_selection_ledger_v1"
    assert ledger["selection_policy_frozen"] is True
    assert len(candidates) >= 6
    for candidate in candidates:
        assert REQUIRED_LEDGER_FIELDS <= set(candidate)
        assert candidate["selection_status"] in ALLOWED_SELECTION_STATUSES
        assert candidate["selection_status"] != "REJECTED_MODEL_FAILURE"
        assert candidate["inspection_date"] == "2026-10-04"

    selected = [candidate for candidate in candidates if candidate["selection_status"] == "SELECTED"]
    assert [candidate["video_id"] for candidate in selected] == EXPECTED_VIDEO_IDS
