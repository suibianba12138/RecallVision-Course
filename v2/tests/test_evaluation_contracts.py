"""Failing-first tests for EVAL-D0 manifest and annotation contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from recallvision.evaluation.contracts import (
    ANNOTATION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    QUERY_TYPES,
)
from recallvision.evaluation.validator import (
    EvaluationValidationError,
    validate_annotation,
    validate_eval_video_selection,
    validate_manifest,
)


def _video(*, video_id: str = "eval:video:001", split: str = "EVAL") -> dict[str, object]:
    return {
        "video_id": video_id,
        "source_type": "LOCAL_PROVENANCE_RECORD",
        "source_provenance": {"source_uri": "https://example.invalid/video-001"},
        "acquisition_date": "2026-10-04",
        "relative_video_path": "data/eval/raw/video-001.mp4",
        "sha256": "a" * 64,
        "duration_seconds": 12.0,
        "fps": 30.0,
        "frame_count": 360,
        "width": 640,
        "height": 360,
        "split": split,
        "annotation_path": "annotations/eval_v1/eval_video_001.json",
        "eligible_query_types": list(QUERY_TYPES),
        "license_note": "source provenance retained in the acquisition record",
        "notes": "fixed-camera indoor pilot candidate",
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pilot": {
            "target_video_count": 6,
            "selected_eval_video_count": 1,
            "annotation_dry_run_target_count": 2,
            "annotation_dry_run_video_count": 0,
            "fully_annotated_video_count": 0,
            "data_collection_status": "INCOMPLETE",
            "annotation_schema_status": "UNFROZEN",
        },
        "evaluation": {
            "v1_method_head": "bc9ca6370962df108bd44d4a50cbdf2132234e8a",
            "evaluation_protocol_head": "e3d66acea82033e7d1875b87cf2504e928a43d84",
            "development_video_ids": ["p1b_pixabay_bottle"],
        },
        "videos": [_video()],
    }


def _annotation() -> dict[str, object]:
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotator_metadata": {"annotator_id": "annotator-local-1"},
        "video": {
            "video_id": "eval:video:001",
            "source_sha256": "a" * 64,
            "duration_seconds": 12.0,
            "fps": 30.0,
            "frame_count": 360,
            "width": 640,
            "height": 360,
        },
        "objects": [
            {
                "object_instance_id": "gt:object:0001",
                "object_class": "cup",
                "description": "white cup",
                "observation_status": "OBSERVED",
                "first_observed_frame": 10,
                "first_observed_time": 10 / 30,
                "last_observed_frame": 240,
                "last_observed_time": 240 / 30,
                "states": [
                    {
                        "state_id": "gt-state-1",
                        "zone": "left_region",
                        "start_frame": 10,
                        "end_frame": 100,
                        "start_time": 10 / 30,
                        "end_time": 100 / 30,
                        "evidence_frames": [10, 100],
                        "ambiguity_status": "CLEAR",
                        "notes": "observed before relocation",
                    },
                    {
                        "state_id": "gt-state-2",
                        "zone": "right_region",
                        "start_frame": 110,
                        "end_frame": 240,
                        "start_time": 110 / 30,
                        "end_time": 240 / 30,
                        "evidence_frames": [110, 240],
                        "ambiguity_status": "CLEAR",
                        "notes": "observed after relocation",
                    },
                ],
                "movement_events": [
                    {
                        "event_id": "gt:event:move:0001",
                        "event_type": "MOVE_OBJECT",
                        "object_instance_id": "gt:object:0001",
                        "from_state_id": "gt-state-1",
                        "to_state_id": "gt-state-2",
                        "temporal_status": "BOUNDED_INTERVAL",
                        "start_frame": 100,
                        "end_frame": 110,
                        "start_time": 100 / 30,
                        "end_time": 110 / 30,
                        "evidence_frames": [100, 110],
                        "movement_support_status": "SUPPORTED",
                        "agent_instance_ids": ["gt:person:0001"],
                        "agent_attribution_status": "SUPPORTED",
                    }
                ],
            }
        ],
        "agent_instances": [
            {
                "agent_instance_id": "gt:person:0001",
                "agent_attribution_status": "SUPPORTED",
            }
        ],
        "interaction_negatives": [],
        "queries": [
            {
                "query_id": "gt-query-0001",
                "video_id": "eval:video:001",
                "query_type": "WHERE_OBJECT",
                "object_instance_id": "gt:object:0001",
                "target_reference_status": "KNOWN_OBJECT",
                "natural_language_question": "Where is the cup?",
                "reference_event_id": None,
                "answerability": "ANSWERABLE",
                "expected_status": "ANSWER",
                "ground_truth": {
                    "state_id": "gt-state-2",
                    "zone": "right_region",
                    "evidence_frames": [240],
                },
                "evidence_refs": [{"frame": 240, "note": "last observed state"}],
                "hard_negative_tags": [],
                "annotation_notes": "controlled template",
            }
        ],
        "ambiguity_notes": [],
    }


def test_valid_manifest_and_annotation_contracts_are_accepted() -> None:
    validate_manifest(_manifest())
    validate_annotation(_annotation(), _manifest())


def test_committed_d0_manifest_has_six_eval_videos_and_excludes_dev_video() -> None:
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "data/eval/manifest.json").read_text())

    validate_manifest(manifest)

    assert manifest["pilot"]["selected_eval_video_count"] == 6
    assert sum(video["split"] == "EVAL" for video in manifest["videos"]) == 6
    dev_record = next(video for video in manifest["videos"] if video["video_id"] == "p1b_pixabay_bottle")
    assert dev_record["split"] == "DEV"


def test_duplicate_video_id_is_rejected() -> None:
    value = _manifest()
    value["videos"] = [_video(), _video()]

    with pytest.raises(EvaluationValidationError, match="duplicate video_id"):
        validate_manifest(value)


def test_missing_video_sha_is_rejected() -> None:
    value = _manifest()
    del value["videos"][0]["sha256"]  # type: ignore[index]

    with pytest.raises(EvaluationValidationError, match="sha256"):
        validate_manifest(value)


def test_formal_eval_video_requires_reproducible_source_uri() -> None:
    value = _manifest()
    value["videos"][0]["source_provenance"] = {"source_note": "not reproducible"}  # type: ignore[index]

    with pytest.raises(EvaluationValidationError, match="source_url or source_uri"):
        validate_manifest(value)


def test_development_video_cannot_be_selected_for_eval_metrics() -> None:
    value = _manifest()
    value["videos"] = [_video(video_id="p1b_pixabay_bottle", split="DEV")]
    value["pilot"]["selected_eval_video_count"] = 0  # type: ignore[index]

    validate_manifest(value)
    with pytest.raises(EvaluationValidationError, match="DEV_ONLY"):
        validate_eval_video_selection(value, ["p1b_pixabay_bottle"])


def test_annotation_schema_version_is_checked() -> None:
    value = _annotation()
    value["schema_version"] = "evaluation_annotation_v0"

    with pytest.raises(EvaluationValidationError, match="schema_version"):
        validate_annotation(value, _manifest())


def test_annotation_validation_does_not_mutate_input() -> None:
    value = _annotation()
    original = deepcopy(value)

    validate_annotation(value, _manifest())

    assert value == original


def test_invalid_frame_range_is_rejected() -> None:
    value = _annotation()
    value["objects"][0]["states"][0]["end_frame"] = 9  # type: ignore[index]

    with pytest.raises(EvaluationValidationError, match="frame range is reversed"):
        validate_annotation(value, _manifest())


def test_unresolved_object_reference_is_rejected() -> None:
    value = _annotation()
    value["queries"][0]["object_instance_id"] = "gt:object:missing"  # type: ignore[index]

    with pytest.raises(EvaluationValidationError, match="unresolved object reference"):
        validate_annotation(value, _manifest())


def test_unresolved_event_reference_is_rejected() -> None:
    value = _annotation()
    value["queries"][0]["reference_event_id"] = "gt:event:missing"  # type: ignore[index]

    with pytest.raises(EvaluationValidationError, match="unresolved event reference"):
        validate_annotation(value, _manifest())


def test_unresolved_state_reference_is_rejected() -> None:
    value = _annotation()
    event = value["objects"][0]["movement_events"][0]  # type: ignore[index]
    event["from_state_id"] = "gt-state:missing"

    with pytest.raises(EvaluationValidationError, match="state reference is unresolved"):
        validate_annotation(value, _manifest())


def test_unresolved_person_reference_is_rejected() -> None:
    value = _annotation()
    event = value["objects"][0]["movement_events"][0]  # type: ignore[index]
    event["agent_instance_ids"] = ["gt:person:missing"]

    with pytest.raises(EvaluationValidationError, match="unresolved agent reference"):
        validate_annotation(value, _manifest())


def test_no_visible_agent_cannot_contain_an_attributed_person() -> None:
    value = _annotation()
    event = value["objects"][0]["movement_events"][0]  # type: ignore[index]
    event["agent_attribution_status"] = "NO_VISIBLE_AGENT"

    with pytest.raises(EvaluationValidationError, match="cannot contain agent IDs"):
        validate_annotation(value, _manifest())


def test_same_zone_movement_is_valid_ground_truth() -> None:
    value = _annotation()
    value["objects"][0]["states"][1]["zone"] = "left_region"  # type: ignore[index]
    value["queries"][0]["hard_negative_tags"] = ["SAME_ZONE_MOVEMENT"]  # type: ignore[index]

    validate_annotation(value, _manifest())


def test_stationary_interaction_without_movement_is_representable() -> None:
    value = _annotation()
    value["interaction_negatives"] = [
        {
            "interaction_id": "gt:interaction:0001",
            "object_instance_id": "gt:object:0001",
            "person_instance_ids": ["gt:person:0001"],
            "start_frame": 30,
            "end_frame": 40,
            "start_time": 30 / 30,
            "end_time": 40 / 30,
            "movement_present": False,
            "hard_negative_tags": ["STATIONARY_INTERACTION"],
            "notes": "person touches the cup without displacement",
        }
    ]

    validate_annotation(value, _manifest())


def test_multiple_event_ambiguity_and_who_abstention_are_representable() -> None:
    value = _annotation()
    first_event = value["objects"][0]["movement_events"][0]  # type: ignore[index]
    second_event = deepcopy(first_event)
    second_event["event_id"] = "gt:event:move:0002"
    value["objects"][0]["movement_events"].append(second_event)  # type: ignore[index]
    value["queries"].extend(  # type: ignore[union-attr]
        [
            {
                "query_id": "gt-query-ambiguous",
                "video_id": "eval:video:001",
                "query_type": "WHEN_OBJECT_MOVED",
                "object_instance_id": "gt:object:0001",
                "target_reference_status": "KNOWN_OBJECT",
                "natural_language_question": "When did the cup move?",
                "reference_event_id": None,
                "answerability": "AMBIGUOUS",
                "expected_status": "ABSTAIN",
                "ground_truth": {"reason": "MULTIPLE_EVENT_REFERENCE"},
                "evidence_refs": [],
                "hard_negative_tags": ["MULTIPLE_MOVE_EVENTS"],
                "annotation_notes": "no event reference was supplied",
            },
            {
                "query_id": "gt-query-who-abstain",
                "video_id": "eval:video:001",
                "query_type": "WHO_MOVED_OBJECT",
                "object_instance_id": "gt:object:0001",
                "target_reference_status": "KNOWN_OBJECT",
                "natural_language_question": "Who moved the cup?",
                "reference_event_id": "gt:event:move:0001",
                "answerability": "AMBIGUOUS",
                "expected_status": "ABSTAIN",
                "ground_truth": {"reason": "AMBIGUOUS_AGENT"},
                "evidence_refs": [],
                "hard_negative_tags": ["AMBIGUOUS_AGENT"],
                "annotation_notes": "agent evidence is ambiguous",
            },
        ]
    )

    validate_annotation(value, _manifest())
