"""Contract and deterministic comparison tests for EVAL-D0-S1-R1."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from recallvision.evaluation.second_review import (
    SecondReviewValidationError,
    compare_second_review,
    match_review_events,
    validate_second_review,
)


ROOT = Path(__file__).parents[1]


def _manifest() -> dict[str, object]:
    return json.loads((ROOT / "data/eval/manifest.json").read_text())


def _primary(video_id: str = "eval_d0_002") -> dict[str, object]:
    path = ROOT / "data/eval/manifest.json"
    manifest = json.loads(path.read_text())
    record = next(video for video in manifest["videos"] if video["video_id"] == video_id)
    return json.loads((ROOT / record["annotation_path"]).read_text())


def _target_catalog() -> dict[str, list[dict[str, str]]]:
    return {
        "eval_d0_001": [
            {
                "target_object_id": "gt:object:001:duffel_bag",
                "object_description": "Large dark duffel bag visible in the white table scene.",
            },
            {
                "target_object_id": "gt:object:001:television",
                "object_description": "Black television/monitor immediately to the left of the white table.",
            },
        ],
        "eval_d0_002": [
            {
                "target_object_id": "gt:object:002:cardboard_box",
                "object_description": "Open cardboard shipping box visible in the foreground table scene.",
            }
        ],
    }


def _review(*, video_id: str = "eval_d0_002") -> dict[str, object]:
    return {
        "review_schema_version": "eval_d0_second_review_v1",
        "reviewer_id_or_alias": "reviewer-alpha",
        "review_date": "2026-10-04",
        "video_id": video_id,
        "video_metadata": {
            "raw_video_path": "data/eval/raw/cand_mixkit_31218.mp4",
            "source_sha256": "cecdf0a12fbc1853a479751b91393ffaa97fa60798b4f0071746d9ce9ae58d5d",
            "duration_seconds": 18.208333,
            "fps": 24.0,
            "frame_count": 437,
            "width": 1280,
            "height": 720,
        },
        "blindness_attestation": {
            "target_list_predefined": True,
            "primary_event_labels_visible_to_reviewer": False,
            "primary_agent_labels_visible_to_reviewer": False,
            "system_predictions_visible_to_reviewer": False,
        },
        "target_reviews": [
            {
                "target_object_id": "gt:object:002:cardboard_box",
                "object_description": "Open cardboard shipping box visible in the foreground table scene.",
                "last_observed_state": {
                    "zone_or_location": "right_region",
                    "state_description": "carried toward the right side",
                    "ambiguity_note": "The last visible frame is the final supported observation.",
                },
                "last_observed_frame_or_time": {"frame": 423, "time_seconds": 17.625},
                "movement_events": [
                    {
                        "review_event_id": "review:event:0001",
                        "evidence_start_frame": 384,
                        "evidence_end_frame": 408,
                        "evidence_start_time_seconds": 16.0,
                        "evidence_end_time_seconds": 17.0,
                        "from_state": {
                            "zone_or_location": "center_region",
                            "state_description": "on the foreground table",
                        },
                        "to_state": {
                            "zone_or_location": "right_region",
                            "state_description": "carried toward the right side",
                        },
                        "agent_status": "SUPPORTED_SINGLE_AGENT",
                        "review_person_ids": ["review:person:0001"],
                        "confidence_or_ambiguity_note": "The visible person carries the box continuously.",
                    }
                ],
                "stationary_interactions": [
                    {
                        "review_interaction_id": "review:interaction:0001",
                        "evidence_start_frame": 0,
                        "evidence_end_frame": 383,
                        "evidence_start_time_seconds": 0.0,
                        "evidence_end_time_seconds": 15.958333333333334,
                        "review_person_ids": ["review:person:0001"],
                        "interaction_description": (
                            "The visible person packs contents and handles the box flaps/tape "
                            "while the box remains on the foreground table."
                        ),
                        "confidence_or_ambiguity_note": (
                            "The box remains approximately stationary during the bounded packing interaction."
                        ),
                    }
                ],
                "notes": "Reviewed from the raw continuous sequence only.",
            }
        ],
        "review_notes": "No system outputs or primary semantic labels were shown during the blind pass.",
    }


def test_valid_independent_review_is_accepted() -> None:
    validate_second_review(_review(), _manifest(), _target_catalog())


def test_second_review_target_catalog_matches_current_primary_annotations() -> None:
    catalog = _target_catalog()

    for video_id in ("eval_d0_001", "eval_d0_002"):
        primary_target_ids = {
            target["object_instance_id"] for target in _primary(video_id)["objects"]
        }
        review_target_ids = {
            target["target_object_id"] for target in catalog[video_id]
        }

        assert review_target_ids == primary_target_ids


def test_default_d002_parity_fixture_uses_corrected_last_observation() -> None:
    target_review = _review()["target_reviews"][0]  # type: ignore[index]

    assert target_review["last_observed_frame_or_time"] == {"frame": 423, "time_seconds": 17.625}
    assert target_review["last_observed_state"]["zone_or_location"] == "right_region"


def test_invalid_video_id_is_rejected() -> None:
    value = _review(video_id="eval_d0_003")

    with pytest.raises(SecondReviewValidationError, match="video_id"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_invalid_target_id_is_rejected() -> None:
    value = _review()
    value["target_reviews"][0]["target_object_id"] = "gt:object:missing"  # type: ignore[index]

    with pytest.raises(SecondReviewValidationError, match="target_object_id"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_invalid_frame_interval_is_rejected() -> None:
    value = _review()
    value["target_reviews"][0]["movement_events"][0]["evidence_end_frame"] = 383  # type: ignore[index]

    with pytest.raises(SecondReviewValidationError, match="interval"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_invalid_agent_status_is_rejected() -> None:
    value = _review()
    value["target_reviews"][0]["movement_events"][0]["agent_status"] = "SUPPORTED"  # type: ignore[index]

    with pytest.raises(SecondReviewValidationError, match="agent_status"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_supported_agent_without_person_id_is_rejected() -> None:
    value = _review()
    value["target_reviews"][0]["movement_events"][0]["review_person_ids"] = []  # type: ignore[index]

    with pytest.raises(SecondReviewValidationError, match="requires"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_no_visible_agent_with_person_id_is_rejected() -> None:
    value = _review()
    event = value["target_reviews"][0]["movement_events"][0]  # type: ignore[index]
    event["agent_status"] = "NO_VISIBLE_AGENT"

    with pytest.raises(SecondReviewValidationError, match="cannot contain"):
        validate_second_review(value, _manifest(), _target_catalog())


@pytest.mark.parametrize("field", ["duration_seconds", "fps", "width", "height"])
def test_video_metadata_must_match_manifest(field: str) -> None:
    value = _review()
    value["video_metadata"][field] = value["video_metadata"][field] * 2  # type: ignore[index]

    with pytest.raises(SecondReviewValidationError, match="does not match manifest"):
        validate_second_review(value, _manifest(), _target_catalog())


def test_event_matching_uses_target_and_temporal_iou_not_review_event_id() -> None:
    primary = _primary()
    secondary = _review()
    secondary_event = secondary["target_reviews"][0]["movement_events"][0]  # type: ignore[index]
    secondary_event["review_event_id"] = "review:event:0099"
    secondary_event["evidence_start_frame"] = 385
    secondary_event["evidence_end_frame"] = 409
    secondary_event["evidence_start_time_seconds"] = 385 / 24
    secondary_event["evidence_end_time_seconds"] = 409 / 24

    result = match_review_events(primary, secondary, fps=24.0)

    assert result["matched_count"] == 1
    assert result["matches"][0]["primary_event_id"] == "gt:event:002:box:carry"
    assert result["matches"][0]["review_event_id"] == "review:event:0099"
    assert 0.9 < result["matches"][0]["temporal_iou"] < 1.0


def test_unmatched_primary_and_secondary_events_are_reported() -> None:
    primary = _primary()
    secondary = _review()
    events = secondary["target_reviews"][0]["movement_events"]  # type: ignore[index]
    events[0]["evidence_start_frame"] = 0
    events[0]["evidence_end_frame"] = 1
    events[0]["evidence_start_time_seconds"] = 0.0
    events[0]["evidence_end_time_seconds"] = 1 / 24

    result = match_review_events(primary, secondary, fps=24.0)

    assert result["matched_count"] == 0
    assert result["unmatched_primary_event_ids"] == ["gt:event:002:box:carry"]
    assert result["unmatched_review_event_ids"] == ["review:event:0001"]


def test_comparison_reports_primary_stationary_interaction_missing_from_review() -> None:
    primary = _primary()
    secondary = _review()
    secondary["target_reviews"][0]["stationary_interactions"] = []  # type: ignore[index]

    result = compare_second_review(primary, secondary, fps=24.0)

    assert result["stationary_interaction_agreement"] == {"agree": 0, "total": 1, "display": "0 / 1"}
    assert result["disagreements"] == [
        {
            "video_id": "eval_d0_002",
            "target_object": "gt:object:002:cardboard_box",
            "variable": "STATIONARY_INTERACTION",
            "primary_value": {
                "target_object_id": "gt:object:002:cardboard_box",
                "start_frame": 0,
                "end_frame": 383,
                "id": "gt:interaction:002:box:packing",
            },
            "secondary_value": None,
            "raw_evidence_reference": {"primary_interaction_id": "gt:interaction:002:box:packing"},
            "disagreement_type": "STATIONARY_INTERACTION",
        }
    ]


def test_comparison_output_is_deterministic_and_reports_raw_case_fields() -> None:
    primary = _primary()
    secondary = _review()

    first = compare_second_review(primary, secondary, fps=24.0)
    second = compare_second_review(deepcopy(primary), deepcopy(secondary), fps=24.0)

    assert first == second
    assert first["target_last_state_agreement"] == {"agree": 1, "total": 1, "display": "1 / 1"}
    assert first["primary_movement_event_count"] == 1
    assert first["secondary_movement_event_count"] == 1
    assert first["matched_event_count"] == 1
    assert first["agent_status_agreement"] == {"agree": 1, "total": 1, "display": "1 / 1"}
    assert first["supported_agent_identity_agreement"]["display"] == "NOT_COMPARABLE (no correspondence supplied)"
    assert first["disagreements"] == []
