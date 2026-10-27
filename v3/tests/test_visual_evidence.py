from __future__ import annotations

from recallvision.video_ai.visual_evidence import (
    combine_detector_crop_review,
    combine_detector_crop_reviews,
    filter_target_candidates,
    review_detector_frame,
)


def test_detector_miss_remains_unknown_not_absent() -> None:
    result = review_detector_frame(
        [], width=1280, height=720, frame_index=140, timestamp_seconds=5.84,
        frame_sha256="a" * 64,
    )
    assert result["verification_status"] == "unknown"
    assert result["visibility_state"] == "unknown"
    assert result["identity_status"] == "unverified"
    assert "miss cannot establish absence" in result["uncertainty_reason"].casefold()
    assert result["visual_evidence"][0]["timestamp_seconds"] == 5.84


def test_overwide_grounding_dino_box_is_not_accepted_as_target_presence() -> None:
    result = review_detector_frame([{
        "bbox_xyxy": [1.3, 1.4, 596.5, 717.4], "score": 0.899,
        "raw_model_label": "black backpack",
    }], width=1280, height=720, frame_index=0, timestamp_seconds=0.0,
       frame_sha256="b" * 64)
    assert result["verification_status"] == "unknown"
    assert result["visibility_state"] == "unknown"
    assert result["visual_evidence"][0]["detections"][0]["candidate_accepted"] is False
    assert "implausibly broad" in result["uncertainty_reason"].casefold()


def test_one_localized_box_supports_presence_only_at_that_frame() -> None:
    result = review_detector_frame([{
        "bbox_xyxy": [110.0, 90.0, 330.0, 410.0], "score": 0.83,
        "raw_model_label": "red backpack",
    }], width=640, height=480, frame_index=15, timestamp_seconds=3.0,
       frame_sha256="c" * 64)
    assert result["verification_status"] == "corroborated"
    assert result["visibility_state"] == "present"
    assert result["identity_status"] == "unverified"
    assert result["visual_evidence"][0]["detections"][0]["bbox_xyxy"] == [110.0, 90.0, 330.0, 410.0]
    assert result["visual_evidence"][0]["detections"][0]["candidate_accepted"] is True
    assert "exact frame" in result["uncertainty_reason"].casefold()


def test_target_crop_review_rejects_dino_lookalike_candidate_without_calling_absence() -> None:
    detector = review_detector_frame([{
        "bbox_xyxy": [997.0, 546.0, 1279.0, 718.0], "score": 0.46,
        "raw_model_label": "black backpack",
    }], width=1280, height=720, frame_index=134, timestamp_seconds=5.589,
       frame_sha256="d" * 64)
    combined = combine_detector_crop_review(detector, {
        "visibility": "absent", "confidence": "high", "model_fingerprint": "qwen-crop-v1",
    })
    assert combined["verification_status"] == "conflicted"
    assert combined["visibility_state"] == "unknown"
    assert combined["identity_status"] == "ambiguous"
    assert "not treated as absence" in combined["uncertainty_reason"].casefold()
    assert combined["visual_evidence"][0]["crop_review"]["visibility"] == "absent"


def test_target_crop_agreement_supports_only_the_exact_frame_not_identity_continuity() -> None:
    detector = review_detector_frame([{
        "bbox_xyxy": [1.7, 2.0, 270.0, 714.0], "score": 0.51,
        "raw_model_label": "black backpack",
    }], width=1280, height=720, frame_index=75, timestamp_seconds=3.128,
       frame_sha256="e" * 64)
    combined = combine_detector_crop_review(detector, {
        "visibility": "present", "location": "on the desk", "confidence": "high",
        "model_fingerprint": "qwen-crop-v1",
    })
    assert combined["verification_status"] == "corroborated"
    assert combined["visibility_state"] == "present"
    assert combined["identity_status"] == "unverified"
    assert combined["visual_evidence"][0]["detections"][0]["candidate_accepted"] is False
    assert combined["visual_evidence"][0]["detections"][0]["target_presence_corroborated"] is True
    assert "exact source frame" in combined["uncertainty_reason"].casefold()


def test_reference_crop_match_resolves_same_target_from_a_category_lookalike() -> None:
    detector = review_detector_frame([
        {"bbox_xyxy": [0.0, 180.0, 50.0, 475.0], "score": 0.22,
         "raw_model_label": "black backpack", "candidate_id": "119:0"},
        {"bbox_xyxy": [1000.0, 330.0, 1278.0, 715.0], "score": 0.21,
         "raw_model_label": "black backpack", "candidate_id": "119:1"},
    ], width=1280, height=720, frame_index=119, timestamp_seconds=4.963,
       frame_sha256="1" * 64, minimum_score=0.2)

    combined = combine_detector_crop_reviews(detector, {
        "119:0": {"visibility": "present", "reference_match": "same", "model_fingerprint": "qwen-pair-v1"},
        "119:1": {"visibility": "absent", "reference_match": "different", "model_fingerprint": "qwen-pair-v1"},
    })

    candidates = combined["visual_evidence"][0]["detections"]
    assert combined["verification_status"] == "corroborated"
    assert combined["visibility_state"] == "present"
    assert combined["identity_status"] == "associated"
    assert candidates[0]["target_presence_corroborated"] is True
    assert candidates[1]["target_presence_corroborated"] is False


def test_reference_identity_omission_keeps_candidate_unknown() -> None:
    detector = review_detector_frame([{
        "bbox_xyxy": [0.0, 180.0, 50.0, 475.0], "score": 0.22,
        "raw_model_label": "backpack", "candidate_id": "119:0",
    }], width=1280, height=720, frame_index=119, timestamp_seconds=4.963,
       frame_sha256="2" * 64, minimum_score=0.2)

    combined = combine_detector_crop_reviews(detector, {
        "119:0": {"visibility": "uncertain", "reference_match": "uncertain", "model_fingerprint": "qwen-pair-v1"},
    })

    assert combined["verification_status"] == "conflicted"
    assert combined["visibility_state"] == "unknown"
    assert combined["identity_status"] == "ambiguous"


def test_subpixel_box_overflow_is_clipped_for_crop_review() -> None:
    detector = review_detector_frame([{
        "bbox_xyxy": [-0.067, 0.0, 265.36, 717.85], "score": 0.438,
        "raw_model_label": "backpack",
    }], width=1280, height=720, frame_index=75, timestamp_seconds=3.128,
       frame_sha256="f" * 64)
    box = detector["visual_evidence"][0]["detections"][0]
    assert box["bbox_xyxy"] == [0.0, 0.0, 265.36, 717.85]
    assert box["bbox_xyxy_unclipped"][0] == -0.067
    combined = combine_detector_crop_review(detector, {"visibility": "present"})
    assert combined["visibility_state"] == "present"


def test_relaxed_target_candidates_drop_wrong_labels_and_suppress_nested_boxes() -> None:
    candidates = filter_target_candidates([
        {"bbox_xyxy": [1.0, 90.0, 54.0, 480.0], "score": 0.265,
         "raw_model_label": "black backpack"},
        {"bbox_xyxy": [1.0, 187.0, 45.0, 474.0], "score": 0.246,
         "raw_model_label": "black backpack"},
        {"bbox_xyxy": [1080.0, 547.0, 1277.0, 716.0], "score": 0.215,
         "raw_model_label": "black"},
    ], "my backpack")

    assert len(candidates) == 1
    assert candidates[0]["bbox_xyxy"] == [1.0, 90.0, 54.0, 480.0]


def test_candidate_id_crop_reviews_can_select_one_real_target_and_reject_a_lookalike() -> None:
    detector = review_detector_frame([
        {"bbox_xyxy": [10.0, 30.0, 100.0, 190.0], "score": 0.26,
         "raw_model_label": "black backpack", "candidate_id": "frame-1:0"},
        {"bbox_xyxy": [500.0, 80.0, 610.0, 220.0], "score": 0.37,
         "raw_model_label": "black backpack", "candidate_id": "frame-1:1"},
    ], width=640, height=480, frame_index=151, timestamp_seconds=5.035,
       frame_sha256="a" * 64, minimum_score=0.2)

    combined = combine_detector_crop_reviews(detector, {
        "frame-1:0": {"visibility": "present", "confidence": "medium"},
        "frame-1:1": {"visibility": "absent", "confidence": "high"},
    })

    assert combined["verification_status"] == "corroborated"
    assert combined["visibility_state"] == "present"
    assert combined["identity_status"] == "unverified"
    detections = combined["visual_evidence"][0]["detections"]
    assert detections[0]["target_presence_corroborated"] is True
    assert detections[1]["target_presence_corroborated"] is False


def test_multiple_distinct_crop_matches_remain_ambiguous() -> None:
    detector = review_detector_frame([
        {"bbox_xyxy": [10.0, 30.0, 100.0, 190.0], "score": 0.26,
         "raw_model_label": "black backpack", "candidate_id": "frame-1:0"},
        {"bbox_xyxy": [500.0, 80.0, 610.0, 220.0], "score": 0.37,
         "raw_model_label": "black backpack", "candidate_id": "frame-1:1"},
    ], width=640, height=480, frame_index=151, timestamp_seconds=5.035,
       frame_sha256="b" * 64, minimum_score=0.2)

    combined = combine_detector_crop_reviews(detector, {
        "frame-1:0": {"visibility": "present"},
        "frame-1:1": {"visibility": "present"},
    })

    assert combined["verification_status"] == "conflicted"
    assert combined["visibility_state"] == "unknown"
    assert combined["identity_status"] == "ambiguous"
