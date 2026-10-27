from __future__ import annotations

from pathlib import Path

from recallvision.video_ai.product_logic import (
    build_visibility_probe_times,
    build_event_probe_times,
    build_adaptive_visibility_probe_times,
    classify_event_claim,
    extract_visibility_observations,
    resolve_event_disagreement,
    resolve_visibility_observations,
    route_question,
    visibility_target,
)
from recallvision.video_ai.video_cache import VideoAnalysisCache


def test_natural_questions_route_to_specialized_evidence_paths() -> None:
    assert route_question("Where did I leave my backpack?") == "last_seen"
    assert route_question("Where was the bicycle at the end?") == "qwen_final_frame"
    assert visibility_target("Where was the bicycle at the end?") == "bicycle"
    assert route_question("When did the person pick up the tablet?") == "when"
    assert route_question("Did the cup actually move?") == "when"
    assert route_question("What happened to the plant?") == "qwen"
    assert route_question("Where was the notebook before it disappeared?") == "last_seen"
    assert visibility_target("Where was the notebook before it disappeared?") == "notebook"
    assert route_question("Where was the backpack last?") == "last_seen"
    assert visibility_target("Where was the backpack last?") == "backpack"


def test_visibility_probes_extend_past_truncated_temporal_interval() -> None:
    probes = build_visibility_probe_times(
        duration=10.9,
        fps=1.0,
        intervals=[[0.0, 4.0]],
        max_frames=8,
    )
    assert probes == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.9]


def test_visibility_probes_can_include_direct_model_candidates_when_grounding_is_empty() -> None:
    probes = build_visibility_probe_times(
        duration=10.0, fps=1.0, intervals=[], extra_times=[2.0, 8.0], max_frames=8
    )
    assert probes == [2.0, 8.0, 9.0]


def test_adaptive_visibility_probes_fill_large_unknown_gap_after_latest_supported_sighting() -> None:
    reviews = [
        {"verification_status": "corroborated", "visibility_state": "present",
         "visual_evidence": [{"timestamp_seconds": 3.128}]},
        {"verification_status": "unknown", "visibility_state": "unknown",
         "visual_evidence": [{"timestamp_seconds": 4.671}]},
        {"verification_status": "unknown", "visibility_state": "unknown",
         "visual_evidence": [{"timestamp_seconds": 6.215}]},
        {"verification_status": "unknown", "visibility_state": "unknown",
         "visual_evidence": [{"timestamp_seconds": 7.758}]},
    ]

    probes = build_adaptive_visibility_probe_times(
        reviews, duration=10.928, fps=29.97, max_extra_frames=3
    )

    assert probes == [3.9, 4.98, 5.289]


def test_adaptive_visibility_probes_do_not_invent_a_positive_anchor_or_repeat_checks() -> None:
    all_unknown = [
        {"verification_status": "unknown", "visibility_state": "unknown",
         "visual_evidence": [{"timestamp_seconds": value}]}
        for value in (0.0, 1.5, 3.0)
    ]
    assert build_adaptive_visibility_probe_times(
        all_unknown, duration=4.0, fps=30.0, max_extra_frames=3
    ) == []

    reviews = [
        {"verification_status": "corroborated", "visibility_state": "present",
         "visual_evidence": [{"timestamp_seconds": 3.0}]},
        {"verification_status": "corroborated", "visibility_state": "present",
         "visual_evidence": [{"timestamp_seconds": 3.75}]},
        {"verification_status": "unknown", "visibility_state": "unknown",
         "visual_evidence": [{"timestamp_seconds": 4.5}]},
    ]
    assert build_adaptive_visibility_probe_times(
        reviews, duration=5.0, fps=30.0, max_extra_frames=3
    ) == []


def test_event_recheck_probes_compare_candidate_endpoints_and_middle() -> None:
    assert build_event_probe_times(10.0, [[0.0, 10.0]], [0.0, 10.0], max_frames=5) == [0.0, 2.25, 4.5, 6.75, 9.0]


def test_event_claim_parser_distinguishes_explicit_denial_from_uncertainty() -> None:
    assert classify_event_claim("The cup stayed in the same place and did not move.") == "stationary"
    assert classify_event_claim("The cup moved across the desk.") == "moved"
    assert classify_event_claim("The person picks up the tablet.") == "moved"
    assert classify_event_claim("The cup is partly hidden.") == "uncertain"


def test_last_seen_uses_last_confirmed_frame_and_keeps_world_location_unknown() -> None:
    result = resolve_visibility_observations(
        [
            {"seconds": 4.0, "visibility": "present", "location": "on the desk"},
            {"seconds": 6.0, "visibility": "present", "location": "on the desk"},
            {"seconds": 7.0, "visibility": "absent"},
            {"seconds": 10.0, "visibility": "absent"},
        ]
    )
    assert result["status"] == "confirmed_last_seen"
    assert result["last_seen_seconds"] == 6.0
    assert result["last_seen_location"] == "on the desk"
    assert result["real_world_location"] == "unknown"


def test_last_seen_abstains_when_no_probe_confirms_visibility() -> None:
    result = resolve_visibility_observations(
        [{"seconds": 8.0, "visibility": "uncertain"}, {"seconds": 10.0, "visibility": "absent"}]
    )
    assert result["status"] == "unconfirmed"
    assert result["last_seen_seconds"] is None


def test_visibility_parser_accepts_qwen_array_and_object_responses() -> None:
    array = '```json\n[{"seconds":"6.006","visibility":"present","location":"on the desk"}]\n```'
    wrapped = '{"observations":[{"seconds":7,"visibility":"absent"}]}'
    assert extract_visibility_observations(array) == [{"seconds": "6.006", "visibility": "present", "location": "on the desk"}]
    assert extract_visibility_observations(wrapped) == [{"seconds": 7, "visibility": "absent"}]
    singleton = '{"seconds":5.005,"visibility":"present","location":"on the desk"}'
    assert extract_visibility_observations(singleton) == [{"seconds": 5.005, "visibility": "present", "location": "on the desk"}]
    keyed = '{"6.006":{"visibility":"absent","location":"","confidence":"high"}}'
    assert extract_visibility_observations(keyed) == [{"visibility": "absent", "location": "", "confidence": "high", "seconds": "6.006"}]


def test_visibility_does_not_claim_complete_last_seen_when_later_probe_is_missing() -> None:
    result = resolve_visibility_observations(
        [{"seconds": 5.0, "visibility": "present", "location": "on the desk"}],
        expected_times=[2.0, 5.0, 10.0],
    )
    assert result["status"] == "incomplete"
    assert result["last_seen_seconds"] == 5.0
    assert result["missing_seconds"] == [2.0, 10.0]


def test_visibility_marks_conflicting_checks_at_the_same_timestamp() -> None:
    result = resolve_visibility_observations(
        [
            {"seconds": 6.0, "visibility": "present", "location": "on the desk"},
            {"seconds": 10.0, "visibility": "present"},
            {"seconds": 10.0, "visibility": "absent"},
        ],
        expected_times=[6.0, 10.0],
    )
    assert result["status"] == "conflicted"
    assert result["last_seen_seconds"] == 6.0
    assert result["conflict_seconds"] == [10.0]


def test_targeted_recheck_can_overrule_empty_timelens_with_camera_motion_evidence() -> None:
    result = resolve_event_disagreement(
        qwen_claim="moved",
        has_temporal_candidate=False,
        recheck={
            "event_status": "stationary",
            "confidence": "high",
            "evidence_seconds": [0.0, 10.0],
            "camera_motion": True,
        },
    )
    assert result["status"] == "resolved_stationary"
    assert result["answer_supported"] is True
    assert result["camera_motion"] is True


def test_event_disagreement_abstains_when_targeted_recheck_is_uncertain() -> None:
    result = resolve_event_disagreement(
        qwen_claim="moved",
        has_temporal_candidate=False,
        recheck={"event_status": "uncertain", "confidence": "low", "evidence_seconds": [4.0]},
    )
    assert result["status"] == "unresolved"
    assert result["answer_supported"] is False


def test_event_recheck_parses_string_evidence_without_coercing_unknown_camera_state() -> None:
    result = resolve_event_disagreement(
        qwen_claim="moved",
        has_temporal_candidate=False,
        recheck={
            "event_status": "no movement",
            "confidence": "high",
            "evidence_seconds": "[0.0, 10.0]",
            "camera_motion": "unknown",
        },
    )
    assert result["status"] == "resolved_stationary"
    assert result["camera_motion"] is None
    assert result["evidence_seconds"] == [0.0, 10.0]


def test_video_analysis_cache_is_content_and_model_specific(tmp_path: Path) -> None:
    cache = VideoAnalysisCache(tmp_path)
    cache.put_analysis("abc123", "qwen", "What moved?", "qwen-2b-r1", {"answer": "cup"})
    assert cache.get_analysis("abc123", "qwen", "What moved?", "qwen-2b-r1") == {"answer": "cup"}
    assert cache.get_analysis("abc123", "qwen", "What moved?", "qwen-2b-r2") is None
    assert cache.get_analysis("abc123", "timelens", "What moved?", "qwen-2b-r1") is None


def test_video_cache_lists_only_saved_analysis_for_the_requested_video(tmp_path: Path) -> None:
    cache = VideoAnalysisCache(tmp_path)
    cache.put_analysis("video-a", "timelens", "when cup moved", "time-r1", {"raw_response": "[]"})
    cache.put_analysis("video-b", "qwen", "what happened", "qwen-r1", {"raw_response": "{}"})
    records = cache.list_analyses("video-a")
    assert len(records) == 1
    assert records[0]["route"] == "timelens"
    assert records[0]["result"] == {"raw_response": "[]"}


def test_cached_source_frame_is_reused_without_decoding_again(tmp_path: Path) -> None:
    cache = VideoAnalysisCache(tmp_path)
    calls: list[int] = []

    def decode() -> bytes:
        calls.append(1)
        return b"jpeg-frame-bytes"

    first = cache.get_or_create_frame("video-hash", 12, decode)
    second = cache.get_or_create_frame("video-hash", 12, decode)
    assert first == second == b"jpeg-frame-bytes"
    assert calls == [1]


def test_batch_frame_cache_decodes_only_missing_indices_once(tmp_path: Path) -> None:
    cache = VideoAnalysisCache(tmp_path)
    first_calls: list[list[int]] = []

    def decode_first(indices: list[int]) -> dict[int, bytes]:
        first_calls.append(indices)
        return {index: f"jpeg-{index}".encode() for index in indices}

    first = cache.get_or_create_frames("video-hash", [1, 3, 5], decode_first)
    assert first == {1: b"jpeg-1", 3: b"jpeg-3", 5: b"jpeg-5"}
    assert first_calls == [[1, 3, 5]]

    second_calls: list[list[int]] = []

    def decode_second(indices: list[int]) -> dict[int, bytes]:
        second_calls.append(indices)
        return {index: f"jpeg-{index}".encode() for index in indices}

    restored = cache.get_or_create_frames("video-hash", [1, 3, 5, 7], decode_second)
    assert restored[7] == b"jpeg-7"
    assert restored[1] == b"jpeg-1"
    assert second_calls == [[7]]
