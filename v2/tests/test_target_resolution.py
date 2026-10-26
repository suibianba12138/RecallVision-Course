"""Generic natural-language target resolution contract tests."""

from __future__ import annotations

from recallvision.query.schema import Query, QueryType, TargetReference
from recallvision.query.target_resolution import (
    TargetCandidate,
    TargetResolutionStatus,
    build_target_reference,
    candidates_from_persistent_entities,
    resolve_query_target,
    resolve_target_reference,
)


VIDEO_ID = "fixture_video_001"


def candidate(entity_id: str, class_name: str, *, video_id: str = VIDEO_ID) -> TargetCandidate:
    return TargetCandidate(
        persistent_id=entity_id,
        class_name=class_name,
        video_id=video_id,
    )


def query(target_class: str, *, target_ids: list[str] | None = None) -> Query:
    reference = build_target_reference(target_class)
    return Query(
        question_id="target-resolution-query",
        video_id=VIDEO_ID,
        question=f"Where is the {target_class}?",
        query_type=QueryType.WHERE_OBJECT,
        target_entity_ids=target_ids or [],
        target_class=reference.canonical_label,
        target_reference=reference,
    )


def test_exact_class_match_resolves_one_persistent_candidate() -> None:
    result = resolve_target_reference(
        build_target_reference("bottle"),
        [candidate("obj:bottle:0001", "bottle")],
        video_id=VIDEO_ID,
    )

    assert result.status is TargetResolutionStatus.RESOLVED
    assert result.resolved_entity_id == "obj:bottle:0001"
    assert result.compatible_system_classes == ("bottle",)


def test_plural_target_normalizes_to_singular_class() -> None:
    reference = build_target_reference("bottles")

    assert reference.canonical_label == "bottle"
    result = resolve_target_reference(
        reference,
        [candidate("obj:bottle:0001", "bottle")],
        video_id=VIDEO_ID,
    )
    assert result.status is TargetResolutionStatus.RESOLVED


def test_plural_candidate_class_normalizes_to_singular_target() -> None:
    result = resolve_target_reference(
        build_target_reference("bottle"),
        [candidate("obj:bottle:0001", "bottles")],
        video_id=VIDEO_ID,
    )

    assert result.status is TargetResolutionStatus.RESOLVED


def test_global_detector_alias_matches_television_to_tv() -> None:
    reference = build_target_reference("television")
    result = resolve_target_reference(
        reference,
        [candidate("obj:tv:0001", "tv")],
        video_id=VIDEO_ID,
    )

    assert reference.canonical_label == "television"
    assert reference.semantic_aliases == ("tv",)
    assert result.status is TargetResolutionStatus.RESOLVED
    assert result.resolved_entity_id == "obj:tv:0001"


def test_unknown_closed_vocabulary_target_fails_as_unsupported() -> None:
    result = resolve_target_reference(
        build_target_reference("world map"),
        [],
        video_id=VIDEO_ID,
    )

    assert result.status is TargetResolutionStatus.DETECTOR_VOCABULARY_UNSUPPORTED
    assert result.resolved_entity_id is None


def test_supported_class_without_entity_fails_as_no_compatible_entity() -> None:
    result = resolve_target_reference(
        build_target_reference("bottle"),
        [],
        video_id=VIDEO_ID,
    )

    assert result.status is TargetResolutionStatus.NO_COMPATIBLE_ENTITY


def test_multiple_compatible_entities_fail_closed_as_ambiguous() -> None:
    result = resolve_target_reference(
        build_target_reference("television"),
        [candidate("obj:tv:0001", "tv"), candidate("obj:tv:0002", "tv")],
        video_id=VIDEO_ID,
    )

    assert result.status is TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES
    assert result.resolved_entity_id is None
    assert result.candidate_entity_ids == ("obj:tv:0001", "obj:tv:0002")


def test_explicit_entity_id_disambiguates_same_class_candidates() -> None:
    result = resolve_query_target(
        query("bottle", target_ids=["obj:bottle:0002"]),
        [candidate("obj:bottle:0001", "bottle"), candidate("obj:bottle:0002", "bottle")],
    )

    assert result.status is TargetResolutionStatus.RESOLVED
    assert result.resolved_entity_id == "obj:bottle:0002"


def test_persistent_entity_records_become_object_candidates_only() -> None:
    candidates = candidates_from_persistent_entities(
        [
            {"persistent_id": "obj:bottle:0001", "class_name": "bottle"},
            {"persistent_id": "obj:person:0001", "class_name": "person", "entity_type": "person"},
            {"persistent_id": "obj:tv:0001", "class_name": "tv"},
        ],
        video_id=VIDEO_ID,
    )

    assert [(item.persistent_id, item.class_name) for item in candidates] == [
        ("obj:bottle:0001", "bottle"),
        ("obj:tv:0001", "tv"),
    ]


def test_target_reference_is_explicit_and_does_not_contain_gt_identity() -> None:
    reference = TargetReference(
        raw_phrase="the bottle",
        canonical_label="bottle",
        semantic_aliases=(),
    )

    assert reference.raw_phrase == "the bottle"
    assert not hasattr(reference, "object_instance_id")
    assert not hasattr(reference, "evidence_frames")
