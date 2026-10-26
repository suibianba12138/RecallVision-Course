"""Deterministic WHERE_OBJECT retrieval over structured observations."""

from __future__ import annotations

from collections.abc import Sequence

from recallvision.memory.schema import EvidenceRef, Observation
from recallvision.query.schema import Answer, AnswerStatus, Query, QueryType
from recallvision.query.target_resolution import (
    TargetResolutionStatus,
    build_target_reference,
    candidates_from_observations,
    resolve_query_target,
    target_reference_matches_class,
)


def answer_where_object(
    query: Query,
    observations: Sequence[Observation],
    *,
    min_confidence: float | None = None,
) -> Answer:
    """Answer a supported object-location query from in-memory observations."""

    if query.query_type is not QueryType.WHERE_OBJECT:
        return Answer(
            question_id=query.question_id,
            answer=f"UNSUPPORTED_QUERY: {query.query_type.value}",
            status=AnswerStatus.UNSUPPORTED_QUERY,
            target_class=query.target_class,
        )

    # P1A's raw-observation contract predates persistent IDs. Preserve its
    # class-only behavior when no observation can participate in entity
    # binding; P2B/P2C/P2D paths always use persistent candidates.
    if not any(observation.persistent_id for observation in observations):
        legacy_candidates = [
            observation
            for observation in observations
            if _is_reliable_candidate(query, observation, min_confidence)
        ]
        if not legacy_candidates:
            target_label = query.target_class or ", ".join(query.target_entity_ids) or "target"
            return Answer(
                question_id=query.question_id,
                answer=f"TARGET_NOT_FOUND: no reliable observation for {target_label}.",
                status=AnswerStatus.TARGET_NOT_FOUND,
                target_class=query.target_class,
                target_resolution_status=TargetResolutionStatus.NO_COMPATIBLE_ENTITY.value,
            )
        selected = max(legacy_candidates, key=lambda observation: (observation.timestamp, observation.frame_index))
        target_class = query.target_class or selected.class_name
        assert selected.zone is not None
        evidence = EvidenceRef(
            video_id=selected.video_id,
            timestamp=selected.timestamp,
            frame_indices=[selected.frame_index],
            segment_ref=f"{selected.video_id}#frame={selected.frame_index}",
        )
        return Answer(
            question_id=query.question_id,
            answer=f"The {target_class} is in the {selected.zone.replace('_', ' ')}.",
            evidence=[evidence],
            status=AnswerStatus.OK,
            target_entity_id=selected.persistent_id or target_class,
            target_class=target_class,
            zone=selected.zone,
            timestamp=selected.timestamp,
            frame_index=selected.frame_index,
            target_resolution_status="LEGACY_RAW_OBSERVATION",
        )

    resolution = resolve_query_target(query, candidates_from_observations(observations))
    target_label = resolution.reference.canonical_label or ", ".join(query.target_entity_ids) or "target"
    if resolution.status is not TargetResolutionStatus.RESOLVED:
        status = (
            AnswerStatus.AMBIGUOUS_TARGET
            if resolution.status is TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES
            else AnswerStatus.TARGET_NOT_FOUND
        )
        return Answer(
            question_id=query.question_id,
            answer=(
                f"AMBIGUOUS_TARGET: multiple persistent entities match {target_label}."
                if status is AnswerStatus.AMBIGUOUS_TARGET
                else f"TARGET_NOT_FOUND: no reliable observation for {target_label}."
            ),
            status=status,
            target_class=target_label or query.target_class,
            target_resolution_status=resolution.status.value,
        )

    target_id = resolution.resolved_entity_id
    assert target_id is not None
    candidates = [
        observation
        for observation in observations
        if _is_reliable_candidate(
            query,
            observation,
            min_confidence,
            resolved_entity_id=target_id,
        )
    ]
    if not candidates:
        return Answer(
            question_id=query.question_id,
            answer=f"INSUFFICIENT_ENTITY_EVIDENCE: resolved target {target_id} has no reliable observation.",
            status=AnswerStatus.TARGET_NOT_FOUND,
            target_class=target_label or query.target_class,
            target_resolution_status=TargetResolutionStatus.INSUFFICIENT_ENTITY_EVIDENCE.value,
        )
    selected = max(candidates, key=lambda observation: (observation.timestamp, observation.frame_index))
    target_class = target_label or selected.class_name
    target_entity_id = selected.persistent_id or target_class
    assert selected.zone is not None
    evidence = EvidenceRef(
        video_id=selected.video_id,
        timestamp=selected.timestamp,
        frame_indices=[selected.frame_index],
        segment_ref=f"{selected.video_id}#frame={selected.frame_index}",
    )
    return Answer(
        question_id=query.question_id,
        answer=f"The {target_class} is in the {selected.zone.replace('_', ' ')}.",
        evidence=[evidence],
        status=AnswerStatus.OK,
        target_entity_id=target_entity_id,
        target_class=target_class,
        zone=selected.zone,
        timestamp=selected.timestamp,
        frame_index=selected.frame_index,
        target_resolution_status=resolution.status.value,
    )


def _is_reliable_candidate(
    query: Query,
    observation: Observation,
    min_confidence: float | None,
    *,
    resolved_entity_id: str | None = None,
) -> bool:
    if observation.video_id != query.video_id:
        return False
    if observation.entity_type != "object":
        return False
    if observation.frame_index < 0 or observation.timestamp < 0.0:
        return False
    if not observation.class_name or not observation.zone:
        return False
    if resolved_entity_id is not None and observation.persistent_id != resolved_entity_id:
        return False
    if resolved_entity_id is None and query.target_entity_ids and observation.persistent_id not in query.target_entity_ids:
        return False
    if resolved_entity_id is None and query.target_class is not None:
        reference = query.target_reference or build_target_reference(query.target_class)
        if not target_reference_matches_class(reference, observation.class_name):
            return False
    if query.target_class is None and not query.target_entity_ids:
        return False
    if min_confidence is not None:
        if observation.confidence is None or observation.confidence < min_confidence:
            return False
    return True
