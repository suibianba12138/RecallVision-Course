"""Deterministic WHERE_OBJECT retrieval over structured observations."""

from __future__ import annotations

from collections.abc import Sequence

from recallvision.memory.schema import EvidenceRef, Observation
from recallvision.query.schema import Answer, AnswerStatus, Query, QueryType


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

    candidates = [
        observation
        for observation in observations
        if _is_reliable_candidate(query, observation, min_confidence)
    ]
    if not candidates:
        target_label = query.target_class or ", ".join(query.target_entity_ids) or "target"
        return Answer(
            question_id=query.question_id,
            answer=f"TARGET_NOT_FOUND: no reliable observation for {target_label}.",
            status=AnswerStatus.TARGET_NOT_FOUND,
            target_class=query.target_class,
        )

    selected = max(candidates, key=lambda observation: (observation.timestamp, observation.frame_index))
    target_class = query.target_class or selected.class_name
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
    )


def _is_reliable_candidate(
    query: Query,
    observation: Observation,
    min_confidence: float | None,
) -> bool:
    if observation.video_id != query.video_id:
        return False
    if observation.entity_type != "object":
        return False
    if observation.frame_index < 0 or observation.timestamp < 0.0:
        return False
    if not observation.class_name or not observation.zone:
        return False
    if query.target_class is not None and observation.class_name.casefold() != query.target_class.casefold():
        return False
    if query.target_entity_ids and observation.persistent_id not in query.target_entity_ids:
        return False
    if query.target_class is None and not query.target_entity_ids:
        return False
    if min_confidence is not None:
        if observation.confidence is None or observation.confidence < min_confidence:
            return False
    return True
