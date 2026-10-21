"""A deliberately narrow parser for supported object queries."""

from __future__ import annotations

from recallvision.query.schema import AnswerStatus, Query, QueryType


class QueryParseError(ValueError):
    """Raised when a question is outside the P1A parser contract."""

    status = AnswerStatus.UNSUPPORTED_QUERY


def parse_query(question: str, *, video_id: str, question_id: str) -> Query:
    """Parse deterministic object-location, movement, and attribution questions."""

    stripped = question.strip()
    lowered = stripped.casefold()
    query_type: QueryType
    if lowered.startswith("where is ") and stripped.endswith("?"):
        query_type = QueryType.WHERE_OBJECT
        target = stripped[len("where is ") : -1].strip()
    elif lowered.startswith("who moved ") and stripped.endswith("?"):
        query_type = QueryType.WHO_MOVED_OBJECT
        target = stripped[len("who moved ") : -1].strip()
    elif lowered.startswith("when was ") and lowered.endswith(" moved?"):
        query_type = QueryType.WHEN_OBJECT_MOVED
        target = stripped[len("when was ") : -len(" moved?")].strip()
    elif lowered.startswith("when did ") and lowered.endswith(" move?"):
        query_type = QueryType.WHEN_OBJECT_MOVED
        target = stripped[len("when did ") : -len(" move?")].strip()
    else:
        raise QueryParseError(
            "UNSUPPORTED_QUERY: only WHERE_OBJECT, WHEN_OBJECT_MOVED, and WHO_MOVED_OBJECT questions are supported"
        )

    target_lower = target.casefold()
    if target_lower.startswith("the "):
        target = target[4:].strip()
    elif target_lower.startswith("object "):
        target = target[7:].strip()
    if not target or "?" in target:
        raise QueryParseError("UNSUPPORTED_QUERY: object target is missing")

    return Query(
        question_id=question_id,
        video_id=video_id,
        question=stripped,
        query_type=query_type,
        target_class=target.casefold(),
    )
