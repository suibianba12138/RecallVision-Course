"""Tests for the deliberately narrow P1A query parser."""

import pytest

from recallvision.query.parser import QueryParseError, parse_query
from recallvision.query.schema import QueryType


def test_parser_supports_where_is_object_question() -> None:
    query = parse_query(
        "Where is the bottle?",
        video_id="fixture_video_001",
        question_id="question-1",
    )

    assert query.query_type is QueryType.WHERE_OBJECT
    assert query.target_class == "bottle"


def test_parser_rejects_unsupported_question_with_explicit_status() -> None:
    with pytest.raises(QueryParseError, match="UNSUPPORTED_QUERY"):
        parse_query(
            "How did the bottle move?",
            video_id="fixture_video_001",
            question_id="question-2",
        )


def test_parser_supports_when_object_moved_question() -> None:
    query = parse_query(
        "When was the bottle moved?",
        video_id="fixture_video_001",
        question_id="question-2",
    )

    assert query.query_type is QueryType.WHEN_OBJECT_MOVED
    assert query.target_class == "bottle"


def test_parser_supports_when_did_object_move_question() -> None:
    query = parse_query(
        "When did object bottle move?",
        video_id="fixture_video_001",
        question_id="question-3",
    )

    assert query.query_type is QueryType.WHEN_OBJECT_MOVED
    assert query.target_class == "bottle"


def test_parser_supports_who_moved_object_question() -> None:
    query = parse_query(
        "Who moved the bottle?",
        video_id="fixture_video_001",
        question_id="question-4",
    )

    assert query.query_type is QueryType.WHO_MOVED_OBJECT
    assert query.target_class == "bottle"


def test_parser_preserves_raw_target_reference_and_normalizes_detector_alias() -> None:
    query = parse_query(
        "Where is the television?",
        video_id="fixture_video_001",
        question_id="question-5",
    )

    assert query.target_reference is not None
    assert query.target_reference.raw_phrase == "television"
    assert query.target_reference.canonical_label == "television"
    assert query.target_reference.semantic_aliases == ("tv",)


def test_parser_supports_event_relative_before_target_reference() -> None:
    query = parse_query(
        "Where was the bottle before?",
        video_id="fixture_video_001",
        question_id="question-6",
    )

    assert query.query_type is QueryType.WHERE_OBJECT_BEFORE
    assert query.target_class == "bottle"
    assert query.target_reference is not None
