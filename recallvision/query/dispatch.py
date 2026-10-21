"""Minimal query dispatch for the supported artifact-only query handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recallvision.events.schema import EventRecord
from recallvision.memory.state_timeline import ObjectStateTimeline
from recallvision.query.object_before import answer_where_object_before
from recallvision.query.object_movement import answer_when_object_moved
from recallvision.query.object_state_location import answer_where_object_state
from recallvision.query.schema import Query, QueryType
from recallvision.query.who_moved import (
    AuthorizedAttributedEvent,
    WhoMovedQueryResult,
    answer_who_moved_object,
)


def dispatch_query(
    query: Query,
    *,
    attributed_events: Sequence[AuthorizedAttributedEvent] = (),
    persistent_entities: Sequence[Mapping[str, Any]] = (),
    timelines: Sequence[ObjectStateTimeline] = (),
    events: Sequence[EventRecord] = (),
    video_end_timestamp: float | None = None,
) -> Any:
    """Route WHO_MOVED_OBJECT through the F3-authorized retrieval handler.

    Existing P3A/P3B routes remain available to avoid changing their contracts.
    """

    if query.query_type is QueryType.WHO_MOVED_OBJECT:
        return answer_who_moved_object(query, attributed_events, persistent_entities)
    if query.query_type is QueryType.WHERE_OBJECT:
        if video_end_timestamp is None:
            raise ValueError("WHERE_OBJECT dispatch requires video_end_timestamp")
        return answer_where_object_state(
            query,
            timelines,
            video_end_timestamp=video_end_timestamp,
        )
    if query.query_type is QueryType.WHEN_OBJECT_MOVED:
        return answer_when_object_moved(query, timelines, events)
    if query.query_type is QueryType.WHERE_OBJECT_BEFORE:
        return answer_where_object_before(query, timelines, events)
    raise ValueError(f"no artifact-only query handler is registered for {query.query_type.value}")


answer_query = dispatch_query
