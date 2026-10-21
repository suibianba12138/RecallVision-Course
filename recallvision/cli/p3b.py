"""Run the artifact-only P3B WHERE_OBJECT_BEFORE query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recallvision.cli.p2c import _load_timelines
from recallvision.cli.p3a import _load_events, _sha256, _single_video_id, _write_json
from recallvision.query.object_before import answer_where_object_before
from recallvision.query.schema import AnswerStatus, Query, QueryType


def run(
    *,
    events_path: Path,
    timelines_path: Path,
    target_class: str | None,
    output_dir: Path,
    target_entity_id: str | None = None,
    reference_event_id: str | None = None,
    video_id: str | None = None,
    question_id: str = "p3b-query",
) -> dict[str, Any]:
    """Answer one WHERE_OBJECT_BEFORE query from frozen P2B/P2D artifacts."""

    timelines = _load_timelines(timelines_path)
    if not timelines:
        raise ValueError("P2B timeline artifact is empty")
    events = _load_events(events_path)
    selected_video_id = video_id or _single_video_id(sorted({item.video_id for item in timelines}))
    query = Query(
        question_id=question_id,
        video_id=selected_video_id,
        question=f"Where was the {target_class or target_entity_id or 'object'} before?",
        query_type=QueryType.WHERE_OBJECT_BEFORE,
        target_entity_ids=[target_entity_id] if target_entity_id is not None else [],
        target_class=target_class,
        reference_event_id=reference_event_id,
    )
    result = answer_where_object_before(query, timelines, events)
    record = result.to_record()
    record["input_artifacts"] = {
        "events": {"path": str(events_path), "sha256": _sha256(events_path)},
        "timelines": {"path": str(timelines_path), "sha256": _sha256(timelines_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "query_result.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--target-class")
    parser.add_argument("--target-entity-id")
    parser.add_argument("--reference-event-id")
    parser.add_argument("--video-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        events_path=arguments.events,
        timelines_path=arguments.timelines,
        target_class=arguments.target_class,
        target_entity_id=arguments.target_entity_id,
        reference_event_id=arguments.reference_event_id,
        output_dir=arguments.output_dir,
        video_id=arguments.video_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    supported = {
        AnswerStatus.OK.value,
        AnswerStatus.TARGET_NOT_FOUND.value,
        AnswerStatus.AMBIGUOUS_TARGET.value,
        AnswerStatus.NO_REFERENCE_EVENT.value,
        AnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE.value,
    }
    return 0 if result["status"] in supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
