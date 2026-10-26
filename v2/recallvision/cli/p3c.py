"""Run the artifact-only P3C WHO_MOVED_OBJECT query."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from recallvision.query.dispatch import dispatch_query
from recallvision.query.parser import parse_query
from recallvision.query.schema import QueryType
from recallvision.query.who_moved import (
    WhoMovedAnswerStatus,
    load_authorized_event_memory,
    load_persistent_entities,
)


def run(
    *,
    f3_root: Path,
    persistent_entities_path: Path,
    question: str,
    video_id: str,
    output_dir: Path,
    target_class: str | None = None,
    target_entity_id: str | None = None,
    reference_event_id: str | None = None,
    question_id: str = "p3c-query",
) -> dict[str, Any]:
    """Answer one WHO_MOVED_OBJECT query from F3/P2A memory artifacts."""

    query = parse_query(question, video_id=video_id, question_id=question_id)
    if query.query_type is not QueryType.WHO_MOVED_OBJECT:
        raise ValueError("P3C CLI accepts only WHO_MOVED_OBJECT questions")
    query = replace(
        query,
        target_class=target_class or query.target_class,
        target_entity_ids=[target_entity_id] if target_entity_id is not None else query.target_entity_ids,
        reference_event_id=reference_event_id,
    )
    attributed_events = load_authorized_event_memory(f3_root)
    persistent_entities = load_persistent_entities(persistent_entities_path)
    result = dispatch_query(
        query,
        attributed_events=attributed_events,
        persistent_entities=persistent_entities,
    )
    record = result.to_record()
    record["query"] = {
        "question_id": query.question_id,
        "video_id": query.video_id,
        "question": query.question,
        "query_type": query.query_type.value,
        "target_entity_ids": list(query.target_entity_ids),
        "target_class": query.target_class,
        "reference_event_id": query.reference_event_id,
    }
    record["input_artifacts"] = {
        "f3_authorized_attributed_events_root": str(f3_root),
        "persistent_entities": {
            "path": str(persistent_entities_path),
            "sha256": _sha256_file(persistent_entities_path),
        },
    }
    record["query_contract"] = {
        "attributed_event_is_authoritative": True,
        "agent_neutral_p2d_used_for_person_answer": False,
        "f1_used_at_query_time": False,
        "f2_used_at_query_time": False,
        "f3_authorization_run_at_query_time": False,
        "vlm_used_at_query_time": False,
        "pose_used_at_query_time": False,
        "raw_video_used_at_query_time": False,
        "identity_reasoning_run_at_query_time": False,
        "brief_reason_parsed": False,
        "authorized_person_ids_read_directly": result.status is WhoMovedAnswerStatus.SUCCESS,
        "source_event_mutated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "who_moved_object.json"
    record["output_artifact"] = str(output_path)
    _write_json(output_path, record)
    # The checksum is returned as run metadata rather than embedded in the
    # file itself; embedding a file's own SHA-256 would be self-referential.
    record["output_sha256"] = _sha256_file(output_path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-root", type=Path, required=True)
    parser.add_argument("--persistent-entities", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--target-class")
    parser.add_argument("--target-entity-id")
    parser.add_argument("--reference-event-id")
    parser.add_argument("--question-id", default="p3c-query")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        f3_root=arguments.f3_root,
        persistent_entities_path=arguments.persistent_entities,
        question=arguments.question,
        video_id=arguments.video_id,
        target_class=arguments.target_class,
        target_entity_id=arguments.target_entity_id,
        reference_event_id=arguments.reference_event_id,
        question_id=arguments.question_id,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == WhoMovedAnswerStatus.SUCCESS.value else 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
