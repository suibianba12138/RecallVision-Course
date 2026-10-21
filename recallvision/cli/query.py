"""Reload an observation memory and answer a deterministic query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recallvision.memory.schema import Observation
from recallvision.memory.store import ObservationMemoryStore
from recallvision.query.object_location import answer_where_object
from recallvision.query.parser import parse_query
from recallvision.query.schema import AnswerStatus


def run(
    *,
    memory_path: Path,
    question: str,
    metadata_path: Path,
    evidence_output: Path | None = None,
) -> int:
    """Answer a query from persisted memory without loading a detector."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observations = ObservationMemoryStore(memory_path).load_all()
    if not observations and "video_id" not in metadata:
        raise ValueError("metadata must contain video_id when memory is empty")
    video_id = str(metadata.get("video_id", observations[0].video_id))
    query = parse_query(
        question,
        video_id=video_id,
        question_id="query-cli",
    )
    threshold = metadata.get("confidence_threshold")
    answer = answer_where_object(query, observations, min_confidence=threshold)
    selected = _matching_observation(answer, observations)

    evidence_image: str | None = None
    if evidence_output is not None and selected is not None:
        from recallvision.visualization.evidence import render_evidence_frame

        video_path = metadata.get("video_path")
        if not video_path:
            raise ValueError("metadata must contain video_path to render evidence")
        rendered_path = render_evidence_frame(video_path, selected, evidence_output)
        evidence_image = str(rendered_path)

    evidence_ref: dict[str, Any] | None = None
    if answer.evidence:
        evidence = answer.evidence[0]
        evidence_ref = {
            "video_id": evidence.video_id,
            "timestamp": evidence.timestamp,
            "frame_indices": evidence.frame_indices,
            "segment_ref": evidence.segment_ref,
        }

    query_result: dict[str, Any] = {
        "status": answer.status.value,
        "question": question,
        "query_type": query.query_type.value,
        "target_class": answer.target_class,
        "target_entity_id": answer.target_entity_id,
        "answer": answer.answer,
        "zone": answer.zone,
        "timestamp": answer.timestamp,
        "frame_index": answer.frame_index,
        "track_id": selected.track_id if selected else None,
        "persistent_id": selected.persistent_id if selected else None,
        "confidence": selected.confidence if selected else None,
        "evidence_ref": evidence_ref,
        "evidence_image": evidence_image,
    }
    output_path = metadata_path.parent / "query_result.json"
    output_path.write_text(json.dumps(query_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(query_result, indent=2, sort_keys=True))
    return 0 if answer.status is AnswerStatus.OK else 2


def _matching_observation(answer: Any, observations: list[Observation]) -> Observation | None:
    if answer.status is not AnswerStatus.OK:
        return None
    candidates = [
        observation
        for observation in observations
        if observation.frame_index == answer.frame_index
        and observation.class_name == answer.target_class
        and observation.video_id == answer.evidence[0].video_id
    ]
    for observation in candidates:
        if observation.persistent_id == answer.target_entity_id:
            return observation
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--evidence-output",
        type=Path,
        help="Optional JPEG path; rendering seeks the selected frame without detection.",
    )
    arguments = parser.parse_args()
    return run(
        memory_path=arguments.memory,
        question=arguments.question,
        metadata_path=arguments.metadata,
        evidence_output=arguments.evidence_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
