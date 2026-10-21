"""Run the P1A fixture through memory serialization and retrieval."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from recallvision.memory.store import ObservationMemoryStore
from recallvision.query.object_location import answer_where_object
from recallvision.query.parser import parse_query


DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "p1a_observations.jsonl"


def run(fixture_path: Path) -> int:
    """Load the fixture, round-trip a temporary memory artifact, and query it."""

    fixture_store = ObservationMemoryStore(fixture_path)
    fixture_observations = fixture_store.load_all()

    with tempfile.TemporaryDirectory(prefix="recallvision-p1a-") as temporary_directory:
        artifact_store = ObservationMemoryStore(Path(temporary_directory) / "observations.jsonl")
        for observation in fixture_observations:
            artifact_store.append(observation)
        observations = artifact_store.load_all()

    query = parse_query(
        "Where is the bottle?",
        video_id="fixture_video_001",
        question_id="p1a-demo-question",
    )
    answer = answer_where_object(query, observations)
    if not answer.evidence:
        print(answer.answer)
        return 1

    evidence = answer.evidence[0]
    print(f"Question:\n{query.question}")
    print(f"\nAnswer:\n{answer.answer}")
    print(f"\nTarget entity:\n{answer.target_entity_id} ({answer.target_class})")
    print(f"\nZone:\n{answer.zone}")
    print(f"\nTimestamp:\n{answer.timestamp} s")
    print(f"\nFrame:\n{answer.frame_index}")
    print(
        f"\nEvidence:\n{evidence.video_id} @ {evidence.timestamp} s "
        f"(frame {evidence.frame_indices[0]})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to a P1A observation JSONL fixture.",
    )
    arguments = parser.parse_args()
    return run(arguments.fixture)


if __name__ == "__main__":
    raise SystemExit(main())
