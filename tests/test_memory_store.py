"""Tests for the dependency-free JSONL observation memory store."""

from pathlib import Path

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.store import MemoryFormatError, ObservationMemoryStore


def make_observation(
    *,
    class_name: str = "bottle",
    frame_index: int = 30,
    timestamp: float = 1.0,
    confidence: float | None = 0.95,
    zone: str | None = "left_region",
) -> Observation:
    return Observation(
        video_id="fixture_video_001",
        frame_index=frame_index,
        timestamp=timestamp,
        track_id=f"track-{class_name}",
        persistent_id=f"{class_name}-1",
        entity_type="object",
        class_name=class_name,
        bbox=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
        confidence=confidence,
        zone=zone,
    )


def test_observations_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    observations = [
        make_observation(timestamp=1.0, frame_index=30),
        make_observation(class_name="book", timestamp=2.0, frame_index=60),
    ]
    store = ObservationMemoryStore(path)

    for observation in observations:
        store.append(observation)

    assert store.load_all() == observations
    assert store.find_by_class("bottle") == [observations[0]]


def test_jsonl_load_preserves_file_order(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    store = ObservationMemoryStore(path)
    expected = [
        make_observation(timestamp=4.0, frame_index=120),
        make_observation(timestamp=1.0, frame_index=30),
    ]
    for observation in expected:
        store.append(observation)

    assert store.load_all() == expected


def test_malformed_jsonl_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(MemoryFormatError, match="line 1"):
        ObservationMemoryStore(path).load_all()
