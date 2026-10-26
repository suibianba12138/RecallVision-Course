"""Generic category and target selection contracts for the unified stages."""

from __future__ import annotations

import json
from pathlib import Path

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.store import ObservationMemoryStore


def test_p2b_builds_non_bottle_memory_and_accepts_explicit_entity_id(tmp_path: Path) -> None:
    from recallvision.cli.p2b import run

    video_id = "sha256:video-hash"
    persistent_id = "obj:cake:0001"
    memory_path = tmp_path / "observations.jsonl"
    ObservationMemoryStore(memory_path).append(
        Observation(
            video_id=video_id,
            frame_index=7,
            timestamp=0.7,
            track_id="cake-track-1",
            persistent_id=persistent_id,
            entity_type="object",
            class_name="cake",
            bbox=BoundingBox(1, 1, 5, 5),
            confidence=0.9,
            zone="center_region",
        )
    )
    entities_path = tmp_path / "persistent_entities.json"
    entities_path.write_text(
        json.dumps([{"persistent_id": persistent_id, "class_name": "cake"}]),
        encoding="utf-8",
    )
    source_summary_path = tmp_path / "entity_summary.json"
    source_summary_path.write_text(
        json.dumps(
            {
                "p2a_r1_pipeline_status": "PASS",
                "p2a_r1_identity_validation_status": "PENDING_HUMAN_REVIEW",
                "video_sha256": "video-hash",
                "video_metadata": {"duration": 1.0},
            }
        ),
        encoding="utf-8",
    )

    result = run(
        consolidated_memory=memory_path,
        persistent_entities=entities_path,
        entity_summary=source_summary_path,
        config_path=Path(__file__).parents[1] / "configs/p2b_state_timeline.yaml",
        output_dir=tmp_path / "p2b",
        question="Where is the cake?",
        target_class="cake",
        target_entity_id=persistent_id,
    )

    assert result["p2b_status"] == "PASS"
    assert result["query"]["persistent_id"] == persistent_id
    assert result["identity_review_status"] == "PENDING_HUMAN_REVIEW"
    assert [item["class_name"] for item in result["timelines"]] == ["cake"]


def test_p2a_r1_consolidates_non_bottle_class_without_bottle_query(monkeypatch, tmp_path: Path) -> None:
    from recallvision.cli import p2a_r1

    video_id = "sha256:video-hash"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"synthetic video bytes")
    memory_path = tmp_path / "observations.jsonl"
    store = ObservationMemoryStore(memory_path)
    for frame in range(3):
        store.append(
            Observation(
                video_id=video_id,
                frame_index=frame,
                timestamp=frame / 10,
                track_id="cake-track-1",
                persistent_id=None,
                entity_type="object",
                class_name="cake",
                bbox=BoundingBox(10, 10, 30, 30),
                confidence=0.9,
                zone="center_region",
            )
        )
    monkeypatch.setattr(
        p2a_r1,
        "read_video_metadata",
        lambda _path: {"width": 100, "height": 100, "fps": 10.0, "duration": 1.0},
    )
    monkeypatch.setattr(p2a_r1, "sha256_file", lambda _path: "video-hash")
    monkeypatch.setattr(
        p2a_r1,
        "_extract_tracklet_descriptors",
        lambda _video, summaries, _config: {item.key: (1.0,) for item in summaries},
    )
    monkeypatch.setattr(
        p2a_r1,
        "_extract_canonical_appearances",
        lambda _video, canonical, _config: {
            (item.group_id, item.observation.frame_index): (1.0,) for item in canonical
        },
    )
    monkeypatch.setattr(p2a_r1, "_write_r1_contact_sheet", lambda *_args: None)
    monkeypatch.setattr(p2a_r1, "_write_fullframe_audit", lambda *_args: None)
    result = p2a_r1.run_p2a_r1(
        input_memory=memory_path,
        input_video=video_path,
        output_dir=tmp_path / "p2a",
        config_path=Path(__file__).parents[1] / "configs/p2a_r1_entity_consolidation.yaml",
    )

    assert result.summary["observations_by_class"] == {"cake": 3}
    assert result.summary["persistent_entity_count"] == 1
    assert result.summary["persistent_entities"][0]["class_name"] == "cake"
    assert result.summary["p2a_r1_identity_validation_status"] == "PENDING_HUMAN_REVIEW"
    assert result.summary["query"] == {"status": "NOT_REQUESTED"}


def test_p2c_reports_non_bottle_zone_events_generically(tmp_path: Path) -> None:
    from recallvision.cli.p2c import run
    from recallvision.memory.schema import EvidenceRef
    from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline

    video_id = "sha256:video-hash"
    persistent_id = "obj:cake:0001"

    def evidence(timestamp: float, frame: int, label: str) -> EvidenceRef:
        return EvidenceRef(
            video_id=video_id,
            timestamp=timestamp,
            frame_indices=[frame],
            segment_ref=f"{video_id}#{label}",
        )

    segments = (
        ObjectStateRecord(
            persistent_id, "cake", video_id, 0, 2, 0.0, 0.2, "left_region", 2,
            evidence(0.0, 0, "first-1"), evidence(0.2, 2, "last-1"), ("track-1",), (0, 2), 0.9,
        ),
        ObjectStateRecord(
            persistent_id, "cake", video_id, 5, 7, 0.5, 0.7, "right_region", 2,
            evidence(0.5, 5, "first-2"), evidence(0.7, 7, "last-2"), ("track-1",), (5, 7), 0.9,
        ),
    )
    timeline = ObjectStateTimeline(persistent_id, "cake", video_id, segments)
    timeline_path = tmp_path / "timelines.json"
    timeline_path.write_text(json.dumps([timeline.to_record()]), encoding="utf-8")
    summary_path = tmp_path / "p2b_summary.json"
    summary_path.write_text(json.dumps({"p2b_status": "PASS", "perception_rerun": False}), encoding="utf-8")

    result = run(
        timeline_path=timeline_path,
        summary_path=summary_path,
        config_path=Path(__file__).parents[1] / "configs/p2c_zone_transition_events.yaml",
        output_dir=tmp_path / "p2c",
    )

    assert result["event_count"] == 1
    assert result["events_by_class"] == {"cake": 1}
    assert (tmp_path / "p2c/zone_change_events.txt").is_file()
