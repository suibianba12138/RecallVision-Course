from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from test_identity_overlay import make_run


def test_observation_review_is_explicit_reversible_and_does_not_edit_source(tmp_path: Path) -> None:
    from recallvision.memory.trust_review import TrustReviewOverlay

    run = make_run(tmp_path)
    source = run / "p2a_r1" / "observations_consolidated.jsonl"
    source_before = source.read_bytes()
    trust = TrustReviewOverlay(
        run,
        tmp_path / "trust" / "reviews.jsonl",
        identity_ledger_path=tmp_path / "identity" / "operations.jsonl",
    )
    candidates = trust.observation_candidates("obj:cup:0001")
    assert len(candidates) == 2
    candidate = candidates[0]
    assert candidate["trust_status"] == "UNCERTAIN"
    assert candidate["source_algorithm"]
    assert candidate["raw_observation_id"].startswith("obs:")

    recorded = trust.record_observation_review(
        candidate["raw_observation_id"],
        "SUPPORTED",
        reviewed_frame_indices=[candidate["frame_index"]],
        reason="The cup is visibly present inside the box.",
        expected_revision=0,
    )
    snapshot = trust.snapshot()
    assert recorded["provenance"] == "HUMAN_REVIEW"
    assert snapshot["observations"][candidate["raw_observation_id"]]["trust_status"] == "SUPPORTED"
    assert snapshot["observations"][candidate["raw_observation_id"]]["reviewed_frame_indices"] == [candidate["frame_index"]]
    assert source.read_bytes() == source_before

    revoked = trust.revoke(recorded["operation_id"], expected_revision=1)
    after_revoke = trust.snapshot()
    assert revoked["status"] == "REVOKED"
    assert after_revoke["observations"][candidate["raw_observation_id"]]["trust_status"] == "UNCERTAIN"
    assert [item["action"] for item in after_revoke["history"]["operations"]] == ["OBSERVATION_REVIEW", "REVOKE"]


def test_observation_review_requires_exact_checked_frame_and_rejects_stale_revision(tmp_path: Path) -> None:
    from recallvision.memory.identity_overlay import IdentityOverlay
    from recallvision.memory.trust_review import TrustReviewOverlay, TrustReviewError

    run = make_run(tmp_path)
    identity_path = tmp_path / "identity" / "operations.jsonl"
    trust = TrustReviewOverlay(run, tmp_path / "trust.jsonl", identity_ledger_path=identity_path)
    candidate = trust.observation_candidates("obj:cup:0001")[0]
    with pytest.raises(TrustReviewError) as missing_frame:
        trust.record_observation_review(
            candidate["raw_observation_id"], "SUPPORTED", reviewed_frame_indices=[],
            reason="review", expected_revision=0,
        )
    assert missing_frame.value.status == "REVIEW_FRAME_REQUIRED"

    overlay = IdentityOverlay(run, identity_path)
    recorded = trust.record_observation_review(
        candidate["raw_observation_id"], "SUPPORTED",
        reviewed_frame_indices=[candidate["frame_index"]], reason="reviewed before identity revision change",
        expected_revision=0,
    )
    overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[
            {"video_id": overlay.video_id, "frame_index": 11},
            {"video_id": overlay.video_id, "frame_index": 20},
        ],
        expected_revision=0,
    )
    snapshot = trust.snapshot()
    assert snapshot["overlay_revision"] == 1
    assert snapshot["stale_operation_ids"] == [recorded["operation_id"]]
    assert snapshot["observations"][candidate["raw_observation_id"]]["trust_status"] == "UNCERTAIN"


def test_supported_event_requires_reviewed_boundaries_and_is_audited(tmp_path: Path) -> None:
    from recallvision.memory.trust_review import TrustReviewOverlay, TrustReviewError

    run = make_run(tmp_path)
    observations_path = run / "p2a_r1" / "observations_consolidated.jsonl"
    observation_rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    observation_rows[1]["bbox"] = {"x1": 3, "y1": 2, "x2": 10, "y2": 9}
    observations_path.write_text("".join(json.dumps(item) + "\n" for item in observation_rows))
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for stage in manifest["stages"]:
        for record in stage["outputs"]:
            if Path(record["path"]).resolve() == observations_path.resolve():
                record["sha256"] = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    video_id = json.loads((run / "p2a_r1" / "entity_consolidation_summary.json").read_text())["video_id"]
    event_dir = run / "p2d"
    event_dir.mkdir()
    event_path = event_dir / "move_object_events.json"
    event = {
        "event_id": "move-object:test:cup:10->11",
        "event_type": "MOVE_OBJECT",
        "start_time": 1.0,
        "end_time": 1.1,
        "object_ids": ["obj:cup:0001"],
        "evidence": [
            {"video_id": video_id, "timestamp": 1.0, "frame_indices": [10]},
            {"video_id": video_id, "timestamp": 1.1, "frame_indices": [11]},
        ],
        "source_segment_ids": [],
    }
    event_path.write_text(json.dumps([event]))
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"].append({"name": "P2D", "outputs": [{
        "path": str(event_path.resolve()),
        "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }]})
    manifest_path.write_text(json.dumps(manifest))
    trust = TrustReviewOverlay(
        run,
        tmp_path / "trust.jsonl",
        identity_ledger_path=tmp_path / "identity.jsonl",
    )
    candidates = trust.observation_candidates("obj:cup:0001")
    observation_ids = [item["raw_observation_id"] for item in candidates]

    with pytest.raises(TrustReviewError) as incomplete:
        trust.record_event_review(
            event["event_id"], "SUPPORTED", reviewed_observation_ids=observation_ids[:1],
            reviewed_frame_indices=[10], reason="checked one boundary", expected_revision=0,
        )
    assert incomplete.value.status == "EVENT_EVIDENCE_INCOMPLETE"

    recorded = trust.record_event_review(
        event["event_id"], "SUPPORTED", reviewed_observation_ids=observation_ids,
        reviewed_frame_indices=[10, 11],
        reason="The two visible positions differ; motion between them was not observed continuously.",
        expected_revision=0,
    )
    snapshot = trust.snapshot()
    assert recorded["event_review_status"] == "SUPPORTED"
    assert recorded["interpretation"] == "POSITION_CHANGE_BETWEEN_TRUSTED_OBSERVATIONS"
    assert snapshot["events"][event["event_id"]]["status"] == "SUPPORTED"
    assert snapshot["events"][event["event_id"]]["provenance"] == "HUMAN_REVIEW"
    event_path.write_text(json.dumps([event, {"event_id": "changed-after-review"}]))
    stale = trust.snapshot()
    assert event["event_id"] not in stale["events"]
    assert recorded["operation_id"] in stale["stale_operation_ids"]


def test_unreviewed_run_rebuild_returns_uncertain_instead_of_an_answer(tmp_path: Path) -> None:
    from recallvision.pipeline.identity_overlay_rebuild import rebuild_visual_memory
    from recallvision.memory.trust_review import TrustReviewOverlay

    run = make_run(tmp_path)
    source_paths = [
        run / "p2a_r1" / "observations_consolidated.jsonl",
        run / "p2a_r1" / "persistent_entities.json",
        run / "p2a_r1" / "track_to_persistent_mapping.json",
        run / "p2a_r1" / "entity_consolidation_summary.json",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    trust_path = tmp_path / "trust.jsonl"
    result = rebuild_visual_memory(
        run_dir=run,
        ledger_path=tmp_path / "identity.jsonl",
        trust_ledger_path=trust_path,
        output_root=tmp_path / "trusted-rebuild",
        target_class="cup",
        target_entity_id="obj:cup:0001",
        questions=["Where is the cup?", "When was the cup moved?", "Where was the cup before?"],
    )
    answers = [json.loads(Path(item["path"]).read_text()) for item in result["query_results"]]
    assert result["run_status"] == "COMPLETED"
    assert result["trust_review"]["trusted_observation_count"] == 0
    assert [item["status"] for item in answers] == [
        "INSUFFICIENT_TRUSTED_EVIDENCE", "INSUFFICIENT_TRUSTED_EVIDENCE", "INSUFFICIENT_TRUSTED_EVIDENCE",
    ]
    assert all(item["evidence"] == [] and item["trust_provenance"]["interpolation_used"] is False for item in answers)
    revision = Path(result["output_dir"])
    assert (revision / "p2b/object_state_timelines.json").read_text() == "[]\n"
    assert (revision / "effective_identity/observations_effective.jsonl").read_bytes() == b""
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())


def test_trusted_rebuild_gates_where_when_and_before_on_explicit_review(tmp_path: Path, monkeypatch) -> None:
    from recallvision.cli import p2c, p2d
    from recallvision.memory.identity_overlay import IdentityOverlay
    from recallvision.memory.state_timeline import StateTimelineConfig, build_state_timelines
    from recallvision.memory.store import ObservationMemoryStore
    from recallvision.memory.trust_review import TrustReviewOverlay
    from recallvision.pipeline import identity_overlay_rebuild, runner
    from recallvision.pipeline.identity_overlay_rebuild import rebuild_visual_memory

    run = make_run(tmp_path)
    observations_path = run / "p2a_r1" / "observations_consolidated.jsonl"
    observation_rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    observation_rows[1]["bbox"] = {"x1": 3, "y1": 2, "x2": 10, "y2": 9}
    observations_path.write_text("".join(json.dumps(item) + "\n" for item in observation_rows))
    event_path = run / "p2d" / "move_object_events.json"
    event_path.parent.mkdir()
    video_id = observation_rows[0]["video_id"]
    event = {
        "event_id": "move-object:reviewed:cup:10->11", "event_type": "MOVE_OBJECT",
        "start_time": 1.0, "end_time": 1.1, "person_ids": [],
        "object_ids": ["obj:cup:0001"], "zone_before": "center_region", "zone_after": "center_region",
        "evidence": [
            {"video_id": video_id, "timestamp": 1.0, "frame_indices": [10], "segment_ref": "before"},
            {"video_id": video_id, "timestamp": 1.1, "frame_indices": [11], "segment_ref": "after"},
        ],
        "source_segment_ids": [],
    }
    event_path.write_text(json.dumps([event]))
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for stage in manifest["stages"]:
        for record in stage["outputs"]:
            if Path(record["path"]).resolve() == observations_path.resolve():
                record["sha256"] = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    manifest["stages"].append({"name": "P2D", "outputs": [{
        "path": str(event_path.resolve()), "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }]})
    manifest_path.write_text(json.dumps(manifest))

    identity_path = tmp_path / "identity.jsonl"
    trust_path = tmp_path / "trust.jsonl"
    identity = IdentityOverlay(run, identity_path)
    trust = TrustReviewOverlay(run, trust_path, identity_ledger_path=identity_path)
    refs = trust.observation_candidates("obj:cup:0001")
    trust.record_event_review(
        event["event_id"], "SUPPORTED",
        reviewed_observation_ids=[item["raw_observation_id"] for item in refs],
        reviewed_frame_indices=[10, 11],
        reason="The object is in different image positions in the checked frames; motion between them is not observed.",
        expected_revision=0,
    )

    def timeline_run(*, consolidated_memory, persistent_entities, entity_summary, config_path, output_dir, **_kwargs):
        observations = ObservationMemoryStore(consolidated_memory).load_all()
        timelines = build_state_timelines(
            observations,
            StateTimelineConfig(max_observation_gap_seconds=1.0, minimum_observations_per_segment=1),
        )
        output_dir.mkdir(parents=True)
        (output_dir / "object_state_timelines.json").write_text(json.dumps([item.to_record() for item in timelines]))
        (output_dir / "state_timeline_summary.json").write_text(json.dumps({"p2b_status": "PASS", "video_id": video_id}))
        (output_dir / "last_seen_states.json").write_text("[]")
        return {"p2b_status": "PASS"}

    def zone_run(*, timeline_path, summary_path, config_path, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        timelines = json.loads(Path(timeline_path).read_text())
        (output_dir / "zone_change_events.json").write_text("[]")
        (output_dir / "zone_change_event_summary.json").write_text(json.dumps({"p2c_status": "PASS", "event_count": 0, "video_id": video_id}))
        (output_dir / "zone_change_events.txt").write_text("none\n")
        return {"p2c_status": "PASS", "event_count": 0}

    def movement_run(*, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "move_object_events.json").write_text("[]")
        (output_dir / "move_object_evidence.json").write_text("[]")
        (output_dir / "move_object_event_summary.json").write_text(json.dumps({"p2d_status": "NO_ACCEPTED_EVENTS", "move_object_event_count": 0}))
        return {"p2d_status": "NO_ACCEPTED_EVENTS", "move_object_event_count": 0}

    monkeypatch.setattr(identity_overlay_rebuild.p2b, "run", timeline_run)
    monkeypatch.setattr(identity_overlay_rebuild.p2c, "run", zone_run)
    monkeypatch.setattr(identity_overlay_rebuild.p2d, "run", movement_run)
    def render(_video, _observation, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"reviewed evidence")
        return output_path
    monkeypatch.setattr(runner, "render_evidence_frame", render)

    rebuilt = rebuild_visual_memory(
        run_dir=run, ledger_path=identity_path, trust_ledger_path=trust_path,
        output_root=tmp_path / "revisions", target_class="cup", target_entity_id="obj:cup:0001",
        questions=["Where is the cup?", "When was the cup moved?", "Where was the cup before?"],
    )
    results = [json.loads(Path(item["path"]).read_text()) for item in rebuilt["query_results"]]
    assert rebuilt["run_status"] == "COMPLETED"
    assert rebuilt["trust_review"]["trusted_observation_count"] == 2
    assert [item["status"] for item in results] == ["TARGET_LAST_SEEN", "POSITION_CHANGE_BETWEEN_OBSERVATIONS", "OK"]
    assert "exact movement onset and path" in results[1]["answer"]
    assert "2D image coordinates" in results[2]["answer"]
    assert results[1]["event_review_provenance"][0]["provenance"] == "HUMAN_REVIEW"
    assert all(item["trust_provenance"]["interpolation_used"] is False for item in results)
