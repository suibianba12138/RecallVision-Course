from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recallvision.memory.identity_overlay import IdentityOverlay, OverlayError


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "p1b").mkdir(parents=True)
    (run / "p2a_r1").mkdir()
    video = run / "source.mp4"
    video.write_bytes(b"test-video-source")
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    video_id = f"sha256:{video_sha}"
    entities = [
        {"persistent_id": "obj:cup:0001", "class_name": "cup", "video_id": video_id, "fragment_ids": ["fragment:a"], "observation_count": 2},
        {"persistent_id": "obj:cup:0002", "class_name": "cup", "video_id": video_id, "fragment_ids": ["fragment:b"], "observation_count": 2},
        {"persistent_id": "obj:cup:0003", "class_name": "cup", "video_id": video_id, "fragment_ids": ["fragment:c"], "observation_count": 1},
        {"persistent_id": "obj:cup:0004", "class_name": "cup", "video_id": video_id, "fragment_ids": ["fragment:e"], "observation_count": 1},
        {"persistent_id": "obj:bottle:0001", "class_name": "bottle", "video_id": video_id, "fragment_ids": ["fragment:d"], "observation_count": 1},
    ]
    observations = []
    for entity, track, frame in [(entities[0], "1", 10), (entities[0], "1", 11), (entities[1], "2", 20), (entities[1], "2", 21), (entities[2], "3", 30), (entities[3], "4", 40), (entities[4], "5", 50)]:
        observations.append({
            "video_id": video_id, "frame_index": frame, "timestamp": frame / 10,
            "track_id": track, "persistent_id": entity["persistent_id"],
            "entity_type": "object", "class_name": entity["class_name"],
            "bbox": {"x1": 1, "y1": 2, "x2": 8, "y2": 9},
            "confidence": 0.9, "zone": "center_region", "source_track_ids": [track],
        })
    (run / "p1b" / "observations.jsonl").write_text("".join(json.dumps(x) + "\n" for x in observations))
    (run / "p2a_r1" / "observations_consolidated.jsonl").write_text("".join(json.dumps(x) + "\n" for x in observations))
    mapping = {f"{video_id}|{x['class_name']}|{x['track_id']}": x["persistent_id"] for x in observations}
    (run / "p2a_r1" / "track_to_persistent_mapping.json").write_text(json.dumps(mapping))
    (run / "p2a_r1" / "persistent_entities.json").write_text(json.dumps(entities))
    (run / "p2a_r1" / "entity_conflicts.json").write_text(json.dumps([{
        "video_id": video_id, "class_name": "cup", "fragment_a": "fragment:b",
        "fragment_b": "fragment:c", "frame_indices": [30],
        "reason": "SIMULTANEOUS_SPATIAL_CONFLICT", "relation_type": "CONFLICT",
    }]))
    (run / "p2a_r1" / "continuation_candidates.json").write_text(json.dumps([{
        "video_id": video_id, "class_name": "cup", "source_fragment_id": "fragment:a",
        "target_fragment_id": "fragment:b", "accepted": False, "score": 0.61,
        "rejection_reason": "score_below_threshold",
    }]))
    summary = {
        "video_id": video_id, "video_sha256": video_sha,
        "video_metadata": {"width": 10, "height": 10, "duration": 5, "fps": 10, "frame_count": 50},
        "p2a_r1_pipeline_status": "PASS",
        "p2a_r1_identity_validation_status": "PENDING_HUMAN_REVIEW",
        "track_to_persistent_mapping": mapping,
        "persistent_entities": entities,
    }
    summary_path = run / "p2a_r1" / "entity_consolidation_summary.json"
    summary_path.write_text(json.dumps(summary))
    pinned_files = [
        run / "p2a_r1" / "track_to_persistent_mapping.json",
        run / "p2a_r1" / "persistent_entities.json",
        run / "p2a_r1" / "entity_conflicts.json",
        run / "p2a_r1" / "continuation_candidates.json",
        run / "p2a_r1" / "observations_consolidated.jsonl",
        summary_path,
    ]
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "test-run",
        "video": {"path": str(video), "sha256": video_sha, "video_id": video_id},
        "stages": [{"name": "P2A-R1", "outputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in pinned_files
        ]}],
    }))
    return run


def repin_manifest_outputs(run: Path, changed_paths: list[Path]) -> None:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest_by_path = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in changed_paths}
    for stage in manifest.get("stages", []):
        for record in stage.get("outputs", []):
            path = str(Path(record.get("path", "")).resolve())
            if path in digest_by_path:
                record["sha256"] = digest_by_path[path]
    manifest_path.write_text(json.dumps(manifest))


def test_explicit_confirmation_builds_effective_mapping_and_revocation_is_append_only(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "overlay" / "operations.jsonl")
    before = overlay.effective_mapping()
    assert before["obj:cup:0001"]["effective_entity_id"] == "obj:cup:0001"

    created = overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}, {"video_id": overlay.video_id, "frame_index": 20}],
        expected_revision=0,
    )
    assert created["status"] == "RECORDED"
    mapping = overlay.effective_mapping()
    assert mapping["obj:cup:0001"]["effective_entity_id"] == mapping["obj:cup:0002"]["effective_entity_id"]
    assert mapping["obj:cup:0001"]["identity_provenance"] == "HUMAN_CONFIRMED"

    revoked = overlay.revoke(created["operation_id"], expected_revision=1)
    assert revoked["status"] == "REVOKED"
    assert overlay.effective_mapping()["obj:cup:0001"]["effective_entity_id"] != overlay.effective_mapping()["obj:cup:0002"]["effective_entity_id"]
    records = [json.loads(line) for line in overlay.ledger_path.read_text().splitlines()]
    assert [item["action"] for item in records] == ["RELATION", "REVOKE"]
    history = overlay.history()
    assert history["operations"][0]["revocation_status"] == "REVOKED"
    assert history["operations"][1]["action"] == "REVOKE"


def test_effective_entity_summary_unions_source_track_ids(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    entities_path = run / "p2a_r1" / "persistent_entities.json"
    entities = json.loads(entities_path.read_text())
    entities[0]["track_ids"] = ["4"]
    entities[1]["track_ids"] = ["206"]
    entities_path.write_text(json.dumps(entities))
    summary_path = run / "p2a_r1" / "entity_consolidation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["persistent_entities"] = entities
    summary_path.write_text(json.dumps(summary))
    repin_manifest_outputs(run, [entities_path, summary_path])

    overlay = IdentityOverlay(run, tmp_path / "overlay" / "ops.jsonl")
    overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[
            {"video_id": overlay.video_id, "frame_index": 11},
            {"video_id": overlay.video_id, "frame_index": 20},
        ],
        expected_revision=0,
    )

    from recallvision.pipeline.identity_overlay_rebuild import _effective_entities

    entities = _effective_entities(overlay, overlay.effective_mapping(), revision=1)
    merged = next(item for item in entities if item["persistent_id"] == "obj:cup:0001")
    assert merged["source_persistent_ids"] == ["obj:cup:0001", "obj:cup:0002"]
    assert merged["track_ids"] == ["206", "4"]


def test_snapshot_pins_one_revision_across_later_ledger_mutation(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    pinned = overlay.snapshot()
    overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
    )
    assert pinned["overlay_revision"] == 0
    assert pinned["history"]["overlay_revision"] == 0
    assert pinned["effective_mapping"]["obj:cup:0001"]["effective_entity_id"] != pinned["effective_mapping"]["obj:cup:0002"]["effective_entity_id"]
    current = overlay.snapshot()
    assert current["overlay_revision"] == 1
    assert current["effective_mapping"]["obj:cup:0001"]["effective_entity_id"] == current["effective_mapping"]["obj:cup:0002"]["effective_entity_id"]


def test_spatial_conflict_blocks_same_instance_with_evidence(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    with pytest.raises(OverlayError) as captured:
        overlay.record_relation(
            "obj:cup:0002", "obj:cup:0003", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 21}],
            expected_revision=0,
        )
    assert captured.value.status == "SPATIAL_CONFLICT"
    assert captured.value.details["conflict_frames"] == [30]
    assert not overlay.ledger_path.exists()


def test_relation_and_transitive_conflicts_are_not_silently_ignored(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    one = overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "DIFFERENT_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
    )
    assert one["status"] == "RECORDED"
    overlay.record_relation(
        "obj:cup:0002", "obj:cup:0004", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 21}], expected_revision=1,
    )
    with pytest.raises(OverlayError) as captured:
        overlay.record_relation(
            "obj:cup:0001", "obj:cup:0004", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=2,
        )
    assert captured.value.status == "RELATION_CONFLICT"
    with pytest.raises(OverlayError) as spatial:
        overlay.record_relation(
            "obj:cup:0002", "obj:cup:0003", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 21}], expected_revision=2,
        )
    assert spatial.value.status == "SPATIAL_CONFLICT"


def test_different_instance_decision_is_visible_in_effective_mapping_provenance(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    operation = overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "DIFFERENT_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
    )
    mapping = overlay.snapshot()["effective_mapping"]
    for entity_id in ("obj:cup:0001", "obj:cup:0002"):
        assert mapping[entity_id]["identity_provenance"] == "HUMAN_REVIEWED_DISTINCT"
        assert mapping[entity_id]["human_confirmation_operation_ids"] == [operation["operation_id"]]


def test_stale_revision_duplicate_missing_entity_and_class_mismatch_fail_explicitly(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    refs = [{"video_id": overlay.video_id, "frame_index": 11}]
    with pytest.raises(OverlayError, match="STALE_REVISION"):
        overlay.record_relation("obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE", evidence_refs=refs, expected_revision=1)
    recorded = overlay.record_relation("obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE", evidence_refs=refs, expected_revision=0)
    with pytest.raises(OverlayError, match="DUPLICATE_RELATION"):
        overlay.record_relation("obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE", evidence_refs=refs, expected_revision=1)
    with pytest.raises(OverlayError, match="ENTITY_NOT_FOUND"):
        overlay.record_relation("obj:cup:0001", "obj:cup:9999", "SAME_INSTANCE", evidence_refs=refs, expected_revision=1)
    with pytest.raises(OverlayError, match="CLASS_MISMATCH"):
        overlay.record_relation("obj:cup:0001", "obj:bottle:0001", "SAME_INSTANCE", evidence_refs=refs, expected_revision=1)
    with pytest.raises(OverlayError, match="ALREADY_REVOKED"):
        overlay.revoke(recorded["operation_id"], expected_revision=1)
        overlay.revoke(recorded["operation_id"], expected_revision=2)


def test_cross_video_entity_and_invalid_evidence_are_rejected(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    entities_path = run / "p2a_r1" / "persistent_entities.json"
    entities = json.loads(entities_path.read_text())
    entities[1]["video_id"] = "sha256:another-video"
    entities_path.write_text(json.dumps(entities))
    summary_path = run / "p2a_r1" / "entity_consolidation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["persistent_entities"] = entities
    summary_path.write_text(json.dumps(summary))
    repin_manifest_outputs(run, [entities_path, summary_path])
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    with pytest.raises(OverlayError) as cross_video:
        overlay.record_relation(
            "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
        )
    assert cross_video.value.status == "CROSS_VIDEO_ENTITY"

    entities[1]["video_id"] = overlay.video_id
    entities_path.write_text(json.dumps(entities))
    summary["persistent_entities"] = entities
    summary_path.write_text(json.dumps(summary))
    repin_manifest_outputs(run, [entities_path, summary_path])
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    with pytest.raises(OverlayError) as invalid_evidence:
        overlay.record_relation(
            "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 49}], expected_revision=0,
        )
    assert invalid_evidence.value.status == "INVALID_EVIDENCE"


def test_review_image_hash_and_source_mapping_consistency_are_audited(tmp_path: Path, monkeypatch) -> None:
    from recallvision.visualization import evidence as evidence_module

    run = make_run(tmp_path)
    image = tmp_path / "review.jpg"
    image.write_bytes(b"review image bytes")

    def render_stub(video_path, observation, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"review image bytes")
        return output

    monkeypatch.setattr(evidence_module, "render_evidence_frame", render_stub)
    overlay = IdentityOverlay(run, tmp_path / "overlay" / "ops.jsonl")
    operation = overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11, "evidence_path": str(image)}],
        expected_revision=0,
    )
    assert operation["evidence_refs"][0]["evidence_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()

    incorrect_image = tmp_path / "incorrect.jpg"
    incorrect_image.write_bytes(b"different frame")
    with pytest.raises(OverlayError) as false_evidence:
        IdentityOverlay(run, tmp_path / "bad-evidence" / "ops.jsonl").record_relation(
            "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
            evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11, "evidence_path": str(incorrect_image)}],
            expected_revision=0,
        )
    assert false_evidence.value.status == "INVALID_EVIDENCE"

    timestamp_overlay = IdentityOverlay(run, tmp_path / "timestamp-overlay" / "ops.jsonl")
    with pytest.raises(OverlayError) as wrong_timestamp:
        timestamp_overlay.record_relation(
            "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
            evidence_refs=[{"video_id": timestamp_overlay.video_id, "frame_index": 11, "timestamp": 123.0}],
            expected_revision=0,
        )
    assert wrong_timestamp.value.status == "INVALID_EVIDENCE"

    mapping_path = run / "p2a_r1" / "track_to_persistent_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    first_key = next(key for key in mapping if key.endswith("|1"))
    mapping[first_key] = "obj:cup:0002"
    mapping_path.write_text(json.dumps(mapping))
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["stages"][0]["outputs"]:
        if Path(record["path"]).resolve() == mapping_path.resolve():
            record["sha256"] = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(OverlayError) as mismatch:
        IdentityOverlay(run, tmp_path / "fresh-overlay" / "ops.jsonl")
    assert mismatch.value.status == "SOURCE_MAPPING_MISMATCH"


def test_frozen_artifact_manifest_pin_and_paths_outside_source_run(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    observations_path = run / "p2a_r1" / "observations_consolidated.jsonl"
    observations_path.write_text(observations_path.read_text() + "\n")
    with pytest.raises(OverlayError) as stale:
        IdentityOverlay(run, tmp_path / "overlay" / "ops.jsonl")
    assert stale.value.status == "STALE_SOURCE"

    with pytest.raises(OverlayError) as unsafe_ledger:
        IdentityOverlay(run, run / "identity_operations.jsonl")
    assert unsafe_ledger.value.status == "OUTPUT_IN_SOURCE_RUN"

    from recallvision.pipeline.identity_overlay_rebuild import IdentityRebuildError, rebuild_visual_memory

    safe_run = make_run(tmp_path / "safe")
    ledger_path = tmp_path / "safe" / "overlay" / "ops.jsonl"
    with pytest.raises(IdentityRebuildError) as unsafe_output:
        rebuild_visual_memory(
            run_dir=safe_run,
            ledger_path=ledger_path,
            output_root=safe_run / "derived",
        )
    assert unsafe_output.value.status == "OUTPUT_IN_SOURCE_RUN"


def test_external_hash_manifest_is_required_for_legacy_unpinned_constraints(tmp_path: Path) -> None:
    run = make_run(tmp_path / "legacy")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    unpinned = {
        str((run / "p2a_r1" / "entity_conflicts.json").resolve()),
        str((run / "p2a_r1" / "continuation_candidates.json").resolve()),
    }
    manifest["stages"][0]["outputs"] = [
        item for item in manifest["stages"][0]["outputs"]
        if str(Path(item["path"]).resolve()) not in unpinned
    ]
    manifest_path.write_text(json.dumps(manifest))
    ledger = tmp_path / "legacy" / "overlay" / "ops.jsonl"
    with pytest.raises(OverlayError) as missing:
        IdentityOverlay(run, ledger)
    assert missing.value.status == "SOURCE_PROVENANCE_INCOMPLETE"

    hashes_path = tmp_path / "legacy" / "source_hashes.sha256"
    rows = []
    for path in (run / "p2a_r1" / "entity_conflicts.json", run / "p2a_r1" / "continuation_candidates.json"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}")
    hashes_path.write_text("\n".join(rows) + "\n")
    overlay = IdentityOverlay(run, ledger, hashes_path)
    assert overlay.source_hash_manifest_sha256 == hashlib.sha256(hashes_path.read_bytes()).hexdigest()
    overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
    )
    hashes_path.write_text(hashes_path.read_text() + "\n")
    with pytest.raises(OverlayError) as changed:
        overlay.snapshot()
    assert changed.value.status == "STALE_SOURCE"


def test_mapping_hash_change_invalidates_prior_overlay(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    ledger = tmp_path / "ops.jsonl"
    overlay = IdentityOverlay(run, ledger)
    overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{"video_id": overlay.video_id, "frame_index": 11}], expected_revision=0,
    )
    mapping_path = run / "p2a_r1" / "track_to_persistent_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["sha256:" + json.loads((run / "p2a_r1" / "entity_consolidation_summary.json").read_text())["video_sha256"] + "|cup|1"] = "obj:cup:0002"
    mapping_path.write_text(json.dumps(mapping))
    summary_path = run / "p2a_r1" / "entity_consolidation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["track_to_persistent_mapping"] = mapping
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(OverlayError) as stale:
        IdentityOverlay(run, ledger)
    assert stale.value.status == "STALE_SOURCE"


def test_unreviewed_candidates_remain_unmerged_and_preserve_pending_status(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    pair = overlay.inspect_pair("obj:cup:0001", "obj:cup:0002")
    assert pair["continuation_candidates"][0]["score"] == 0.61
    mapping = overlay.effective_mapping()
    assert mapping["obj:cup:0001"]["effective_entity_id"] != mapping["obj:cup:0002"]["effective_entity_id"]
    assert mapping["obj:cup:0001"]["source_identity_review_status"] == "PENDING_HUMAN_REVIEW"


def test_rebuild_no_event_is_completed_and_revision_output_is_no_clobber(tmp_path: Path) -> None:
    from recallvision.pipeline.identity_overlay_rebuild import IdentityRebuildError, rebuild_visual_memory

    run = make_run(tmp_path)
    result = rebuild_visual_memory(
        run_dir=run, ledger_path=tmp_path / "ops.jsonl", output_root=tmp_path / "revisions",
    )
    assert result["run_status"] == "COMPLETED"
    assert result["p2b_status"] == "PASS"
    assert result["p2d_status"] == "NO_ACCEPTED_EVENTS"
    assert result["move_object_event_count"] == 0
    with pytest.raises(IdentityRebuildError) as existing:
        rebuild_visual_memory(
            run_dir=run, ledger_path=tmp_path / "ops.jsonl", output_root=tmp_path / "revisions",
        )
    assert existing.value.status == "OUTPUT_EXISTS"


def test_render_entity_evidence_uses_existing_renderer_and_is_no_clobber(tmp_path: Path, monkeypatch) -> None:
    from recallvision.visualization import evidence as evidence_module

    run = make_run(tmp_path)
    rendered_frames: list[int] = []

    def render_stub(video_path, observation, output_path):
        rendered_frames.append(observation.frame_index)
        output = Path(output_path)
        output.write_bytes(f"frame-{observation.frame_index}".encode())
        return output

    monkeypatch.setattr(evidence_module, "render_evidence_frame", render_stub)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    listed = overlay.list_entities("cup")
    assert listed[0]["source_track_ids"] == ["1"]
    output_dir = tmp_path / "entity-evidence"
    records = overlay.render_entity_evidence("obj:cup:0001", output_dir)

    assert rendered_frames == [10, 11]
    assert [item["frame_index"] for item in records] == [10, 11]
    assert all(Path(item["evidence_path"]).is_file() for item in records)
    with pytest.raises(OverlayError) as captured:
        overlay.render_entity_evidence("obj:cup:0001", output_dir)
    assert captured.value.status == "OUTPUT_EXISTS"


def test_evidence_cli_accepts_one_entity_for_candidate_gallery() -> None:
    from recallvision.cli.identity_overlay import build_parser

    args = build_parser().parse_args([
        "evidence", "--run-dir", "/run", "--ledger", "/ops.jsonl",
        "--entity-id", "obj:cup:0001", "--output-dir", "/frames",
    ])

    assert args.command == "evidence"
    assert args.entity_id == "obj:cup:0001"
    assert args.entity_a is None
    assert args.entity_b is None


def test_pair_evidence_validation_matches_the_observed_entity_on_shared_frames(tmp_path: Path, monkeypatch) -> None:
    from recallvision.visualization import evidence as evidence_module

    run = make_run(tmp_path)
    observation_path = run / "p2a_r1" / "observations_consolidated.jsonl"
    observations = [json.loads(line) for line in observation_path.read_text().splitlines()]
    entity_b = next(item for item in observations if item["persistent_id"] == "obj:cup:0002")
    shared_frame = dict(entity_b, frame_index=11, timestamp=1.1)
    observations.append(shared_frame)
    observation_path.write_text("".join(json.dumps(item) + "\n" for item in observations))
    repin_manifest_outputs(run, [observation_path])

    def render_stub(video_path, observation, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"{observation.persistent_id}:{observation.frame_index}".encode())
        return output

    monkeypatch.setattr(evidence_module, "render_evidence_frame", render_stub)
    overlay = IdentityOverlay(run, tmp_path / "ops.jsonl")
    records = overlay.render_pair_evidence("obj:cup:0001", "obj:cup:0002", tmp_path / "pair-evidence")
    evidence_b = next(
        item for item in records
        if item["persistent_id"] == "obj:cup:0002" and item["frame_index"] == 11
    )

    operation = overlay.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[{
            "video_id": overlay.video_id,
            "frame_index": 11,
            "timestamp": 1.1,
            "evidence_path": evidence_b["evidence_path"],
        }],
        expected_revision=0,
    )
    assert operation["status"] == "RECORDED"
    assert operation["evidence_refs"][0]["timestamp"] == 1.1
