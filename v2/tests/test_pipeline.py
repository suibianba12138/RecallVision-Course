"""Mock-stage integration tests for the unified pipeline contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.memory.state_timeline import StateTimelineConfig, build_state_timelines
from recallvision.memory.store import ObservationMemoryStore
from recallvision.pipeline.runner import run_pipeline


VIDEO_METADATA = {"fps": 10.0, "width": 100, "height": 100, "frame_count": 10, "duration": 1.0}


def _setup_mock_stages(
    monkeypatch,
    tmp_path: Path,
    *,
    entity_count: int = 1,
    mismatched_p2a_hash: bool = False,
    invalid_p2d_status: bool = False,
    supported_move_event: bool = False,
    mismatched_p2c_input: bool = False,
):
    from recallvision.pipeline import runner

    video_path = tmp_path / "cake-new.mp4"
    video_path.write_bytes(b"synthetic new video; not an inference fixture")
    model_path = tmp_path / "local-model.pt"
    model_path.write_bytes(b"local test checkpoint placeholder")
    video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    video_id = f"sha256:{video_hash}"
    monkeypatch.setattr(runner, "read_video_metadata", lambda _path: dict(VIDEO_METADATA))
    monkeypatch.setattr(runner, "_package_version", lambda _name: "test-only")
    monkeypatch.setattr(runner, "_git_value", lambda *_args: "test-head")

    def perception_stage(*, video_path, output_dir, model_name, video_id, **_kwargs):
        output_dir.mkdir(parents=True)
        store = ObservationMemoryStore(output_dir / "observations.jsonl")
        for index in range(entity_count):
            store.append(
                Observation(
                    video_id=video_id,
                    frame_index=7 + index,
                    timestamp=0.7 + index / 10,
                    track_id=f"cake-track-{index + 1}",
                    persistent_id=f"track:cake:{index + 1}",
                    entity_type="object",
                    class_name="cake",
                    bbox=BoundingBox(10 + index * 30, 10, 30 + index * 30, 30),
                    confidence=0.9,
                    zone="center_region",
                )
            )
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "video_sha256": video_hash,
                    "video_path": str(video_path),
                    "video_fps": VIDEO_METADATA["fps"],
                    "video_width": VIDEO_METADATA["width"],
                    "video_height": VIDEO_METADATA["height"],
                    "frame_count": VIDEO_METADATA["frame_count"],
                    "duration": VIDEO_METADATA["duration"],
                    "model_name": str(model_name),
                    "model_weights_path": str(model_name),
                    "model_weights_sha256": hashlib.sha256(Path(model_name).read_bytes()).hexdigest(),
                    "model_version": "test-only",
                    "tracker": "bytetrack.yaml",
                    "perception_status": "PASS_OBSERVATIONS_WRITTEN",
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "analysis_summary.json").write_text(
            json.dumps({"total_observations": entity_count, "observations_by_class": {"cake": entity_count}}),
            encoding="utf-8",
        )
        return 0

    def association_stage(*, input_memory, input_video, output_dir, config_path, **_kwargs):
        output_dir.mkdir(parents=True)
        raw = ObservationMemoryStore(input_memory).load_all()
        entities = []
        consolidated = []
        for index, observation in enumerate(raw):
            persistent_id = f"obj:cake:{index + 1:04d}"
            enriched = replace(observation, persistent_id=persistent_id, source_track_ids=(observation.track_id,))
            consolidated.append(enriched)
            entities.append(
                {
                    "persistent_id": persistent_id,
                    "class_name": "cake",
                    "entity_type": "object",
                    "track_ids": [observation.track_id],
                }
            )
        consolidated_store = ObservationMemoryStore(output_dir / "observations_consolidated.jsonl")
        for observation in consolidated:
            consolidated_store.append(observation)
        (output_dir / "persistent_entities.json").write_text(json.dumps(entities), encoding="utf-8")
        source_summary = {
            "p2a_r1_pipeline_status": "PASS",
            "p2a_r1_identity_validation_status": "PENDING_HUMAN_REVIEW",
            "input_video": str(Path(input_video).resolve()),
            "input_memory": str(Path(input_memory).resolve()),
            "video_sha256": "different-video-hash" if mismatched_p2a_hash else video_hash,
            "video_id": video_id,
            "video_metadata": dict(VIDEO_METADATA),
            "persistent_entities": entities,
            "config_file": str(config_path),
        }
        (output_dir / "entity_consolidation_summary.json").write_text(
            json.dumps(source_summary), encoding="utf-8"
        )
        diagnostic_json = {
            "duplicate_candidates.json": [],
            "duplicate_edges.json": [],
            "duplicate_groups.json": [],
            "continuation_candidates.json": [],
            "continuation_edges.json": [],
            "entity_conflicts.json": [],
            "track_to_persistent_mapping.json": {},
            "tracklet_summaries.json": [],
        }
        for name, value in diagnostic_json.items():
            (output_dir / name).write_text(json.dumps(value), encoding="utf-8")
        (output_dir / "observations_with_persistent_ids.jsonl").write_text("", encoding="utf-8")
        return SimpleNamespace(output_dir=output_dir, summary=source_summary)

    def timeline_stage(
        *, consolidated_memory, persistent_entities, entity_summary, config_path, output_dir,
        target_class=None, target_entity_id=None, question=None, **_kwargs
    ):
        output_dir.mkdir(parents=True)
        observations = ObservationMemoryStore(consolidated_memory).load_all()
        timelines = build_state_timelines(
            observations,
            StateTimelineConfig(max_observation_gap_seconds=1.0, minimum_observations_per_segment=1),
        )
        (output_dir / "object_state_timelines.json").write_text(
            json.dumps([item.to_record() for item in timelines]), encoding="utf-8"
        )
        p2a_record = json.loads(Path(entity_summary).read_text(encoding="utf-8"))
        summary = {
            "p2b_status": "PASS",
            "video_id": timelines[0].video_id,
            "authoritative_inputs": {"video_sha256": p2a_record["video_sha256"]},
            "identity_review_status": p2a_record["p2a_r1_identity_validation_status"],
            "query": {"status": "NOT_REQUESTED"},
            "passed_target_class": target_class,
            "passed_target_entity_id": target_entity_id,
            "passed_question": question,
            "config_file": str(config_path),
        }
        (output_dir / "state_timeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output_dir / "last_seen_states.json").write_text("[]", encoding="utf-8")
        return summary

    def zone_stage(*, timeline_path, summary_path, config_path, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        timelines = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        (output_dir / "zone_change_events.json").write_text("[]", encoding="utf-8")
        (output_dir / "zone_change_event_summary.json").write_text(
            json.dumps(
                {
                    "p2c_status": "PASS",
                    "video_id": timelines[0]["video_id"],
                    "input_p2b_status": summary["p2b_status"],
                    "input_artifact": Path(timeline_path).name,
                    "input_artifact_sha256": (
                        "different-timeline-hash"
                        if mismatched_p2c_input
                        else hashlib.sha256(Path(timeline_path).read_bytes()).hexdigest()
                    ),
                    "config_file": str(config_path),
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "zone_change_events.txt").write_text("no zone transitions\n", encoding="utf-8")
        return {"p2c_status": "PASS", "event_count": 0}

    def movement_stage(
        *, timeline_path, zone_events_path, consolidated_memory, persistent_entities,
        entity_summary, config_path, output_dir, motion_observations_path=None,
        motion_config_path=None, video_path=None, **_kwargs
    ):
        output_dir.mkdir(parents=True)
        event_records = []
        if supported_move_event:
            event_records = [
                {
                    "event_id": "move-object:synthetic:obj:cake:0001:1->2",
                    "event_type": "MOVE_OBJECT",
                    "start_time": 0.4,
                    "end_time": 0.6,
                    "person_ids": [],
                    "object_ids": ["obj:cake:0001"],
                    "zone_before": "left_region",
                    "zone_after": "right_region",
                    "evidence": [
                        {
                            "video_id": f"sha256:{video_hash}",
                            "timestamp": 0.4,
                            "frame_indices": [7],
                            "segment_ref": "synthetic#from",
                        },
                        {
                            "video_id": f"sha256:{video_hash}",
                            "timestamp": 0.6,
                            "frame_indices": [7],
                            "segment_ref": "synthetic#to",
                        },
                    ],
                    "source_segment_ids": ["synthetic#segment=1", "synthetic#segment=2"],
                }
            ]
        (output_dir / "move_object_events.json").write_text(json.dumps(event_records), encoding="utf-8")
        (output_dir / "move_object_evidence.json").write_text("[]", encoding="utf-8")
        summary = {
            "p2d_status": "INVALID" if invalid_p2d_status else ("PASS" if supported_move_event else "NO_ACCEPTED_EVENTS"),
            "video_id": f"sha256:{video_hash}",
            "move_object_event_count": len(event_records),
            "input_artifacts": {
                name: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for name, path in {
                    "timeline": timeline_path,
                    "zone_events": zone_events_path,
                    "consolidated_memory": consolidated_memory,
                    "persistent_entities": persistent_entities,
                    "entity_summary": entity_summary,
                    "motion_observations": motion_observations_path,
                    "config": config_path,
                    "motion_config": motion_config_path,
                    "source_video": video_path,
                }.items()
            },
        }
        (output_dir / "move_object_event_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(runner.analyze, "run", perception_stage)
    monkeypatch.setattr(runner.p2a_r1, "run_p2a_r1", association_stage)
    monkeypatch.setattr(runner.p2b, "run", timeline_stage)
    monkeypatch.setattr(runner.p2c, "run", zone_stage)
    monkeypatch.setattr(runner.p2d, "run", movement_stage)
    rendered = []

    def render(video_path, observation, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mock-rendered-evidence")
        rendered.append((video_path, observation, output_path))
        return output_path

    monkeypatch.setattr(runner, "render_evidence_frame", render)
    return video_path, model_path, video_hash, video_id, rendered


def test_pipeline_resolves_cake_evidence_and_keeps_event_state_separate(monkeypatch, tmp_path: Path) -> None:
    video, model, video_hash, video_id, rendered = _setup_mock_stages(monkeypatch, tmp_path)

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )

    assert result["run_status"] == "COMPLETED"
    assert result["video"]["sha256"] == video_hash
    assert result["video"]["video_id"] == video_id
    assert result["target_selection"]["status"] == "RESOLVED"
    assert result["target_selection"]["resolved_entity_id"] == "obj:cake:0001"
    assert result["event_status"] == "NO_SUPPORTED_EVENT"
    assert result["identity_review_status"] == "PENDING_HUMAN_REVIEW"
    assert result["query_result"]["status"] == "INSUFFICIENT_TRUSTED_EVIDENCE"
    assert result["query_result"]["automatic_candidate_result"]["status"] == "TARGET_LAST_SEEN"
    assert result["query_result"]["evidence_role"] == "AUTOMATIC_CANDIDATE"
    assert result["query_result"]["evidence"][0]["video_path"] == str(video.resolve())
    assert result["query_result"]["evidence"][0]["frame_index"] == 7
    assert result["query_result"]["evidence"][0]["state_or_event_id"]
    assert Path(result["query_result"]["evidence"][0]["evidence_path"]).is_file()
    assert rendered and rendered[0][0] == video.resolve()
    assert all(Path(path).is_relative_to(Path(result["run_dir"])) for path in result["query_result"]["evidence_paths"])


def test_pipeline_returns_ambiguous_target_with_candidate_metadata(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, entity_count=2
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )

    assert result["run_status"] == "COMPLETED"
    assert result["target_selection"]["status"] == "AMBIGUOUS_TARGET"
    assert [item["persistent_id"] for item in result["target_selection"]["candidates"]] == [
        "obj:cake:0001", "obj:cake:0002"
    ]
    assert result["query_result"]["status"] == "AMBIGUOUS_TARGET"


def test_explicit_entity_id_selects_candidate_without_promoting_identity_review(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, entity_count=2
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        target_entity_id="obj:cake:0002",
        question="Where is the cake?",
    )

    assert result["target_selection"]["status"] == "RESOLVED"
    assert result["target_selection"]["resolved_entity_id"] == "obj:cake:0002"
    assert result["query_result"]["persistent_id"] == "obj:cake:0002"
    assert result["identity_review_status"] == "PENDING_HUMAN_REVIEW"
    assert result["query_result"]["identity_review_status"] == "PENDING_HUMAN_REVIEW"


def test_pipeline_rejects_identity_review_status_promotion(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    from recallvision.pipeline import runner

    original_run = runner.p2b.run

    def promoted_review_status(*args, **kwargs):
        summary = original_run(*args, **kwargs)
        summary["identity_review_status"] = "PASS_HUMAN_REVIEW"
        summary_path = kwargs["output_dir"] / "state_timeline_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(runner.p2b, "run", promoted_review_status)

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "IDENTITY_STATUS_MISMATCH"
    assert manifest["stages"][-1]["name"] == "P2B"
    assert manifest["stages"][-1]["status"] == "FAILED"


def test_no_supported_event_is_a_completed_pipeline_outcome(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="When was the cake moved?",
    )

    assert result["run_status"] == "COMPLETED"
    assert result["event_status"] == "NO_SUPPORTED_EVENT"
    assert result["query_result"]["status"] == "INSUFFICIENT_TRUSTED_EVIDENCE"
    assert result["query_result"]["target_event_status"] == "INSUFFICIENT_TRUSTED_EVIDENCE"
    assert result["query_result"]["automatic_candidate_result"]["status"] == "NO_SUPPORTED_EVENT"


def test_supported_move_event_is_queryable_with_event_evidence(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, rendered = _setup_mock_stages(
        monkeypatch, tmp_path, supported_move_event=True
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="When was the cake moved?",
    )

    assert result["run_status"] == "COMPLETED"
    assert result["event_status"] == "SUPPORTED_EVENT"
    assert result["query_result"]["status"] == "EVENT_REQUIRES_REVIEW"
    assert result["query_result"]["automatic_candidate_result"]["status"] == "MOVEMENT_FOUND"
    assert result["query_result"]["automatic_candidate_result"]["event_ids"][0].startswith("move-object:synthetic:")
    assert result["query_result"]["evidence_role"] == "AUTOMATIC_CANDIDATE"
    assert rendered


def test_query_exposes_target_event_status_separately_from_legacy_global_status(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, entity_count=2, supported_move_event=True
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        target_entity_id="obj:cake:0002",
        question="When was the cake moved?",
    )

    query = result["query_result"]
    assert result["event_status"] == "SUPPORTED_EVENT"
    assert query["status"] == "INSUFFICIENT_TRUSTED_EVIDENCE"
    # Preserve the old field for existing readers while making its scope explicit.
    assert query["event_status"] == "SUPPORTED_EVENT"
    assert query["global_event_status"] == "SUPPORTED_EVENT"
    assert query["target_event_status"] == "INSUFFICIENT_TRUSTED_EVIDENCE"
    assert query["target_event_detection_status"] == "NO_SUPPORTED_EVENT"
    assert query["event_status_scope"] == "GLOBAL_LEGACY"


def test_unsupported_target_class_is_target_not_found_not_a_positive_answer(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="croissant",
        question="Where is the croissant?",
    )

    assert result["run_status"] == "COMPLETED"
    assert result["target_selection"]["status"] == "TARGET_NOT_FOUND"
    assert result["target_selection"]["resolution_status"] == "DETECTOR_VOCABULARY_UNSUPPORTED"
    assert result["query_result"]["status"] == "TARGET_NOT_FOUND"
    assert result["query_result"]["evidence"] == []


def test_invalid_p2d_status_does_not_become_a_completed_run(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, invalid_p2d_status=True
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "P2D_NOT_COMPLETED"
    assert manifest["stages"][-1]["name"] == "P2D"
    assert manifest["stages"][-1]["status"] == "FAILED"


def test_pipeline_manifest_records_stage_artifacts_and_source_hashes(monkeypatch, tmp_path: Path) -> None:
    video, model, video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )
    manifest_path = Path(result["run_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["run_id"]
    assert manifest["video"]["sha256"] == video_hash
    assert manifest["git"]["head"] == "test-head"
    assert {stage["name"] for stage in manifest["stages"]} == {
        "P1B", "P2A-R1", "P2B", "P2C", "P2D", "QUERY"
    }
    assert manifest["configs"]["p2a"]["sha256"]
    assert manifest["query"]["question"] == "Where is the cake?"
    assert manifest["query_result"] == manifest["query_results"][0]
    assert len(manifest["query_results"]) == 1
    assert manifest["query"]["results"] == [manifest["query"]["result"]]
    assert all(
        artifact["path"].startswith(str(Path(result["run_dir"])))
        for stage in manifest["stages"]
        for artifact in stage["outputs"]
    )


def test_pipeline_marks_source_mismatch_partial_and_preserves_manifest(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, mismatched_p2a_hash=True
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "SOURCE_MISMATCH"
    assert manifest["stages"][-1]["status"] == "FAILED"
    assert not (Path(result["run_dir"]) / "p2b").exists()


def test_pipeline_rejects_p2c_summary_from_another_timeline(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(
        monkeypatch, tmp_path, mismatched_p2c_input=True
    )

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "SOURCE_MISMATCH"
    assert manifest["stages"][-1]["name"] == "P2C"
    assert manifest["stages"][-1]["status"] == "FAILED"


def test_pipeline_does_not_clobber_an_existing_run_directory(monkeypatch, tmp_path: Path) -> None:
    video, model, video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    run_id = "11111111-1111-4111-8111-111111111111"
    run_dir = tmp_path / "runs" / video_hash[:16] / run_id
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    from recallvision.pipeline import runner

    monkeypatch.setattr(runner.uuid, "uuid4", lambda: run_id)

    with pytest.raises(FileExistsError):
        run_pipeline(
            video_path=video,
            output_root=tmp_path / "runs",
            model_path=model,
            target_class="cake",
            question="Where is the cake?",
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_missing_local_model_fails_preflight_without_starting_perception(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    model.unlink()

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "FAILED"
    assert manifest["failure"]["type"] == "MODEL_WEIGHTS_NOT_FOUND"
    assert manifest["stages"] == []


def test_keyboard_interrupt_records_partial_run_and_interrupted_stage(monkeypatch, tmp_path: Path) -> None:
    video, model, video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    from recallvision.pipeline import runner

    monkeypatch.setattr(runner.p2b, "run", lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            video_path=video,
            output_root=tmp_path / "runs",
            model_path=model,
            target_class="cake",
            question="Where is the cake?",
        )

    run_dirs = list((tmp_path / "runs" / video_hash[:16]).iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "INTERRUPTED"
    assert manifest["stages"][-1]["name"] == "P2B"
    assert manifest["stages"][-1]["status"] == "INTERRUPTED"
    assert (run_dirs[0] / "p2a_r1/entity_consolidation_summary.json").is_file()


def test_empty_perception_is_partial_and_keeps_manifest(monkeypatch, tmp_path: Path) -> None:
    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    from recallvision.pipeline import runner

    real_run = runner.analyze.run

    def no_detection_run(*, video_path, output_dir, model_name, video_id, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "observations.jsonl").write_text("", encoding="utf-8")
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                    "video_path": str(video_path),
                    "duration": 1.0,
                    "model_name": str(model_name),
                    "model_weights_path": str(model_name),
                    "model_weights_sha256": hashlib.sha256(Path(model_name).read_bytes()).hexdigest(),
                    "perception_status": "COMPLETED_NO_OBSERVATIONS",
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "analysis_summary.json").write_text(
            json.dumps({"total_observations": 0, "observations_by_class": {}}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(runner.analyze, "run", no_detection_run)
    assert real_run is not None

    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        question="Where is the cake?",
    )

    manifest = json.loads((Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "PARTIAL"
    assert manifest["failure"]["type"] == "NO_OBSERVATIONS"
    assert [stage["name"] for stage in manifest["stages"]] == ["P1B"]



def test_pipeline_saves_each_requested_query_and_preserves_legacy_single_result(monkeypatch, tmp_path: Path) -> None:
    from copy import deepcopy
    from recallvision.pipeline import runner

    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    original_write_manifest = runner._write_manifest
    initial_manifests = []

    def capture_manifest(path, manifest):
        if manifest.get("run_status") == "RUNNING" and not manifest.get("stages"):
            initial_manifests.append(deepcopy(manifest))
        original_write_manifest(path, manifest)

    monkeypatch.setattr(runner, "_write_manifest", capture_manifest)
    questions = ["Where is the cake?", "When was the cake moved?"]
    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        questions=questions,
    )

    assert result["run_status"] == "COMPLETED"
    assert len(result["query_results"]) == 2
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["query"]["questions"] == questions
    assert manifest["query"]["question"] == questions[-1]  # legacy scalar retains last-query semantics
    assert len(manifest["query"]["results"]) == 2
    assert [item["question"] for item in manifest["query_results"]] == questions
    assert manifest["query_result"] == manifest["query_results"][-1]
    query_stage = next(item for item in manifest["stages"] if item["name"] == "QUERY")
    assert len(query_stage["outputs"]) == 2
    result_paths = [Path(item["path"]) for item in query_stage["outputs"]]
    assert result_paths[0] != result_paths[1]
    assert all(path.is_file() for path in result_paths)
    assert [json.loads(path.read_text())["question"] for path in result_paths] == questions


def test_pipeline_plans_and_hashes_all_structured_p2a_outputs(monkeypatch, tmp_path: Path) -> None:
    from copy import deepcopy
    from recallvision.pipeline import runner

    video, model, _video_hash, _video_id, _rendered = _setup_mock_stages(monkeypatch, tmp_path)
    original_write_manifest = runner._write_manifest
    initial_manifests = []

    def capture_manifest(path, manifest):
        if manifest.get("run_status") == "RUNNING" and not manifest.get("stages"):
            initial_manifests.append(deepcopy(manifest))
        original_write_manifest(path, manifest)

    monkeypatch.setattr(runner, "_write_manifest", capture_manifest)
    result = run_pipeline(
        video_path=video,
        output_root=tmp_path / "runs",
        model_path=model,
        target_class="cake",
        question="Where is the cake?",
    )
    expected_names = {
        "persistent_entities.json",
        "observations_consolidated.jsonl",
        "entity_consolidation_summary.json",
        "duplicate_candidates.json",
        "duplicate_edges.json",
        "duplicate_groups.json",
        "continuation_candidates.json",
        "continuation_edges.json",
        "entity_conflicts.json",
        "track_to_persistent_mapping.json",
        "tracklet_summaries.json",
        "observations_with_persistent_ids.jsonl",
    }
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert initial_manifests
    initial_plan = initial_manifests[0]["planned_artifacts"]["P2A-R1"]
    assert {Path(item["path"]).name for item in initial_plan} == expected_names
    assert all(item["status"] == "PLANNED" and item["sha256"] is None for item in initial_plan)
    final_plan = manifest["planned_artifacts"]["P2A-R1"]
    assert {Path(item["path"]).name for item in final_plan} == expected_names
    assert all(item["status"] == "RECORDED" and item["sha256"] for item in final_plan)
    p2a_stage = next(item for item in manifest["stages"] if item["name"] == "P2A-R1")
    assert {Path(item["path"]).name for item in p2a_stage["outputs"]} == expected_names
    assert all(item["sha256"] for item in p2a_stage["outputs"])
