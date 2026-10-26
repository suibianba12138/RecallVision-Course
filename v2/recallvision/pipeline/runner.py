"""Orchestrate one isolated video run through the existing RecallVision stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from recallvision.cli import analyze, p2a_r1, p2b, p2c, p2d
from recallvision.events.schema import EventRecord
from recallvision.memory.schema import Observation
from recallvision.memory.store import ObservationMemoryStore
from recallvision.perception.ultralytics_backend import read_video_metadata
from recallvision.query.dispatch import dispatch_query
from recallvision.query.parser import QueryParseError, parse_query
from recallvision.query.schema import Query
from recallvision.query.target_resolution import (
    TargetResolutionStatus,
    build_target_reference,
    candidates_from_timelines,
    resolve_query_target,
)
from recallvision.visualization.evidence import render_evidence_frame


class _PipelineFailure(RuntimeError):
    def __init__(self, failure_type: str, message: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type


def run_pipeline(
    *,
    video_path: str | Path,
    output_root: str | Path,
    target_class: str | None = None,
    target_entity_id: str | None = None,
    question: str | None = None,
    questions: Sequence[str] | None = None,
    model_path: str | Path = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    source_url: str | None = None,
    config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Run P1B→P2A-R1→P2B→P2C→P2D→Query into a unique, no-clobber directory."""

    normalized_questions = _normalize_questions(question, questions)
    legacy_question = normalized_questions[-1] if normalized_questions else None
    source_video = Path(video_path).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"input video does not exist or is not a file: {source_video}")
    video_sha256 = sha256_file(source_video)
    video_id = f"sha256:{video_sha256}"
    run_id = str(uuid.uuid4())
    output_base = Path(output_root).expanduser().resolve()
    run_dir = output_base / video_sha256[:16] / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    paths = _config_paths(config_paths)
    p2a_dir = run_dir / "p2a_r1"
    p2a_outputs = [
        p2a_dir / name
        for name in (
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
        )
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_status": "RUNNING",
        "started_at": _utc_now(),
        "finished_at": None,
        "video": {
            "path": str(source_video),
            "sha256": video_sha256,
            "video_id": video_id,
            "metadata": None,
        },
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
        },
        "model": {
            "requested_path": str(model_path),
            "resolved_path": None,
            "sha256": None,
            "ultralytics_version": _package_version("ultralytics"),
            "tracker": tracker,
            "confidence_threshold": confidence_threshold,
            "imgsz": imgsz,
            "device": device,
        },
        "configs": {name: _file_record(path) for name, path in paths.items()},
        "stages": [],
        "planned_artifacts": {
            "P2A-R1": [
                {"name": path.name, "path": str(path), "sha256": None, "status": "PLANNED"}
                for path in p2a_outputs
            ]
        },
        "target_selection": {"status": "NOT_REQUESTED", "candidates": []},
        "identity_review_status": None,
        "event_status": None,
        "query": {
            "requested": bool(normalized_questions),
            "question": legacy_question,
            "result": None,
            "questions": list(normalized_questions),
            "results": [],
        },
        "query_result": None,
        "query_results": [],
        "failure": None,
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_manifest(manifest_path, manifest)

    try:
        _validate_configs(paths)
        resolved_model = _resolve_local_file(model_path)
        manifest["model"]["resolved_path"] = str(resolved_model)
        manifest["model"]["sha256"] = sha256_file(resolved_model)
        manifest["video"]["metadata"] = read_video_metadata(source_video)
        _write_manifest(manifest_path, manifest)
    except KeyboardInterrupt:
        _finish_failure(
            manifest,
            manifest_path,
            stage="PREFLIGHT",
            failure_type="INTERRUPTED",
            explanation="Pipeline interrupted by user during preflight.",
        )
        raise
    except Exception as error:
        _finish_failure(
            manifest,
            manifest_path,
            stage="PREFLIGHT",
            failure_type=("MODEL_WEIGHTS_NOT_FOUND" if isinstance(error, FileNotFoundError) and "model" in str(error).lower() else _failure_type(error)),
            explanation=str(error),
        )
        return _public_result(manifest)

    def execute_stage(
        name: str,
        stage_dir_name: str,
        *,
        inputs: Sequence[Path],
        outputs: Sequence[Path],
        action: Callable[[], Any],
        validate: Callable[[Any], None] | None = None,
    ) -> Any:
        entry = {
            "name": name,
            "status": "RUNNING",
            "started_at": _utc_now(),
            "finished_at": None,
            "inputs": [_artifact_record(path) for path in inputs],
            "outputs": [],
            "failure": None,
        }
        manifest["stages"].append(entry)
        _write_manifest(manifest_path, manifest)
        try:
            value = action()
            if validate is not None:
                validate(value)
            missing = [str(path) for path in outputs if not path.is_file()]
            if missing:
                raise _PipelineFailure("MISSING_STAGE_OUTPUT", f"{name} did not create: {', '.join(missing)}")
            entry["outputs"] = [_artifact_record(path) for path in outputs]
            _record_planned_artifacts(manifest, name)
            entry["status"] = "COMPLETED"
            entry["finished_at"] = _utc_now()
            _write_manifest(manifest_path, manifest)
            return value
        except KeyboardInterrupt:
            entry["status"] = "INTERRUPTED"
            entry["finished_at"] = _utc_now()
            entry["outputs"] = [_artifact_record(path) for path in outputs if path.is_file()]
            _record_planned_artifacts(manifest, name)
            entry["failure"] = {
                "type": "INTERRUPTED",
                "explanation": "Pipeline interrupted by user.",
            }
            _finish_failure(
                manifest,
                manifest_path,
                stage=name,
                failure_type="INTERRUPTED",
                explanation="Pipeline interrupted by user.",
            )
            raise
        except Exception as error:
            entry["status"] = "FAILED"
            entry["finished_at"] = _utc_now()
            entry["outputs"] = [_artifact_record(path) for path in outputs if path.is_file()]
            _record_planned_artifacts(manifest, name)
            entry["failure"] = {"type": _failure_type(error), "explanation": str(error)}
            _finish_failure(
                manifest,
                manifest_path,
                stage=name,
                failure_type=_failure_type(error),
                explanation=str(error),
            )
            raise _StopPipeline from error

    p1b_dir = run_dir / "p1b"
    p1b_outputs = [
        p1b_dir / "observations.jsonl",
        p1b_dir / "run_metadata.json",
        p1b_dir / "analysis_summary.json",
    ]
    try:
        execute_stage(
            "P1B",
            "p1b",
            inputs=[source_video, resolved_model],
            outputs=p1b_outputs,
            action=lambda: analyze.run(
                video_path=source_video,
                output_dir=p1b_dir,
                model_name=str(resolved_model),
                tracker=tracker,
                confidence_threshold=confidence_threshold,
                imgsz=imgsz,
                device=device,
                video_id=video_id,
                source_url=source_url,
                emit_json=False,
            ),
            validate=lambda value: _validate_perception(
                p1b_outputs[0], p1b_outputs[1], video_id, video_sha256, source_video, value
            ),
        )
        perception_metadata = _read_json(p1b_outputs[1], "P1B run metadata")
        manifest["video"]["metadata"] = {
            "fps": perception_metadata.get("video_fps"),
            "width": perception_metadata.get("video_width"),
            "height": perception_metadata.get("video_height"),
            "frame_count": perception_metadata.get("frame_count"),
            "duration": perception_metadata.get("duration"),
        }
        manifest["perception_status"] = perception_metadata.get("perception_status")
        _write_manifest(manifest_path, manifest)
        observations = ObservationMemoryStore(p1b_outputs[0]).load_all()
        if not observations:
            _skip_after_no_observations(manifest, manifest_path)
            return _public_result(manifest)
    except _StopPipeline:
        return _public_result(manifest)
    except Exception as error:
        _finish_failure(manifest, manifest_path, "P1B", _failure_type(error), str(error))
        return _public_result(manifest)

    try:
        execute_stage(
            "P2A-R1",
            "p2a_r1",
            inputs=[p1b_outputs[0], source_video, paths["p2a"]],
            outputs=p2a_outputs,
            action=lambda: p2a_r1.run_p2a_r1(
                input_memory=p1b_outputs[0],
                input_video=source_video,
                output_dir=p2a_dir,
                config_path=paths["p2a"],
                render_diagnostics=False,
            ),
            validate=lambda _value: _validate_p2a(
                p2a_outputs[0],
                p2a_outputs[1],
                p2a_outputs[2],
                video_id,
                video_sha256,
                p1b_outputs[0],
                source_video,
            ),
        )
        p2a_summary = _read_json(p2a_outputs[2], "P2A-R1 summary")
        manifest["identity_review_status"] = p2a_summary.get("p2a_r1_identity_validation_status")
        _write_manifest(manifest_path, manifest)
    except _StopPipeline:
        return _public_result(manifest)

    p2b_dir = run_dir / "p2b"
    p2b_outputs = [
        p2b_dir / "object_state_timelines.json",
        p2b_dir / "last_seen_states.json",
        p2b_dir / "state_timeline_summary.json",
    ]
    try:
        execute_stage(
            "P2B",
            "p2b",
            inputs=[p2a_outputs[0], p2a_outputs[1], p2a_outputs[2], paths["p2b"]],
            outputs=p2b_outputs,
            action=lambda: p2b.run(
                consolidated_memory=p2a_outputs[1],
                persistent_entities=p2a_outputs[0],
                entity_summary=p2a_outputs[2],
                config_path=paths["p2b"],
                output_dir=p2b_dir,
                target_class=target_class,
                target_entity_id=target_entity_id,
            ),
            validate=lambda _value: _validate_p2b(
                p2b_outputs[0],
                p2b_outputs[2],
                video_id,
                video_sha256,
                manifest["identity_review_status"],
            ),
        )
        timelines = p2c._load_timelines(p2b_outputs[0])
    except _StopPipeline:
        return _public_result(manifest)
    except Exception as error:
        _finish_failure(manifest, manifest_path, "P2B", _failure_type(error), str(error))
        return _public_result(manifest)

    p2c_dir = run_dir / "p2c"
    p2c_outputs = [
        p2c_dir / "zone_change_events.json",
        p2c_dir / "zone_change_event_summary.json",
        p2c_dir / "zone_change_events.txt",
    ]
    try:
        execute_stage(
            "P2C",
            "p2c",
            inputs=[p2b_outputs[0], p2b_outputs[2], paths["p2c"]],
            outputs=p2c_outputs,
            action=lambda: p2c.run(
                timeline_path=p2b_outputs[0],
                summary_path=p2b_outputs[2],
                config_path=paths["p2c"],
                output_dir=p2c_dir,
            ),
            validate=lambda _value: _validate_p2c(
                p2c_outputs[0], p2c_outputs[1], video_id, p2b_outputs[0]
            ),
        )
        p2c_events = p2d._load_events(p2c_outputs[0])
    except _StopPipeline:
        return _public_result(manifest)
    except Exception as error:
        _finish_failure(manifest, manifest_path, "P2C", _failure_type(error), str(error))
        return _public_result(manifest)

    p2d_dir = run_dir / "p2d"
    p2d_outputs = [
        p2d_dir / "move_object_events.json",
        p2d_dir / "move_object_evidence.json",
        p2d_dir / "move_object_event_summary.json",
    ]
    p2d_inputs = [
        p2b_outputs[0],
        p2c_outputs[0],
        p2a_outputs[1],
        p2a_outputs[0],
        p2a_outputs[2],
        p2a_outputs[11],
        paths["p2d"],
        paths["motion"],
        source_video,
    ]
    try:
        execute_stage(
            "P2D",
            "p2d",
            inputs=p2d_inputs,
            outputs=p2d_outputs,
            action=lambda: p2d.run(
                timeline_path=p2b_outputs[0],
                zone_events_path=p2c_outputs[0],
                consolidated_memory=p2a_outputs[1],
                persistent_entities=p2a_outputs[0],
                entity_summary=p2a_outputs[2],
                config_path=paths["p2d"],
                output_dir=p2d_dir,
                motion_observations_path=p2a_outputs[11],
                motion_config_path=paths["motion"],
                video_path=source_video,
            ),
            validate=lambda _value: _validate_p2d(p2d_outputs[0], p2d_outputs[2], video_id, p2d_inputs),
        )
        move_events = p2d._load_events(p2d_outputs[0])
        manifest["event_status"] = "SUPPORTED_EVENT" if move_events else "NO_SUPPORTED_EVENT"
        _write_manifest(manifest_path, manifest)
    except _StopPipeline:
        return _public_result(manifest)
    except Exception as error:
        _finish_failure(manifest, manifest_path, "P2D", _failure_type(error), str(error))
        return _public_result(manifest)

    if normalized_questions:
        query_dir = run_dir / "query"
        if len(normalized_questions) == 1:
            query_output_dirs = [query_dir]
        else:
            query_output_dirs = [
                query_dir / f"query-{index:03d}"
                for index in range(1, len(normalized_questions) + 1)
            ]
        query_outputs = [output_dir / "query_result.json" for output_dir in query_output_dirs]

        def run_queries() -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for index, (question_text, output_dir) in enumerate(
                zip(normalized_questions, query_output_dirs)
            ):
                record = _run_query(
                    question=question_text,
                    video_id=video_id,
                    target_class=target_class,
                    target_entity_id=target_entity_id,
                    timelines=timelines,
                    zone_events=p2c_events,
                    move_events=move_events,
                    observations=ObservationMemoryStore(p2a_outputs[1]).load_all(),
                    persistent_entities=_read_json_list(p2a_outputs[0], "persistent entities"),
                    video_end_timestamp=float(manifest["video"]["metadata"]["duration"]),
                    video_path=source_video,
                    video_sha256=video_sha256,
                    identity_review_status=manifest["identity_review_status"],
                    output_dir=output_dir,
                )
                records.append(record)
                manifest["query_results"] = list(records)
                manifest["query_result"] = record
                manifest["target_selection"] = record.get("target_selection", manifest["target_selection"])
                manifest["query"]["results"] = [str(path) for path in query_outputs[: index + 1]]
                manifest["query"]["result"] = str(query_outputs[index])
                _write_manifest(manifest_path, manifest)
            return records

        try:
            execute_stage(
                "QUERY",
                "query",
                inputs=[
                    p2a_outputs[1], p2a_outputs[0], p2b_outputs[0],
                    p2c_outputs[0], p2d_outputs[0], source_video,
                ],
                outputs=query_outputs,
                action=run_queries,
            )
        except _StopPipeline:
            return _public_result(manifest)
    else:
        target_selection = _resolve_target_without_question(
            timelines,
            video_id=video_id,
            target_class=target_class,
            target_entity_id=target_entity_id,
        )
        manifest["target_selection"] = target_selection
        manifest["query"]["status"] = "NOT_REQUESTED"
        manifest["query_results"] = []
        manifest["stages"].append(
            {
                "name": "QUERY",
                "status": "SKIPPED",
                "started_at": _utc_now(),
                "finished_at": _utc_now(),
                "inputs": [],
                "outputs": [],
                "failure": None,
            }
        )

    manifest["run_status"] = "COMPLETED"
    manifest["finished_at"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    return _public_result(manifest)


class _StopPipeline(Exception):
    """Internal control flow after execute_stage has recorded a failure."""


def _run_query(
    *,
    question: str,
    video_id: str,
    target_class: str | None,
    target_entity_id: str | None,
    timelines: Sequence[Any],
    zone_events: Sequence[EventRecord],
    move_events: Sequence[EventRecord],
    observations: Sequence[Observation],
    persistent_entities: Sequence[Mapping[str, Any]],
    video_end_timestamp: float,
    video_path: Path,
    video_sha256: str,
    identity_review_status: str | None,
    output_dir: Path,
    trust_evidence_available: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_id = f"query-{uuid.uuid4().hex[:12]}"
    try:
        query = parse_query(question, video_id=video_id, question_id=query_id)
    except QueryParseError as error:
        record = {
            "question_id": query_id,
            "question": question,
            "status": "UNSUPPORTED_QUERY",
            "answer": str(error),
            "target_selection": {"status": "NOT_REQUESTED", "candidates": []},
            "identity_review_status": identity_review_status,
            "evidence": [],
            "evidence_paths": [],
        }
        _write_json_exclusive(output_dir / "query_result.json", record)
        return record

    if target_class is not None:
        reference = build_target_reference(target_class)
        query = Query(
            question_id=query.question_id,
            video_id=query.video_id,
            question=query.question,
            query_type=query.query_type,
            target_entity_ids=query.target_entity_ids,
            target_class=reference.canonical_label,
            target_reference=reference,
            reference_event_id=query.reference_event_id,
        )
    if target_entity_id is not None:
        query = Query(
            question_id=query.question_id,
            video_id=query.video_id,
            question=query.question,
            query_type=query.query_type,
            target_entity_ids=[target_entity_id],
            target_class=query.target_class,
            target_reference=query.target_reference,
            reference_event_id=query.reference_event_id,
        )

    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    selection = _target_selection_record(resolution, timelines, persistent_entities)
    result = dispatch_query(
        query,
        persistent_entities=persistent_entities,
        timelines=timelines,
        events=move_events,
        video_end_timestamp=video_end_timestamp,
    )
    record = result.to_record() if hasattr(result, "to_record") else {"answer": str(result)}
    query_type = getattr(query.query_type, "value", str(query.query_type))
    global_event_status = "SUPPORTED_EVENT" if move_events else "NO_SUPPORTED_EVENT"
    movement_query = query_type in {"WHEN_OBJECT_MOVED", "WHERE_OBJECT_BEFORE"}
    if not movement_query:
        target_event_detection_status = "NOT_APPLICABLE"
        target_event_status = "NOT_APPLICABLE"
    elif resolution.resolved_entity_id is None:
        target_event_detection_status = "TARGET_NOT_RESOLVED"
        target_event_status = "TARGET_NOT_RESOLVED"
    else:
        target_has_event = any(
            resolution.resolved_entity_id in (event.object_ids or [])
            for event in move_events
        )
        target_event_detection_status = "SUPPORTED_EVENT" if target_has_event else "NO_SUPPORTED_EVENT"
        target_event_status = (
            target_event_detection_status
            if trust_evidence_available
            else "EVENT_REQUIRES_REVIEW" if target_has_event else "INSUFFICIENT_TRUSTED_EVIDENCE"
        )
    automatic_candidate_result = {
        "status": record.get("status"),
        "answer": record.get("answer"),
        "event_ids": [event.event_id for event in getattr(result, "events", ()) or ()],
    }
    record.update(
        {
            "question": question,
            "target_selection": selection,
            "identity_review_status": identity_review_status,
            # Keep the historical field intact; the explicit target field is the
            # authoritative status for a question about one selected object.
            "event_status": global_event_status,
            "global_event_status": global_event_status,
            "target_event_status": target_event_status,
            "target_event_detection_status": target_event_detection_status,
            "global_event_trust_status": "UNVERIFIED_CANDIDATE" if move_events and not trust_evidence_available else (
                "SUPPORTED_EVENT" if move_events else "NO_REVIEWED_EVENT_EVIDENCE" if trust_evidence_available else "NO_CANDIDATE_EVENT"
            ),
            "event_status_scope": "GLOBAL_LEGACY",
        }
    )
    evidence, evidence_status = _render_query_evidence(
        result,
        observations,
        video_path=video_path,
        video_sha256=video_sha256,
        video_id=video_id,
        output_dir=output_dir / "evidence",
    )
    record["evidence"] = evidence
    record["evidence_paths"] = [item["evidence_path"] for item in evidence]
    record["evidence_status"] = evidence_status
    if not evidence and _is_positive_query_result(result):
        record["handler_status"] = record.get("status")
        record["handler_answer"] = record.get("answer")
        record["status"] = "INSUFFICIENT_EVIDENCE"
        record["answer"] = "INSUFFICIENT_EVIDENCE: no matching source observation was found for the query evidence."
    if not trust_evidence_available:
        _apply_unreviewed_query_policy(
            record,
            query_type=query_type,
            selection=selection,
            target_event_status=target_event_status,
            automatic_candidate_result=automatic_candidate_result,
            has_candidate_events=bool(move_events),
        )
    _write_json_exclusive(output_dir / "query_result.json", record)
    return record


def _apply_unreviewed_query_policy(
    record: dict[str, Any],
    *,
    query_type: str,
    selection: dict[str, Any],
    target_event_status: str,
    automatic_candidate_result: dict[str, Any],
    has_candidate_events: bool,
) -> None:
    """Prevent raw automatic outputs from becoming trusted user answers."""
    status = str(record.get("status", ""))
    if status in {"AMBIGUOUS_TARGET", "TARGET_NOT_FOUND", "UNSUPPORTED_QUERY", "TARGET_NOT_RESOLVED"}:
        return
    if query_type == "WHERE_OBJECT" and status in {"TARGET_LAST_SEEN", "OK"}:
        record["automatic_candidate_result"] = automatic_candidate_result
        record["status"] = "INSUFFICIENT_TRUSTED_EVIDENCE"
        record["answer"] = "The detector found a possible last-seen location, but no exact frame has been reviewed for this object yet."
        record["evidence_role"] = "AUTOMATIC_CANDIDATE"
        record["target_event_status"] = "NOT_APPLICABLE"
        return
    if query_type in {"WHEN_OBJECT_MOVED", "WHERE_OBJECT_BEFORE"}:
        record["automatic_candidate_result"] = automatic_candidate_result
        if target_event_status == "EVENT_REQUIRES_REVIEW":
            record["status"] = "EVENT_REQUIRES_REVIEW"
            record["answer"] = "An automatic movement candidate exists for this object. Review its video boundary evidence before treating it as a confirmed event."
            record["evidence_role"] = "AUTOMATIC_CANDIDATE"
        else:
            record["status"] = "INSUFFICIENT_TRUSTED_EVIDENCE"
            record["answer"] = (
                "No movement candidate was detected for this selected object, but the available frames have not been reviewed enough to establish that no movement occurred."
                if not has_candidate_events or target_event_status == "INSUFFICIENT_TRUSTED_EVIDENCE"
                else "There is not enough explicitly reviewed video evidence to determine whether this object moved."
            )
            record["evidence_role"] = "UNVERIFIED_AUTOMATIC_RESULT"


def _render_query_evidence(
    result: Any,
    observations: Sequence[Observation],
    *,
    video_path: Path,
    video_sha256: str,
    video_id: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    seeds: list[tuple[str | None, int, float | None, str | None]] = []
    persistent_id = getattr(result, "persistent_id", None) or getattr(result, "target_entity_id", None)
    frame_index = getattr(result, "frame_index", None)
    timestamp = getattr(result, "last_seen_timestamp", None) or getattr(result, "timestamp", None)
    if isinstance(frame_index, int):
        evidence_ref = getattr(result, "evidence_ref", None)
        state_id = getattr(result, "from_source_segment_id", None) or getattr(
            evidence_ref, "segment_ref", None
        )
        seeds.append((persistent_id, frame_index, timestamp, state_id))
    evidence_refs = []
    evidence_ref = getattr(result, "evidence_ref", None)
    if evidence_ref is not None:
        evidence_refs.append(evidence_ref)
    evidence_refs.extend(getattr(result, "evidence", ()) or ())
    for event in getattr(result, "events", ()) or ():
        event_id = getattr(event, "event_id", None)
        event_entity = next(iter(getattr(event, "object_ids", ()) or ()), persistent_id)
        evidence_refs.extend((ref, event_id, event_entity) for ref in getattr(event, "evidence", ()) or ())
    for item in evidence_refs:
        event_id = None
        entity_id = persistent_id
        reference = item
        if isinstance(item, tuple):
            reference, event_id, entity_id = item
        if getattr(reference, "video_id", video_id) != video_id:
            raise _PipelineFailure("SOURCE_MISMATCH", "query evidence references a different video_id")
        frames = getattr(reference, "frame_indices", ()) or ()
        for frame in frames:
            seeds.append((entity_id, int(frame), getattr(reference, "timestamp", None), event_id))

    unique: list[tuple[str | None, int, float | None, str | None]] = []
    seen: set[tuple[str | None, int, str | None]] = set()
    for seed in seeds:
        key = (seed[0], seed[1], seed[3])
        if key not in seen:
            seen.add(key)
            unique.append(seed)

    if not unique:
        if _is_positive_query_result(result):
            return [], "INSUFFICIENT_EVIDENCE"
        return [], "NOT_AVAILABLE_FOR_RESULT_STATUS"

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_records = []
    for index, (entity_id, frame, evidence_timestamp, state_or_event_id) in enumerate(unique, start=1):
        candidates = [
            item for item in observations
            if item.video_id == video_id and item.frame_index == frame
            and (entity_id is None or item.persistent_id == entity_id)
        ]
        if not candidates:
            continue
        observation = candidates[0]
        image_path = output_dir / f"evidence_{index:03d}_frame_{frame:06d}.jpg"
        rendered_path = render_evidence_frame(video_path, observation, image_path)
        evidence_records.append(
            {
                "video_path": str(video_path),
                "video_sha256": video_sha256,
                "video_id": video_id,
                "timestamp": evidence_timestamp if evidence_timestamp is not None else observation.timestamp,
                "frame_index": observation.frame_index,
                "target_entity_id": observation.persistent_id,
                "class_name": observation.class_name,
                "state_or_event_id": state_or_event_id,
                "evidence_path": str(Path(rendered_path).resolve()),
            }
        )
    return evidence_records, "RENDERED" if evidence_records else "INSUFFICIENT_EVIDENCE"


def _is_positive_query_result(result: Any) -> bool:
    status = getattr(result, "status", None)
    value = status.value if hasattr(status, "value") else str(status)
    return value in {"TARGET_LAST_SEEN", "OK", "MOVEMENT_FOUND", "SUCCESS"}


def _target_selection_record(resolution: Any, timelines: Sequence[Any], entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapping = {str(item.get("persistent_id")): item for item in entities if item.get("persistent_id")}
    by_timeline = {item.persistent_id: item for item in timelines}
    candidates = []
    for persistent_id in resolution.candidate_entity_ids:
        timeline = by_timeline.get(persistent_id)
        entity = mapping.get(persistent_id, {})
        segments = list(getattr(timeline, "segments", ())) if timeline else []
        candidates.append(
            {
                "persistent_id": persistent_id,
                "class_name": timeline.class_name if timeline else entity.get("class_name"),
                "entity_type": entity.get("entity_type", "object"),
                "track_ids": entity.get("track_ids", entity.get("source_track_ids", [])),
                "timeline_segment_count": len(segments),
                "first_observed_timestamp": segments[0].start_time if segments else None,
                "last_observed_timestamp": segments[-1].end_time if segments else None,
                "first_observed_frame": segments[0].start_frame if segments else None,
                "last_observed_frame": segments[-1].end_frame if segments else None,
            }
        )
    mapped_status = {
        TargetResolutionStatus.RESOLVED: "RESOLVED",
        TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES: "AMBIGUOUS_TARGET",
    }.get(resolution.status, "TARGET_NOT_FOUND")
    return {
        "status": mapped_status,
        "resolution_status": resolution.status.value,
        "resolved_entity_id": resolution.resolved_entity_id,
        "candidate_ids": list(resolution.candidate_entity_ids),
        "candidates": candidates,
        "compatible_system_classes": list(resolution.compatible_system_classes),
    }


def _resolve_target_without_question(
    timelines: Sequence[Any], *, video_id: str, target_class: str | None, target_entity_id: str | None
) -> dict[str, Any]:
    if target_class is None and target_entity_id is None:
        all_candidates = candidates_from_timelines(timelines)
        return {
            "status": "NOT_REQUESTED",
            "candidate_ids": [item.persistent_id for item in all_candidates],
            "candidates": [
                {
                    "persistent_id": item.persistent_id,
                    "class_name": item.class_name,
                    "video_id": item.video_id,
                }
                for item in all_candidates
            ],
        }
    reference = build_target_reference(target_class) if target_class else None
    query = Query(
        question_id="target-selection",
        video_id=video_id,
        question="Target selection",
        query_type=parse_query("Where is the object?", video_id=video_id, question_id="target-selection").query_type,
        target_entity_ids=[target_entity_id] if target_entity_id else [],
        target_class=reference.canonical_label if reference else None,
        target_reference=reference,
    )
    resolution = resolve_query_target(query, candidates_from_timelines(timelines))
    return _target_selection_record(resolution, timelines, ())


def _validate_perception(
    observations_path: Path,
    metadata_path: Path,
    video_id: str,
    video_sha256: str,
    video_path: Path,
    exit_code: Any,
) -> None:
    metadata = _read_json(metadata_path, "P1B run metadata")
    if exit_code not in (None, 0):
        raise _PipelineFailure("P1B_EXIT_STATUS", f"P1B returned exit code {exit_code}")
    if (
        metadata.get("video_id") != video_id
        or metadata.get("video_sha256") != video_sha256
        or Path(str(metadata.get("video_path", ""))).resolve() != video_path.resolve()
    ):
        raise _PipelineFailure("SOURCE_MISMATCH", "P1B metadata does not identify the input video")
    observations = ObservationMemoryStore(observations_path).load_all()
    if any(item.video_id != video_id for item in observations):
        raise _PipelineFailure("SOURCE_MISMATCH", "P1B observations contain a different video_id")


def _validate_p2a(
    entities_path: Path,
    memory_path: Path,
    summary_path: Path,
    video_id: str,
    video_sha256: str,
    input_memory_path: Path,
    input_video_path: Path,
) -> None:
    summary = _read_json(summary_path, "P2A-R1 summary")
    if summary.get("p2a_r1_pipeline_status") != "PASS":
        raise _PipelineFailure("P2A_NOT_COMPLETED", "P2A-R1 summary does not report PASS")
    if (
        summary.get("video_sha256") != video_sha256
        or summary.get("video_id") != video_id
        or Path(str(summary.get("input_video", ""))).resolve() != input_video_path.resolve()
        or Path(str(summary.get("input_memory", ""))).resolve() != input_memory_path.resolve()
    ):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2A-R1 summary does not identify the input video")
    entities = _read_json_list(entities_path, "persistent_entities")
    entity_ids = {item.get("persistent_id") for item in entities}
    observations = ObservationMemoryStore(memory_path).load_all()
    if any(item.video_id != video_id for item in observations):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2A-R1 memory contains a different video_id")
    if any(item.persistent_id not in entity_ids for item in observations):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2A-R1 memory contains an entity absent from its entity artifact")


def _validate_p2b(
    timeline_path: Path,
    summary_path: Path,
    video_id: str,
    video_sha256: str,
    identity_review_status: str | None,
) -> None:
    summary = _read_json(summary_path, "P2B summary")
    if summary.get("p2b_status") != "PASS":
        raise _PipelineFailure("P2B_NOT_COMPLETED", "P2B summary does not report PASS")
    authoritative = summary.get("authoritative_inputs", {})
    if summary.get("video_id") != video_id or authoritative.get("video_sha256") != video_sha256:
        raise _PipelineFailure("SOURCE_MISMATCH", "P2B timeline provenance does not identify the input video")
    if summary.get("identity_review_status") != identity_review_status:
        raise _PipelineFailure("IDENTITY_STATUS_MISMATCH", "P2B changed the P2A identity review status")
    timelines = p2c._load_timelines(timeline_path)
    if any(item.video_id != video_id for item in timelines):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2B timelines contain a different video_id")


def _validate_p2c(events_path: Path, summary_path: Path, video_id: str, timeline_path: Path) -> None:
    summary = _read_json(summary_path, "P2C summary")
    if summary.get("p2c_status") != "PASS" or summary.get("video_id") != video_id:
        raise _PipelineFailure("SOURCE_MISMATCH", "P2C output does not match the current video")
    if (
        summary.get("input_artifact") != timeline_path.name
        or summary.get("input_artifact_sha256") != sha256_file(timeline_path)
    ):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2C summary does not reference the current P2B timeline")
    events = p2d._load_events(events_path)
    if any(_event_video_id(event) != video_id for event in events):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2C events contain a different video_id")


def _validate_p2d(events_path: Path, summary_path: Path, video_id: str, input_paths: Sequence[Path]) -> None:
    summary = _read_json(summary_path, "P2D summary")
    events = p2d._load_events(events_path)
    if summary.get("video_id") != video_id:
        raise _PipelineFailure("SOURCE_MISMATCH", "P2D summary does not identify the current video")
    if summary.get("p2d_status") not in {"PASS", "NO_ACCEPTED_EVENTS"}:
        raise _PipelineFailure("P2D_NOT_COMPLETED", "P2D summary has no supported completion status")
    if summary.get("move_object_event_count") != len(events):
        raise _PipelineFailure("P2D_ARTIFACT_MISMATCH", "P2D event count does not match its event artifact")
    if any(_event_video_id(event) != video_id for event in events):
        raise _PipelineFailure("SOURCE_MISMATCH", "P2D events contain a different video_id")
    declared = summary.get("input_artifacts", {})
    expected_paths = {path.resolve() for path in input_paths}
    declared_paths = {Path(record.get("path", "")).resolve() for record in declared.values()}
    if declared_paths != expected_paths:
        raise _PipelineFailure("SOURCE_MISMATCH", "P2D summary input artifact list is incomplete")
    for record in declared.values():
        path = Path(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise _PipelineFailure("SOURCE_MISMATCH", f"P2D input artifact checksum mismatch: {path}")


def _event_video_id(event: EventRecord) -> str | None:
    return event.evidence[0].video_id if event.evidence else None


def _config_paths(overrides: Mapping[str, str | Path] | None) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    defaults = {
        "p2a": root / "configs/p2a_r1_entity_consolidation.yaml",
        "p2b": root / "configs/p2b_state_timeline.yaml",
        "p2c": root / "configs/p2c_zone_transition_events.yaml",
        "p2d": root / "configs/p2d_move_object_events.yaml",
        "motion": root / "configs/p3b1_robust_motion.yaml",
    }
    if overrides:
        for key, value in overrides.items():
            if key not in defaults:
                raise ValueError(f"unknown pipeline config key: {key}")
            defaults[key] = Path(value).expanduser().resolve()
    return defaults


def _validate_configs(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("pipeline config file(s) not found: " + ", ".join(missing))


def _resolve_local_file(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, Path(__file__).resolve().parents[2] / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"local model weights not found: {value}; pass --model with an existing local checkpoint; downloads are disabled"
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _skip_after_no_observations(manifest: dict[str, Any], path: Path) -> None:
    manifest["run_status"] = "PARTIAL"
    manifest["finished_at"] = _utc_now()
    manifest["failure"] = {
        "stage": "P1B",
        "type": "NO_OBSERVATIONS",
        "explanation": "P1B completed without observations; identity, timeline, event, and query stages were not run.",
    }
    manifest["event_status"] = "NO_SUPPORTED_EVENT"
    _write_manifest(path, manifest)


def _finish_failure(
    manifest: dict[str, Any],
    path: Path,
    stage: str,
    failure_type: str,
    explanation: str,
) -> None:
    completed = any(item.get("status") == "COMPLETED" for item in manifest["stages"])
    manifest["run_status"] = "PARTIAL" if completed else "FAILED"
    manifest["finished_at"] = _utc_now()
    manifest["failure"] = {"stage": stage, "type": failure_type, "explanation": explanation}
    _write_manifest(path, manifest)


def _public_result(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "run_dir": manifest["run_dir"],
        "manifest_path": str(Path(manifest["run_dir"]) / "run_manifest.json"),
        "run_status": manifest["run_status"],
        "video": manifest["video"],
        "target_selection": manifest["target_selection"],
        "identity_review_status": manifest["identity_review_status"],
        "event_status": manifest["event_status"],
        "query_result": manifest["query_result"],
        "query_results": manifest.get("query_results", []),
        "failure": manifest["failure"],
        "manifest": manifest,
    }


def _normalize_questions(
    question: str | None, questions: Sequence[str] | None
) -> list[str]:
    if question is not None and questions is not None:
        raise ValueError("pass either question or questions, not both")
    if questions is None:
        values = [] if question is None or question == "" else [question]
    elif isinstance(questions, str):
        values = [questions]
    else:
        values = list(questions)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("every query question must be a non-empty string")
    return values


def _record_planned_artifacts(manifest: dict[str, Any], stage_name: str) -> None:
    for item in manifest.get("planned_artifacts", {}).get(stage_name, []):
        path = Path(item["path"])
        if path.is_file():
            item["sha256"] = sha256_file(path)
            item["status"] = "RECORDED"
        else:
            item["sha256"] = None
            item["status"] = "MISSING"


def _failure_type(error: Exception) -> str:
    return getattr(error, "failure_type", type(error).__name__.upper())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(*arguments: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
