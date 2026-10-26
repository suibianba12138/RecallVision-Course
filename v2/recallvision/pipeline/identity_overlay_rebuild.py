"""Recompute visual memory from frozen P2A observations and active identity overlay."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from recallvision.cli import p2b, p2c, p2d
from recallvision.memory.identity_overlay import IdentityOverlay, OverlayError
from recallvision.memory.store import ObservationMemoryStore
from recallvision.memory.trust_review import TrustReviewOverlay, TrustReviewError
from recallvision.pipeline.runner import _run_query, sha256_file
from recallvision.query.target_resolution import build_target_reference


class IdentityRebuildError(ValueError):
    """A rebuild could not be safely completed."""

    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.status = status
        self.details = details or {}
        super().__init__(f"{status}: {message}")


def rebuild_visual_memory(
    *,
    run_dir: str | Path,
    ledger_path: str | Path,
    output_root: str | Path,
    source_hash_manifest: str | Path | None = None,
    questions: Sequence[str] = (),
    target_class: str | None = None,
    target_entity_id: str | None = None,
    trust_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a new immutable overlay revision and recompute P2B/P2C/P2D/queries."""
    try:
        overlay = IdentityOverlay(run_dir, ledger_path, source_hash_manifest)
    except OverlayError:
        raise
    snapshot = overlay.snapshot()
    history = snapshot["history"]
    revision = snapshot["overlay_revision"]
    trust_overlay = None
    trust_snapshot = None
    if trust_ledger_path is not None:
        trust_overlay = TrustReviewOverlay(
            run_dir=run_dir,
            ledger_path=trust_ledger_path,
            identity_ledger_path=ledger_path,
            source_hash_manifest=source_hash_manifest,
        )
        trust_snapshot = trust_overlay.snapshot()
        if trust_snapshot["source_identity_overlay_revision"] != revision:
            raise IdentityRebuildError(
                "STALE_TRUST_REVIEW",
                "identity mapping changed after trust reviews; review the selected evidence again",
                {"trust_identity_revision": trust_snapshot["source_identity_overlay_revision"], "identity_revision": revision},
            )
    output_base = Path(output_root).expanduser().resolve()
    try:
        output_base.relative_to(overlay.run_dir)
    except ValueError:
        pass
    else:
        raise IdentityRebuildError(
            "OUTPUT_IN_SOURCE_RUN",
            "rebuild output must be outside the frozen source run",
            {"output_root": str(output_base), "source_run_dir": str(overlay.run_dir)},
        )
    output_base.mkdir(parents=True, exist_ok=True)
    revision_dir = output_base / f"revision-{revision:06d}"
    if revision_dir.exists():
        raise IdentityRebuildError("OUTPUT_EXISTS", "this overlay revision already has output", {"revision_dir": str(revision_dir)})
    if target_entity_id is not None and target_entity_id not in overlay.entity_by_id:
        raise IdentityRebuildError("ENTITY_NOT_FOUND", "selected target entity is absent from the source run", {"target_entity_id": target_entity_id})
    effective = snapshot["effective_mapping"]
    effective_target = effective[target_entity_id]["effective_entity_id"] if target_entity_id else None
    if target_class and target_entity_id:
        expected_class = build_target_reference(target_class).canonical_label
        actual_class = overlay.entity_by_id[target_entity_id].get("class_name")
        if expected_class != actual_class:
            raise IdentityRebuildError("CLASS_MISMATCH", "selected entity does not match target_class", {"target_class": expected_class, "entity_class": actual_class})

    revision_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = revision_dir / "rebuild_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "rebuild_id": str(uuid.uuid4()),
        "run_id": overlay.run_id,
        "source_video_sha256": overlay.video_sha256,
        "source_p2a_mapping_sha256": overlay.mapping_sha256,
        "source_artifact_sha256": overlay.source_artifact_sha256,
        "source_hash_manifest_path": str(overlay.source_hash_manifest_path) if overlay.source_hash_manifest_path else None,
        "source_hash_manifest_sha256": overlay.source_hash_manifest_sha256,
        "overlay_revision": revision,
        "run_status": "RUNNING",
        "identity_review_status": overlay.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
        "trust_review": None,
        "stages": [],
        "failure": None,
        "started_at": _utc_now(),
        "finished_at": None,
        "output_dir": str(revision_dir),
    }
    _write_json_exclusive(manifest_path, manifest)
    try:
        effective_dir = revision_dir / "effective_identity"
        effective_dir.mkdir()
        source_memory = overlay.observations_path
        effective_memory = effective_dir / "observations_effective.jsonl"
        observations = ObservationMemoryStore(source_memory).load_all()
        observation_map = effective
        auto_remapped = []
        for observation in observations:
            source_id = observation.persistent_id
            if source_id is None or source_id not in observation_map:
                raise IdentityRebuildError("SOURCE_MISMATCH", "an observation has no source identity mapping", {"persistent_id": source_id})
            auto_remapped.append(replace(observation, persistent_id=observation_map[source_id]["effective_entity_id"]))
        remapped = auto_remapped
        trust_records = []
        if trust_overlay is not None and trust_snapshot is not None:
            supported_ids = set(trust_snapshot["supported_observation_ids"])
            remapped = []
            for line_index, observation in enumerate(observations, start=1):
                ref = trust_overlay.observation_refs[line_index - 1]
                review = trust_snapshot["observations"].get(ref["raw_observation_id"], {})
                trust_status = "SUPPORTED" if ref["raw_observation_id"] in supported_ids else review.get("trust_status", "UNCERTAIN")
                provenance = review.get("provenance", "UNREVIEWED")
                if ref["raw_observation_id"] in supported_ids and provenance == "UNREVIEWED":
                    provenance = "HUMAN_REVIEWED_EVENT_BOUNDARY"
                trust_records.append({
                    "raw_observation_id": ref["raw_observation_id"],
                    "raw_observation_ref": {"source_path": str(source_memory), "line_number": ref["source_line_number"], "sha256": ref["raw_record_sha256"]},
                    "source_algorithm": ref["source_algorithm"],
                    "source_entity_id": observation.persistent_id,
                    "effective_entity_id": observation_map[observation.persistent_id]["effective_entity_id"],
                    "identity_provenance": observation_map[observation.persistent_id]["identity_provenance"],
                    "trust_status": trust_status,
                    "provenance": provenance,
                    "reviewed_frame_indices": review.get("reviewed_frame_indices", [ref["frame_index"]] if ref["raw_observation_id"] in supported_ids else []),
                    "review_operation_id": review.get("operation_id"),
                    "reason": review.get("reason", "Referenced as a boundary in a human-reviewed event." if ref["raw_observation_id"] in supported_ids and provenance == "HUMAN_REVIEWED_EVENT_BOUNDARY" else "No human review recorded for this exact observation."),
                })
                if ref["raw_observation_id"] in supported_ids:
                    remapped.append(replace(observation, persistent_id=observation_map[observation.persistent_id]["effective_entity_id"]))
        effective_store = ObservationMemoryStore(effective_memory)
        for item in remapped:
            effective_store.append(item)
        if trust_overlay is not None and not effective_memory.exists():
            effective_memory.write_text("", encoding="utf-8")
        auto_effective_memory = None
        trust_records_path = None
        if trust_overlay is not None:
            auto_effective_memory = effective_dir / "observations_automatic_effective.jsonl"
            auto_store = ObservationMemoryStore(auto_effective_memory)
            for item in auto_remapped:
                auto_store.append(item)
            trust_records_path = effective_dir / "trusted_observation_reviews.jsonl"
            with trust_records_path.open("x", encoding="utf-8") as handle:
                for item in trust_records:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
            if not auto_effective_memory.exists():
                auto_effective_memory.write_text("", encoding="utf-8")

        entities = _effective_entities(overlay, effective, revision)
        entities_path = effective_dir / "persistent_entities_effective.json"
        summary_path = effective_dir / "entity_summary_effective.json"
        mapping_path = effective_dir / "effective_entity_mapping.json"
        _write_json_exclusive(entities_path, entities)
        effective_summary = dict(overlay.source_summary)
        effective_summary["persistent_entity_count"] = len(entities)
        effective_summary["consolidated_observation_count"] = len(remapped)
        effective_summary["effective_identity_overlay"] = {
            "overlay_revision": revision,
            "source_run_id": overlay.run_id,
            "source_video_sha256": overlay.video_sha256,
            "source_p2a_mapping_sha256": overlay.mapping_sha256,
            "source_hash_manifest_sha256": overlay.source_hash_manifest_sha256,
            "identity_review_status_preserved": overlay.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
        }
        _write_json_exclusive(summary_path, effective_summary)
        _write_json_exclusive(mapping_path, {
            "source_run_id": overlay.run_id,
            "source_video_sha256": overlay.video_sha256,
            "source_p2a_mapping_sha256": overlay.mapping_sha256,
            "overlay_revision": revision,
            "mapping": effective,
        })
        _write_json_exclusive(revision_dir / "identity_overlay_snapshot.json", history)
        snapshot_path = revision_dir / "identity_overlay_snapshot.json"
        if trust_snapshot is not None:
            trust_counts = {status: 0 for status in TrustReviewOverlay.TRUST_STATUSES}
            for record in trust_records:
                trust_counts[record["trust_status"]] += 1
            manifest["trust_review"] = {
                "overlay_revision": trust_snapshot["overlay_revision"],
                "source_identity_overlay_revision": revision,
                "trusted_observation_count": len(remapped),
                "observation_status_counts": trust_counts,
                "supported_event_count": sum(1 for item in trust_snapshot["events"].values() if item["status"] == "SUPPORTED"),
                "stale_operation_ids": trust_snapshot["stale_operation_ids"],
                "unreviewed_observations_default_to": "UNCERTAIN",
                "interpolation_used": False,
                "review_ledger_sha256": sha256_file(trust_overlay.ledger_path) if trust_overlay.ledger_path.is_file() else None,
            }
        effective_outputs = [effective_memory, entities_path, summary_path, mapping_path, snapshot_path]
        if auto_effective_memory is not None:
            effective_outputs.append(auto_effective_memory)
        if trust_records_path is not None:
            effective_outputs.append(trust_records_path)
        manifest["stages"].append(_stage_record(
            "EFFECTIVE_IDENTITY", "COMPLETED",
            [overlay.manifest_path, source_memory, overlay.mapping_path, overlay.entities_path, overlay.summary_path,
             overlay.conflicts_path, overlay.candidates_path],
            effective_outputs,
        ))
        _write_manifest(manifest_path, manifest)

        project_root = Path(__file__).resolve().parents[2]
        configs = {
            "p2b": project_root / "configs/p2b_state_timeline.yaml",
            "p2c": project_root / "configs/p2c_zone_transition_events.yaml",
            "p2d": project_root / "configs/p2d_move_object_events.yaml",
            "motion": project_root / "configs/p3b1_robust_motion.yaml",
        }
        manifest["configs"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in configs.items()
        }
        p2b_dir = revision_dir / "p2b"
        timeline_path = p2b_dir / "object_state_timelines.json"
        if trust_snapshot is not None and not remapped:
            p2b_dir.mkdir()
            _write_json_exclusive(timeline_path, [])
            _write_json_exclusive(p2b_dir / "state_timeline_summary.json", {
                "p2b_status": "NO_TRUSTED_OBSERVATIONS", "video_id": overlay.video_id,
                "identity_review_status": overlay.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
            })
            p2b_summary = {"p2b_status": "NO_TRUSTED_OBSERVATIONS"}
            p2b_stage_status = "NO_TRUSTED_INPUT"
        else:
            p2b_summary = p2b.run(
                consolidated_memory=effective_memory,
                persistent_entities=entities_path,
                entity_summary=summary_path,
                config_path=configs["p2b"],
                output_dir=p2b_dir,
            )
            p2b_stage_status = "COMPLETED"
        manifest["stages"].append(_stage_record("P2B_STATE_TIMELINE", p2b_stage_status, [effective_memory, entities_path, summary_path, configs["p2b"]], [timeline_path, p2b_dir / "state_timeline_summary.json"]))
        _write_manifest(manifest_path, manifest)

        p2c_dir = revision_dir / "p2c"
        zone_events_path = p2c_dir / "zone_change_events.json"
        p2b_summary_path = p2b_dir / "state_timeline_summary.json"
        if trust_snapshot is not None and not remapped:
            p2c_dir.mkdir()
            _write_json_exclusive(zone_events_path, [])
            _write_json_exclusive(p2c_dir / "zone_change_event_summary.json", {"p2c_status": "NO_TRUSTED_OBSERVATIONS", "event_count": 0})
            p2c_summary = {"p2c_status": "NO_TRUSTED_OBSERVATIONS", "event_count": 0}
            p2c_stage_status = "NO_TRUSTED_INPUT"
        else:
            p2c_summary = p2c.run(
                timeline_path=timeline_path,
                summary_path=p2b_summary_path,
                config_path=configs["p2c"],
                output_dir=p2c_dir,
            )
            p2c_stage_status = "COMPLETED"
        manifest["stages"].append(_stage_record("P2C_ZONE_EVENTS", p2c_stage_status, [timeline_path, p2b_summary_path, configs["p2c"]], [zone_events_path, p2c_dir / "zone_change_event_summary.json"]))
        _write_manifest(manifest_path, manifest)

        p2d_dir = revision_dir / "p2d"
        move_events_path = p2d_dir / "move_object_events.json"
        if trust_snapshot is not None and not remapped:
            p2d_dir.mkdir()
            _write_json_exclusive(move_events_path, [])
            _write_json_exclusive(p2d_dir / "move_object_evidence.json", [])
            _write_json_exclusive(p2d_dir / "move_object_event_summary.json", {"p2d_status": "NO_TRUSTED_OBSERVATIONS", "move_object_event_count": 0})
            p2d_summary = {"p2d_status": "NO_TRUSTED_OBSERVATIONS", "move_object_event_count": 0}
            p2d_stage_status = "NO_TRUSTED_INPUT"
        else:
            p2d_summary = p2d.run(
                timeline_path=timeline_path,
                zone_events_path=zone_events_path,
                consolidated_memory=effective_memory,
                persistent_entities=entities_path,
                entity_summary=summary_path,
                config_path=configs["p2d"],
                output_dir=p2d_dir,
                motion_observations_path=overlay.run_dir / "p2a_r1/observations_with_persistent_ids.jsonl",
                effective_identity_mapping_path=mapping_path,
                motion_config_path=configs["motion"],
                video_path=overlay.video_path,
            )
            p2d_stage_status = "COMPLETED"
        event_trust_status_path = None
        automatic_events = p2d._load_events(move_events_path)
        if trust_snapshot is not None:
            event_trust_status_path = p2d_dir / "event_trust_status.json"
            _write_json_exclusive(event_trust_status_path, {
                "source_run_id": overlay.run_id,
                "source_video_sha256": overlay.video_sha256,
                "identity_overlay_revision": revision,
                "candidate_only": True,
                "events": [
                    {
                        "event_id": event.event_id,
                        "effective_entity_ids": (event.motion_metadata or {}).get("effective_entity_id", event.object_ids),
                        "trust_status": "UNCERTAIN",
                        "provenance": "AUTOMATIC_CANDIDATE",
                        "reason": "P2D output is an automatic candidate and has not been explicitly reviewed in this derived revision.",
                    }
                    for event in automatic_events
                ],
            })
            manifest["trust_review"]["automatic_event_candidate_count"] = len(automatic_events)
            manifest["trust_review"]["supported_event_count_for_queries"] = sum(
                1 for event in _reviewed_move_events(
                    trust_overlay=trust_overlay,
                    trust_snapshot=trust_snapshot,
                    identity_mapping=effective,
                )
            )
            manifest["automatic_event_output_semantics"] = "CANDIDATE_ONLY"
        p2d_outputs = [move_events_path, p2d_dir / "move_object_evidence.json", p2d_dir / "move_object_event_summary.json"]
        if event_trust_status_path is not None:
            p2d_outputs.append(event_trust_status_path)
        manifest["stages"].append(_stage_record("P2D_MOVEMENT_EVENTS", p2d_stage_status, [timeline_path, zone_events_path, effective_memory, entities_path, summary_path, overlay.run_dir / "p2a_r1/observations_with_persistent_ids.jsonl", mapping_path, overlay.video_path, configs["p2d"], configs["motion"]], p2d_outputs))
        _write_manifest(manifest_path, manifest)

        query_results = []
        if questions:
            timelines = p2c._load_timelines(timeline_path)
            zone_events = p2d._load_events(zone_events_path)
            automatic_move_events = p2d._load_events(move_events_path)
            move_events = automatic_move_events
            if trust_overlay is not None and trust_snapshot is not None:
                move_events = _reviewed_move_events(
                    trust_overlay=trust_overlay,
                    trust_snapshot=trust_snapshot,
                    identity_mapping=effective,
                )
                reviewed_events_path = p2d_dir / "events_reviewed_for_queries.json"
                _write_json_exclusive(reviewed_events_path, [event.to_record() for event in move_events])
                manifest["stages"][-1]["outputs"].append({
                    "path": str(reviewed_events_path.resolve()),
                    "sha256": sha256_file(reviewed_events_path),
                })
            for index, question in enumerate(questions, start=1):
                query_dir = revision_dir / "queries" / f"query-{index:03d}"
                query_target_id = effective_target
                query_kind = _query_kind(question)
                target_has_trusted_observations = (
                    query_target_id is None
                    or any(item.persistent_id == query_target_id for item in remapped)
                )
                if trust_snapshot is not None and (not remapped or not target_has_trusted_observations):
                    result = _write_insufficient_trust_query(
                        query_dir, question=question, target_entity_id=query_target_id,
                        target_class=target_class, query_kind=query_kind,
                        trust_revision=trust_snapshot["overlay_revision"],
                        identity_revision=revision,
                    )
                else:
                    result = _run_query(
                        question=question,
                        video_id=overlay.video_id,
                        target_class=target_class,
                        target_entity_id=query_target_id,
                        timelines=timelines,
                        zone_events=zone_events,
                        move_events=move_events,
                        observations=remapped,
                        persistent_entities=entities,
                        video_end_timestamp=float(overlay.metadata["duration"]),
                        video_path=overlay.video_path,
                        video_sha256=overlay.video_sha256,
                        identity_review_status=overlay.source_summary.get("p2a_r1_identity_validation_status"),
                        output_dir=query_dir,
                        trust_evidence_available=trust_snapshot is not None,
                    )
                if trust_snapshot is not None:
                    result = _apply_trust_query_policy(
                        result,
                        question=question,
                        query_kind=query_kind,
                        target_entity_id=query_target_id,
                        identity_mapping=effective,
                        trust_overlay=trust_overlay,
                        trust_snapshot=trust_snapshot,
                        reviewed_move_events=move_events,
                    )
                provenance_entity = result.get("persistent_id") or result.get("target_entity_id") or query_target_id
                source_relation = _provenance_for_effective(provenance_entity, effective) if provenance_entity else None
                result["overlay_provenance"] = {
                    "overlay_revision": revision,
                    "source_run_id": overlay.run_id,
                    "source_video_sha256": overlay.video_sha256,
                    "source_p2a_mapping_sha256": overlay.mapping_sha256,
                    "source_hash_manifest_sha256": overlay.source_hash_manifest_sha256,
                    "identity_provenance": source_relation["identity_provenance"] if source_relation else "AUTOMATIC_SOURCE_OR_UNRESOLVED",
                    "source_entity_ids": source_relation["source_entity_ids"] if source_relation else [],
                    "human_confirmation_operation_ids": source_relation["human_confirmation_operation_ids"] if source_relation else [],
                    "human_review_relations": source_relation["human_review_relations"] if source_relation else [],
                    "source_identity_review_status": overlay.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
                }
                if trust_snapshot is not None:
                    result["trust_provenance"] = {
                        "trust_overlay_revision": trust_snapshot["overlay_revision"],
                        "identity_overlay_revision": revision,
                        "trusted_observation_count": len(remapped),
                        "unreviewed_observations_are": "UNCERTAIN",
                        "interpolation_used": False,
                        "reviewed_event_operation_ids": [
                            item["operation_id"] for item in trust_snapshot["events"].values()
                            if item["status"] == "SUPPORTED" and any(
                                event.event_id == item["event_record"].get("event_id") for event in move_events
                            )
                        ],
                    }
                result.setdefault(
                    "identity_review_status",
                    overlay.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
                )
                _write_json_replace(query_dir / "query_result.json", result)
                query_results.append({"question": question, "path": str(query_dir / "query_result.json"), "status": result.get("status"), "evidence_paths": result.get("evidence_paths", []), "overlay_provenance": result["overlay_provenance"]})
            query_files = sorted(path for path in (revision_dir / "queries").rglob("*") if path.is_file())
            manifest["stages"].append(_stage_record(
                "QUERY_DISPATCH", "COMPLETED",
                [timeline_path, zone_events_path, move_events_path, effective_memory,
                 entities_path, mapping_path, snapshot_path, overlay.video_path],
                query_files,
            ))
        else:
            manifest["stages"].append({"name": "QUERY_DISPATCH", "status": "NOT_REQUESTED", "inputs": [], "outputs": []})
        manifest["query_results"] = query_results
        manifest["p2b_status"] = p2b_summary.get("p2b_status")
        manifest["p2c_status"] = p2c_summary.get("p2c_status")
        manifest["p2d_status"] = p2d_summary.get("p2d_status")
        manifest["zone_change_event_count"] = p2c_summary.get("event_count")
        manifest["move_object_event_count"] = p2d_summary.get("move_object_event_count")
        manifest["effective_target_entity_id"] = effective_target
        manifest["target_identity_provenance"] = _provenance_for_effective(effective_target, effective) if effective_target else None
        if trust_overlay is not None:
            trust_overlay.identity.validate_source_current()
            current_trust = trust_overlay.snapshot()
            if current_trust["overlay_revision"] != trust_snapshot["overlay_revision"]:
                raise IdentityRebuildError("STALE_TRUST_REVISION", "trust review ledger changed while memory was rebuilding")
        overlay.validate_source_current()
        manifest["run_status"] = "COMPLETED"
        manifest["finished_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        return manifest
    except Exception as error:
        manifest["run_status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["finished_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        raise


def _effective_entities(
    overlay: IdentityOverlay, mapping: dict[str, dict[str, Any]], revision: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for source_id, item in mapping.items():
        groups.setdefault(item["effective_entity_id"], []).append(source_id)
    records = []
    for effective_id in sorted(groups):
        source_ids = sorted(groups[effective_id])
        originals = [dict(overlay.entity_by_id[source_id]) for source_id in source_ids]
        record = dict(originals[0])
        record["persistent_id"] = effective_id
        record["source_persistent_ids"] = source_ids
        record["track_ids"] = sorted({
            track_id
            for item in originals
            for track_id in item.get("track_ids", [])
            if isinstance(track_id, str) and track_id
        })
        record["source_fragment_ids"] = sorted({fragment for item in originals for fragment in item.get("fragment_ids", [])})
        record["fragment_ids"] = record["source_fragment_ids"]
        record["observation_count"] = sum(1 for item in overlay.observations if item.persistent_id in source_ids)
        record["canonical_observation_count"] = sum(int(item.get("canonical_observation_count", 0)) for item in originals)
        relation_info = mapping[source_ids[0]]
        record["identity_provenance"] = relation_info["identity_provenance"]
        record["identity_confirmation_operation_ids"] = relation_info["human_confirmation_operation_ids"]
        record["source_identity_review_status"] = relation_info["source_identity_review_status"]
        record["source_p2a_mapping_sha256"] = overlay.mapping_sha256
        record["overlay_revision"] = revision
        records.append(record)
    return records


def _provenance_for_effective(effective_id: str, mapping: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    values = [item for item in mapping.values() if item["effective_entity_id"] == effective_id]
    if not values:
        return None
    return values[0]


def _reviewed_move_events(
    *,
    trust_overlay: TrustReviewOverlay,
    trust_snapshot: dict[str, Any],
    identity_mapping: dict[str, dict[str, Any]],
) -> list[Any]:
    """Load only user-supported source candidates and bind them to current identity."""
    source_path = trust_overlay.run_dir / "p2d" / "move_object_events.json"
    if not source_path.is_file():
        return []
    source_events = p2d._load_events(source_path)
    supported_ids = set(trust_snapshot["supported_observation_ids"])
    decisions = trust_snapshot["events"]
    dimensions = trust_overlay.identity.metadata
    width = float(dimensions["width"])
    height = float(dimensions["height"])
    output = []
    for event in source_events:
        decision = decisions.get(event.event_id)
        if not decision or decision["status"] != "SUPPORTED":
            continue
        reviewed_ids = decision["reviewed_observation_ids"]
        if not reviewed_ids or not set(reviewed_ids).issubset(supported_ids):
            continue
        effective_ids = {
            identity_mapping[source_id]["effective_entity_id"]
            for source_id in event.object_ids
            if source_id in identity_mapping
        }
        if effective_ids != {decision["effective_entity_id"]}:
            continue
        refs = []
        for raw_id in reviewed_ids:
            source = trust_overlay.observation_by_id.get(raw_id)
            if source is None:
                continue
            raw = source["raw_observation"]
            box = raw.get("bbox", {})
            try:
                center = [
                    ((float(box["x1"]) + float(box["x2"])) / 2.0) / width,
                    ((float(box["y1"]) + float(box["y2"])) / 2.0) / height,
                ]
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            refs.append({
                "video_id": trust_overlay.video_id,
                "frame_index": source["frame_index"],
                "timestamp": source["timestamp"],
                "center_normalized": center,
                "effective_entity_id": decision["effective_entity_id"],
                "source_entity_id": source["source_entity_id"],
            })
        refs.sort(key=lambda item: (item["frame_index"], item["timestamp"]))
        if len(refs) < 2 or refs[0]["frame_index"] >= refs[-1]["frame_index"]:
            continue
        before, after = refs[0], refs[-1]
        before_center, after_center = before["center_normalized"], after["center_normalized"]
        dx = after_center[0] - before_center[0]
        dy = after_center[1] - before_center[1]
        identity = _provenance_for_effective(decision["effective_entity_id"], identity_mapping) or {}
        motion = dict(event.motion_metadata or {})
        motion.update({
            "candidate_type": "HUMAN_REVIEWED_POSITION_CHANGE",
            "effective_entity_id": decision["effective_entity_id"],
            "position_before": {"center_normalized": before_center, "zone": event.zone_before},
            "position_after": {"center_normalized": after_center, "zone": event.zone_after},
            "displacement": {"normalized_l2": (dx * dx + dy * dy) ** 0.5},
            "trigger_provenance": ["HUMAN_REVIEWED_BOUNDARY_POSITIONS"],
            "evidence_status": "SUPPORTED_EVENT",
            "identity_provenance": {
                "identity_provenance": identity.get("identity_provenance", "AUTOMATIC_SOURCE"),
                "source_identity_review_status": identity.get("source_identity_review_status", "UNKNOWN"),
                "overlay_revision": trust_snapshot["source_identity_overlay_revision"],
                "human_confirmation_operation_ids": identity.get("human_confirmation_operation_ids", []),
            },
            "observation_references": refs,
            "d1_trust_review": {
                "event_review_status": "SUPPORTED",
                "provenance": decision["provenance"],
                "operation_id": decision["operation_id"],
                "reviewed_observation_ids": reviewed_ids,
                "reviewed_frame_indices": decision["reviewed_frame_indices"],
                "interpretation": "POSITION_CHANGE_BETWEEN_TRUSTED_OBSERVATIONS",
                "reason": decision["reason"],
            },
        })
        output.append(replace(
            event,
            object_ids=[decision["effective_entity_id"]],
            source_segment_ids=(),
            motion_metadata=motion,
        ))
    return output


def _query_kind(question: str) -> str:
    normalized = question.casefold()
    if "before" in normalized or "earlier" in normalized:
        return "before"
    if "when" in normalized or "move" in normalized:
        return "when"
    return "where"


def _write_insufficient_trust_query(
    output_dir: Path,
    *,
    question: str,
    target_entity_id: str | None,
    target_class: str | None,
    query_kind: str,
    trust_revision: int,
    identity_revision: int,
) -> dict[str, Any]:
    message = "No exact video frame has been explicitly reviewed for this object, so RecallVision cannot report a trusted location yet."
    record: dict[str, Any] = {
        "question": question,
        "status": "INSUFFICIENT_TRUSTED_EVIDENCE",
        "answer": message,
        "target_entity_id": target_entity_id,
        "persistent_id": target_entity_id,
        "class_name": target_class,
        "target_selection": {
            "status": "RESOLVED" if target_entity_id else "NOT_RESOLVED",
            "resolved_entity_id": target_entity_id,
            "candidates": [],
        },
        "evidence": [],
        "evidence_paths": [],
        "evidence_status": "NO_TRUSTED_EVIDENCE",
        "global_event_status": "UNVERIFIED_CANDIDATE_STATUS",
        "target_event_status": "INSUFFICIENT_TRUSTED_EVIDENCE" if query_kind in {"when", "before"} else "NOT_APPLICABLE",
        "event_status": "UNVERIFIED_CANDIDATE_STATUS",
        "event_status_scope": "TRUST_REVIEWED_VIEW",
        "trust_provenance": {
            "trust_overlay_revision": trust_revision,
            "identity_overlay_revision": identity_revision,
            "interpolation_used": False,
        },
    }
    _write_json_exclusive(output_dir / "query_result.json", record)
    return record


def _apply_trust_query_policy(
    result: dict[str, Any],
    *,
    question: str,
    query_kind: str,
    target_entity_id: str | None,
    identity_mapping: dict[str, dict[str, Any]],
    trust_overlay: TrustReviewOverlay,
    trust_snapshot: dict[str, Any],
    reviewed_move_events: Sequence[Any],
) -> dict[str, Any]:
    if query_kind == "where" and result.get("status") == "TARGET_LAST_SEEN":
        timestamp = result.get("last_seen_timestamp") or result.get("timestamp")
        frame = result.get("frame_index")
        zone = result.get("zone")
        where = zone.replace("_", " ") if isinstance(zone, str) else "the visible image area"
        result["answer"] = f"The last explicitly reviewed observation was at {where}, at {timestamp:.3f} seconds (frame {frame})." if isinstance(timestamp, (int, float)) else "The object's latest location is based on an explicitly reviewed frame."
        result["location_statement_scope"] = "EXACT_REVIEWED_FRAME"

    if query_kind in {"when", "before"}:
        resolved_id = target_entity_id or (result.get("target_selection") or {}).get("resolved_entity_id") or result.get("persistent_id") or result.get("target_entity_id")
        source_path = trust_overlay.run_dir / "p2d" / "move_object_events.json"
        try:
            all_source_candidates = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else []
        except (OSError, json.JSONDecodeError):
            all_source_candidates = []
        if not isinstance(all_source_candidates, list):
            all_source_candidates = []
        global_candidate_exists = any(isinstance(item, dict) and item.get("event_type") == "MOVE_OBJECT" for item in all_source_candidates)
        result["event_status"] = "SUPPORTED_EVENT" if global_candidate_exists else "NO_SUPPORTED_EVENT"
        result["global_event_status"] = result["event_status"]
        result["global_event_trust_status"] = (
            "HUMAN_REVIEWED_EVENT_PRESENT" if reviewed_move_events
            else "UNVERIFIED_CANDIDATE" if global_candidate_exists
            else "NO_REVIEWED_EVENT_EVIDENCE"
        )
        result["event_status_scope"] = "GLOBAL_LEGACY"
        approved = [event for event in reviewed_move_events if resolved_id in (event.object_ids or [])]
        if approved:
            event_ids = {event.event_id for event in approved}
            result["target_event_status"] = "SUPPORTED_EVENT"
            result["global_event_status"] = "SUPPORTED_EVENT"
            result["event_status"] = "SUPPORTED_EVENT"
            result["event_status_scope"] = "TRUST_REVIEWED_VIEW"
            if query_kind == "when" and result.get("status") in {"MOVEMENT_FOUND", "POSITION_CHANGE_BETWEEN_OBSERVATIONS"}:
                before_after = []
                for event in approved:
                    refs = (event.motion_metadata or {}).get("observation_references", [])
                    if len(refs) >= 2:
                        before, after = refs[0], refs[-1]
                        before_after.append(
                            f"between trusted observations at {before['timestamp']:.3f} s (frame {before['frame_index']}) "
                            f"and {after['timestamp']:.3f} s (frame {after['frame_index']})"
                        )
                result["handler_status"] = result.get("status")
                result["status"] = "POSITION_CHANGE_BETWEEN_OBSERVATIONS"
                result["answer"] = "A user-reviewed change in observed position was confirmed " + "; ".join(before_after) + ". The exact movement onset and path between those observations were not observed."
                result["movement_time_precision"] = "BETWEEN_TRUSTED_OBSERVATIONS"
            elif query_kind == "before" and result.get("answer"):
                result["answer"] = str(result["answer"]).replace(
                    "Before the supported movement,",
                    "Before the user-reviewed position change interval,",
                ) + " The position change is established between the reviewed observations; continuous motion and its exact onset were not observed."
            result["target_event_detection_status"] = "SUPPORTED_EVENT"
            decisions = trust_snapshot["events"]
            result["event_review_provenance"] = [
                {"event_id": event.event_id, "operation_id": decisions[event.event_id]["operation_id"], "provenance": "HUMAN_REVIEW"}
                for event in approved if event.event_id in decisions
            ]
        elif result.get("status") not in {"AMBIGUOUS_TARGET", "TARGET_NOT_FOUND", "TARGET_NOT_RESOLVED"}:
            target_effective_id = resolved_id
            target_source_ids = {
                source_id for source_id, item in identity_mapping.items()
                if item.get("effective_entity_id") == target_effective_id
            }
            candidate_statuses = {}
            if source_path.is_file():
                for candidate in all_source_candidates:
                    if not isinstance(candidate, dict) or candidate.get("event_type") != "MOVE_OBJECT" or not (set(candidate.get("object_ids", [])) & target_source_ids):
                        continue
                    candidate_id = candidate.get("event_id")
                    decision = trust_snapshot["events"].get(candidate_id)
                    candidate_statuses[candidate_id] = decision["status"] if decision else "UNREVIEWED"
            result["target_event_detection_status"] = "SUPPORTED_EVENT" if candidate_statuses else "NO_SUPPORTED_EVENT"
            if candidate_statuses and all(value == "MISSING" for value in candidate_statuses.values()):
                result["status"] = "NO_SUPPORTED_EVENT"
                result["answer"] = "The reviewed movement candidates for this object were not supported by the checked video evidence."
                result["target_event_status"] = "NO_SUPPORTED_EVENT"
                result["target_event_trust_status"] = "NO_SUPPORTED_EVENT"
            elif candidate_statuses:
                result["status"] = "EVENT_REQUIRES_REVIEW"
                result["answer"] = "An automatic movement candidate exists, but no user has confirmed it from its video evidence yet."
                result["target_event_status"] = "EVENT_REQUIRES_REVIEW"
                result["target_event_trust_status"] = "EVENT_REQUIRES_REVIEW"
            else:
                result["status"] = "INSUFFICIENT_TRUSTED_EVIDENCE"
                result["answer"] = "There is not enough explicitly reviewed video evidence to determine whether this object moved."
                result["target_event_status"] = "INSUFFICIENT_TRUSTED_EVIDENCE"
                result["target_event_trust_status"] = "INSUFFICIENT_TRUSTED_EVIDENCE"
        if approved:
            result["target_event_trust_status"] = "SUPPORTED_EVENT"
        result["trust_provenance"] = {
            **(result.get("trust_provenance") or {}),
            "trust_overlay_revision": trust_snapshot["overlay_revision"],
            "source_identity_overlay_revision": trust_snapshot["source_identity_overlay_revision"],
            "interpolation_used": False,
        }
    return result


def _stage_record(name: str, status: str, inputs: Sequence[Path], outputs: Sequence[Path]) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs if path.is_file()],
        "outputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in outputs if path.is_file()],
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json_replace(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
