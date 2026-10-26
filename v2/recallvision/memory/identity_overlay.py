"""Append-only, human-auditable identity relations over one frozen pipeline run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from recallvision.memory.store import ObservationMemoryStore


class OverlayError(ValueError):
    """An explicit, machine-readable identity-overlay rejection."""

    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.status = status
        self.details = details or {}
        super().__init__(f"{status}: {message}")


class IdentityOverlay:
    """Manage auditable identity relations for an immutable source run.

    The JSONL ledger is append-only. Each mutation uses an advisory file lock and
    optimistic revision check, so a stale writer cannot silently overwrite a
    newer confirmation or revocation.
    """

    SCHEMA_VERSION = 2
    RELATIONS = {"SAME_INSTANCE", "DIFFERENT_INSTANCE"}

    def __init__(
        self,
        run_dir: str | Path,
        ledger_path: str | Path,
        source_hash_manifest: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.ledger_path = Path(ledger_path).resolve()
        self._require_outside_source(self.ledger_path, "identity ledger")
        self._require_outside_source(
            self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock").resolve(),
            "identity ledger lock",
        )
        self.source_hash_manifest_path = Path(source_hash_manifest).expanduser().resolve() if source_hash_manifest else None
        if self.source_hash_manifest_path is not None:
            self._require_outside_source(self.source_hash_manifest_path, "source hash manifest")
            if not self.source_hash_manifest_path.is_file():
                raise OverlayError("SOURCE_PROVENANCE_INCOMPLETE", "source hash manifest does not exist", {"path": str(self.source_hash_manifest_path)})
            self.source_hash_manifest_sha256 = _sha256(self.source_hash_manifest_path)
        else:
            self.source_hash_manifest_sha256 = None
        self._load_source()

    def _require_outside_source(self, path: Path, label: str) -> None:
        try:
            path.resolve().relative_to(self.run_dir)
        except ValueError:
            return
        raise OverlayError(
            "OUTPUT_IN_SOURCE_RUN",
            f"{label} must be outside the frozen source run",
            {"path": str(path), "source_run_dir": str(self.run_dir)},
        )

    def _load_source(self) -> None:
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.summary_path = self.run_dir / "p2a_r1" / "entity_consolidation_summary.json"
        self.mapping_path = self.run_dir / "p2a_r1" / "track_to_persistent_mapping.json"
        self.entities_path = self.run_dir / "p2a_r1" / "persistent_entities.json"
        self.conflicts_path = self.run_dir / "p2a_r1" / "entity_conflicts.json"
        self.candidates_path = self.run_dir / "p2a_r1" / "continuation_candidates.json"
        self.observations_path = self.run_dir / "p2a_r1" / "observations_consolidated.jsonl"
        required = [
            self.manifest_path, self.summary_path, self.mapping_path,
            self.entities_path, self.conflicts_path, self.candidates_path,
            self.observations_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise OverlayError("SOURCE_INCOMPLETE", "required frozen run artifacts are missing", {"missing": missing})
        initial_source_hashes = self._current_source_artifact_hashes()
        self.manifest = _read_object(self.manifest_path, "run manifest")
        self._validate_manifest_pins(required)
        self.source_summary = _read_object(self.summary_path, "P2A summary")
        if self.source_summary.get("p2a_r1_pipeline_status") != "PASS":
            raise OverlayError("SOURCE_INCOMPLETE", "P2A-R1 did not complete successfully")
        self.run_id = _required_text(self.manifest, "run_id")
        self.video_sha256 = _required_text(self.source_summary, "video_sha256")
        self.video_id = _required_text(self.source_summary, "video_id")
        manifest_video = self.manifest.get("video", {})
        if isinstance(manifest_video, dict) and manifest_video.get("sha256") not in (None, self.video_sha256):
            raise OverlayError("STALE_SOURCE", "run manifest and P2A summary identify different video hashes")
        if self.video_id != f"sha256:{self.video_sha256}":
            raise OverlayError("STALE_SOURCE", "video_id and source video SHA-256 disagree")
        self.mapping_sha256 = _sha256(self.mapping_path)
        self.mapping = _read_object(self.mapping_path, "track-to-persistent mapping")
        summary_mapping = self.source_summary.get("track_to_persistent_mapping")
        if not isinstance(summary_mapping, dict) or summary_mapping != self.mapping:
            raise OverlayError("SOURCE_MAPPING_MISMATCH", "P2A mapping file differs from the mapping embedded in its summary")
        self.entities = _read_list(self.entities_path, "persistent entities")
        summary_entities = self.source_summary.get("persistent_entities")
        if not isinstance(summary_entities, list) or summary_entities != self.entities:
            raise OverlayError("SOURCE_ENTITY_MISMATCH", "persistent entity file differs from the entities embedded in its P2A summary")
        self.entity_by_id = {
            _required_text(entity, "persistent_id"): entity for entity in self.entities
        }
        if len(self.entity_by_id) != len(self.entities):
            raise OverlayError("SOURCE_INVALID", "persistent entity IDs are not unique")
        for key, persistent_id in self.mapping.items():
            parts = key.split("|") if isinstance(key, str) else []
            if len(parts) != 3 or parts[0] != self.video_id:
                raise OverlayError("SOURCE_INVALID", "track mapping contains a malformed or cross-video key", {"mapping_key": key})
            entity = self.entity_by_id.get(persistent_id)
            if entity is None:
                raise OverlayError("SOURCE_INVALID", "track mapping references a missing persistent entity", {"mapping_key": key, "persistent_id": persistent_id})
            if entity.get("class_name") != parts[1]:
                raise OverlayError("SOURCE_INVALID", "track mapping class does not match its entity", {"mapping_key": key, "persistent_id": persistent_id})
        self.observations = ObservationMemoryStore(self.observations_path).load_all()
        if not self.observations:
            raise OverlayError("SOURCE_INCOMPLETE", "P2A consolidated observations are empty")
        observed_video_ids = {item.video_id for item in self.observations}
        if observed_video_ids != {self.video_id}:
            raise OverlayError("CROSS_VIDEO_SOURCE", "consolidated observations do not belong to one source video", {"video_ids": sorted(observed_video_ids)})
        self.conflicts = _read_list(self.conflicts_path, "entity conflicts")
        self.continuation_candidates = _read_list(self.candidates_path, "continuation candidates")
        self.video_path = self._source_video_path()
        if not self.video_path.is_file():
            raise OverlayError("SOURCE_VIDEO_MISSING", "the original source video is unavailable", {"video_path": str(self.video_path)})
        if _sha256(self.video_path) != self.video_sha256:
            raise OverlayError("STALE_SOURCE", "source video no longer matches the frozen run SHA-256", {"video_path": str(self.video_path)})
        self.metadata = self.source_summary.get("video_metadata", {})
        if not isinstance(self.metadata, dict):
            raise OverlayError("SOURCE_INVALID", "P2A summary video_metadata is invalid")
        self._entity_video_ids = self._resolve_entity_video_ids()
        self._entity_observations: dict[str, list[Any]] = {entity_id: [] for entity_id in self.entity_by_id}
        for observation in self.observations:
            if not observation.persistent_id or observation.persistent_id not in self.entity_by_id:
                raise OverlayError("SOURCE_INVALID", "an observation references an unknown persistent entity", {"persistent_id": observation.persistent_id})
            mapping_key = f"{observation.video_id}|{observation.class_name}|{observation.track_id}"
            mapped_entity_id = self.mapping.get(mapping_key)
            if mapped_entity_id != observation.persistent_id:
                raise OverlayError(
                    "SOURCE_MAPPING_MISMATCH",
                    "a consolidated observation disagrees with its source track-to-persistent mapping",
                    {
                        "mapping_key": mapping_key,
                        "mapping_persistent_id": mapped_entity_id,
                        "observation_persistent_id": observation.persistent_id,
                        "frame_index": observation.frame_index,
                    },
                )
            self._entity_observations[observation.persistent_id].append(observation)
        self._spatial_conflicts = self._build_spatial_conflicts()
        self.source_artifact_sha256 = self._current_source_artifact_hashes()
        if self.source_artifact_sha256 != initial_source_hashes:
            raise OverlayError("STALE_SOURCE", "one or more frozen source artifacts changed while the overlay source was loading")
        self._validate_ledger_source()

    def _validate_manifest_pins(self, required: list[Path]) -> None:
        """Require provenance for conflict/candidate inputs and verify available pins."""
        stage_records = self.manifest.get("stages", [])
        if not isinstance(stage_records, list):
            raise OverlayError("SOURCE_INVALID", "run manifest stages field is invalid")
        pinned: dict[Path, str] = {}
        for stage in stage_records:
            if not isinstance(stage, dict):
                continue
            for item in stage.get("outputs", []):
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                raw = Path(item["path"])
                path = raw if raw.is_absolute() else (self.run_dir / raw)
                digest = item.get("sha256")
                if isinstance(digest, str):
                    pinned[path.resolve()] = digest
        external_pins = self._external_hash_pins(required)
        for path in required:
            resolved = path.resolve()
            manifest_hash = pinned.get(resolved)
            external_hash = external_pins.get(resolved)
            if manifest_hash and external_hash and manifest_hash != external_hash:
                raise OverlayError("SOURCE_PROVENANCE_CONFLICT", "run manifest and external source hash manifest disagree", {"path": str(path), "run_manifest_sha256": manifest_hash, "external_sha256": external_hash})
            expected = manifest_hash or external_hash
            if path in (self.conflicts_path, self.candidates_path) and expected is None:
                raise OverlayError(
                    "SOURCE_PROVENANCE_INCOMPLETE",
                    "conflict and continuation candidate inputs must have a run-manifest pin or external source hash",
                    {"path": str(path)},
                )
            if expected is not None:
                actual = _sha256(path)
                if actual != expected:
                    source = "external source hash manifest" if external_hash else "run-manifest hash"
                    raise OverlayError(
                        "STALE_SOURCE",
                        f"a frozen run artifact no longer matches its {source}",
                        {"path": str(path), "expected_sha256": expected, "actual_sha256": actual},
                    )

    def _external_hash_pins(self, required: list[Path]) -> dict[Path, str]:
        manifest_path = self.source_hash_manifest_path
        if manifest_path is None:
            return {}
        wanted = {path.resolve() for path in required}
        bases = [Path.cwd(), manifest_path.parent, self.run_dir, *self.run_dir.parents]
        base_paths = list(dict.fromkeys(base.resolve() for base in bases))
        pins: dict[Path, str] = {}
        for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.rstrip()
            if not line or line.startswith("#"):
                continue
            if len(line) < 66 or any(char not in "0123456789abcdefABCDEF" for char in line[:64]) or not line[64].isspace():
                raise OverlayError("SOURCE_PROVENANCE_INVALID", "source hash manifest must use sha256sum-compatible lines", {"line": line_number})
            digest = line[:64].lower()
            raw_path = line[64:].lstrip().lstrip("*")
            item_path = Path(raw_path)
            candidates = [item_path.resolve()] if item_path.is_absolute() else [(base / item_path).resolve() for base in base_paths]
            match = next((candidate for candidate in candidates if candidate in wanted), None)
            if match is None:
                continue
            if match in pins and pins[match] != digest:
                raise OverlayError("SOURCE_PROVENANCE_INVALID", "source hash manifest contains conflicting hashes for one artifact", {"path": str(match)})
            pins[match] = digest
        return pins

    def _source_video_path(self) -> Path:
        video = self.manifest.get("video", {})
        if not isinstance(video, dict):
            raise OverlayError("SOURCE_INVALID", "run manifest video field is invalid")
        raw = video.get("path") or video.get("video_path") or video.get("input_path")
        if not isinstance(raw, str) or not raw:
            metadata = self.run_dir / "p1b" / "run_metadata.json"
            if metadata.is_file():
                record = _read_object(metadata, "P1B metadata")
                raw = record.get("video_path")
        if not isinstance(raw, str) or not raw:
            raise OverlayError("SOURCE_INVALID", "run manifest has no source video path")
        path = Path(raw)
        return path if path.is_absolute() else (self.run_dir / path).resolve()

    def _resolve_entity_video_ids(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for entity_id, entity in self.entity_by_id.items():
            value = entity.get("video_id", self.video_id)
            if not isinstance(value, str) or not value:
                raise OverlayError("SOURCE_INVALID", "entity video_id is invalid", {"entity_id": entity_id})
            result[entity_id] = value
        return result

    def _build_spatial_conflicts(self) -> list[dict[str, Any]]:
        fragment_to_entity: dict[str, str] = {}
        for entity_id, entity in self.entity_by_id.items():
            fragments = entity.get("fragment_ids", [])
            if not isinstance(fragments, list):
                continue
            for fragment in fragments:
                if isinstance(fragment, str):
                    fragment_to_entity[fragment] = entity_id
        result = []
        for conflict in self.conflicts:
            a = fragment_to_entity.get(conflict.get("fragment_a"))
            b = fragment_to_entity.get(conflict.get("fragment_b"))
            if a is None or b is None or a == b:
                continue
            result.append({
                "entity_a": a,
                "entity_b": b,
                "reason": conflict.get("reason", "SPATIAL_CONFLICT"),
                "video_id": conflict.get("video_id"),
                "frame_indices": list(conflict.get("frame_indices", [])),
                "center_distance": conflict.get("center_distance"),
                "median_iou": conflict.get("median_iou"),
            })
        return result

    def _current_source_artifact_hashes(self) -> dict[str, str]:
        paths = {
            "run_manifest": self.manifest_path,
            "p2a_summary": self.summary_path,
            "p2a_mapping": self.mapping_path,
            "persistent_entities": self.entities_path,
            "entity_conflicts": self.conflicts_path,
            "continuation_candidates": self.candidates_path,
            "consolidated_observations": self.observations_path,
        }
        return {name: _sha256(path) for name, path in paths.items()}

    def _validate_ledger_source(self) -> None:
        with self._locked(create=False):
            operations = self._read_operations_unlocked()
        for operation in operations:
            if operation.get("source_run_id") != self.run_id:
                raise OverlayError("STALE_SOURCE", "overlay ledger belongs to a different source run")
            if operation.get("source_video_sha256") != self.video_sha256:
                raise OverlayError("STALE_SOURCE", "overlay ledger belongs to a different source video")
            if operation.get("source_p2a_mapping_sha256") != self.mapping_sha256:
                raise OverlayError("STALE_SOURCE", "P2A mapping changed after this overlay was recorded")
            if operation.get("source_artifact_sha256") != self.source_artifact_sha256:
                raise OverlayError("STALE_SOURCE", "one or more frozen source artifacts changed after this overlay was recorded", {"operation_id": operation.get("operation_id")})
            if operation.get("source_hash_manifest_sha256") != self.source_hash_manifest_sha256:
                raise OverlayError("STALE_SOURCE", "source hash manifest changed after this overlay was recorded", {"operation_id": operation.get("operation_id")})

    @property
    def revision(self) -> int:
        return self.snapshot()["overlay_revision"]

    def snapshot(self) -> dict[str, Any]:
        """Return a revision-consistent ledger, history and effective mapping."""
        with self._locked(create=False):
            operations = self._read_operations_unlocked()
            self._validate_operation_sources(operations)
        active = _active_relations(operations)
        revoked = {item.get("target_operation_id") for item in operations if item.get("action") == "REVOKE"}
        history_operations = []
        for item in operations:
            view = dict(item)
            if item.get("action") == "RELATION":
                view["ledger_revocation_status_at_creation"] = item.get("revocation_status")
                view["revocation_status"] = "REVOKED" if item.get("operation_id") in revoked else "ACTIVE"
            history_operations.append(view)
        revision = len(operations)
        history = {
            "source_run_id": self.run_id,
            "source_video_sha256": self.video_sha256,
            "source_p2a_mapping_sha256": self.mapping_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_hash_manifest_path": str(self.source_hash_manifest_path) if self.source_hash_manifest_path else None,
            "source_hash_manifest_sha256": self.source_hash_manifest_sha256,
            "overlay_revision": revision,
            "operations": history_operations,
            "active_relations": active,
            "revoked_operation_ids": sorted(item for item in revoked if isinstance(item, str)),
        }
        return {
            "overlay_revision": revision,
            "history": history,
            "active_relations": active,
            "effective_mapping": self._effective_mapping_for(active),
        }

    def list_entities(
        self, class_name: str | None = None, *, snapshot: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List automatic entities and the observation/candidate evidence for review."""
        snapshot = snapshot or self.snapshot()
        active = snapshot["active_relations"]
        items = []
        for entity_id in sorted(self.entity_by_id):
            entity = self.entity_by_id[entity_id]
            if class_name and entity.get("class_name") != class_name:
                continue
            observations = self._entity_observations[entity_id]
            frames = sorted({item.frame_index for item in observations})
            related_candidates = self._candidates_for_entity(entity_id)
            items.append({
                "persistent_entity": entity,
                "source_video_sha256": self.video_sha256,
                "source_p2a_mapping_sha256": self.mapping_sha256,
                "overlay_revision": snapshot["overlay_revision"],
                "observation_count_in_memory": len(observations),
                "source_track_ids": sorted({str(item.track_id) for item in observations}),
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "evidence_frame_options": _sample_frames(frames),
                "continuation_candidates": related_candidates,
                "active_relations": [item for item in active if entity_id in (item["entity_a"], item["entity_b"])],
                "spatial_conflicts": [item for item in self._spatial_conflicts if entity_id in (item["entity_a"], item["entity_b"])],
            })
        return items

    def inspect_pair(
        self, entity_a: str, entity_b: str, *, snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = snapshot or self.snapshot()
        a, b = self._validate_pair(entity_a, entity_b, require_same_class=False)
        return {
            "status": "REVIEW_REQUIRED",
            "source_run_id": self.run_id,
            "source_video_sha256": self.video_sha256,
            "source_p2a_mapping_sha256": self.mapping_sha256,
            "entity_a": self._entity_evidence(a),
            "entity_b": self._entity_evidence(b),
            "continuation_candidates": self._candidate_records(a, b),
            "spatial_conflicts": self._conflicts_for_pair(a, b),
            "active_relations": [item for item in snapshot["active_relations"] if {item["entity_a"], item["entity_b"]} == {a, b}],
            "overlay_revision": snapshot["overlay_revision"],
            "confirmation_required": True,
        }

    def _entity_evidence(self, entity_id: str) -> dict[str, Any]:
        entity = self.entity_by_id[entity_id]
        observations = self._entity_observations[entity_id]
        ordered = sorted(observations, key=lambda item: (item.frame_index, item.timestamp))
        return {
            "persistent_entity": entity,
            "video_id": self._entity_video_ids[entity_id],
            "observation_count_in_memory": len(ordered),
            "first_observation": _observation_ref(ordered[0], self.video_path, self.video_sha256) if ordered else None,
            "last_observation": _observation_ref(ordered[-1], self.video_path, self.video_sha256) if ordered else None,
            "evidence_frame_options": [
                _observation_ref(ordered[index], self.video_path, self.video_sha256)
                for index in _sample_indices(len(ordered))
            ],
        }

    def _candidates_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        fragments = set(self.entity_by_id[entity_id].get("fragment_ids", []))
        return [item for item in self.continuation_candidates if item.get("source_fragment_id") in fragments or item.get("target_fragment_id") in fragments]

    def _candidate_records(self, entity_a: str, entity_b: str) -> list[dict[str, Any]]:
        fragments_a = set(self.entity_by_id[entity_a].get("fragment_ids", []))
        fragments_b = set(self.entity_by_id[entity_b].get("fragment_ids", []))
        return [
            item for item in self.continuation_candidates
            if (item.get("source_fragment_id") in fragments_a and item.get("target_fragment_id") in fragments_b)
            or (item.get("source_fragment_id") in fragments_b and item.get("target_fragment_id") in fragments_a)
        ]

    def _conflicts_for_pair(self, entity_a: str, entity_b: str) -> list[dict[str, Any]]:
        return [item for item in self._spatial_conflicts if {item["entity_a"], item["entity_b"]} == {entity_a, entity_b}]

    def render_entity_evidence(self, entity_id: str, output_dir: str | Path) -> list[dict[str, Any]]:
        """Render representative source frames for one entity without running perception."""
        self.validate_source_current()
        if entity_id not in self.entity_by_id:
            raise OverlayError("ENTITY_NOT_FOUND", "persistent entity is absent from the source run", {"entity_id": entity_id})
        observations = sorted(
            self._entity_observations[entity_id],
            key=lambda item: (item.frame_index, item.timestamp),
        )
        if not observations:
            raise OverlayError("ENTITY_NO_OBSERVATIONS", "persistent entity has no source observations", {"entity_id": entity_id})
        output = Path(output_dir).resolve()
        self._require_outside_source(output, "review evidence output")
        if output.exists():
            raise OverlayError("OUTPUT_EXISTS", "evidence output directory already exists", {"output_dir": str(output)})
        output.mkdir(parents=True)
        from recallvision.visualization.evidence import render_evidence_frame

        records = []
        for index in _sample_indices(len(observations)):
            observation = observations[index]
            destination = output / f"{entity_id.replace(':', '_')}_frame_{observation.frame_index:06d}.jpg"
            rendered = render_evidence_frame(self.video_path, observation, destination)
            records.append(_observation_ref(observation, self.video_path, self.video_sha256) | {"evidence_path": str(rendered.resolve())})
        self.validate_source_current()
        return records

    def render_pair_evidence(self, entity_a: str, entity_b: str, output_dir: str | Path) -> list[dict[str, Any]]:
        """Render a few source-video frames with the selected entity boxes."""
        self.validate_source_current()
        a, b = self._validate_pair(entity_a, entity_b, require_same_class=False)
        output = Path(output_dir).resolve()
        self._require_outside_source(output, "review evidence output")
        if output.exists():
            raise OverlayError("OUTPUT_EXISTS", "evidence output directory already exists", {"output_dir": str(output)})
        output.mkdir(parents=True)
        from recallvision.visualization.evidence import render_evidence_frame

        records = []
        for entity_id in (a, b):
            observations = sorted(
                self._entity_observations[entity_id],
                key=lambda item: (item.frame_index, item.timestamp),
            )
            for index in _sample_indices(len(observations)):
                observation = observations[index]
                destination = output / f"{entity_id.replace(':', '_')}_frame_{observation.frame_index:06d}.jpg"
                rendered = render_evidence_frame(self.video_path, observation, destination)
                records.append(_observation_ref(observation, self.video_path, self.video_sha256) | {"evidence_path": str(rendered.resolve())})
        self.validate_source_current()
        return records

    def record_relation(
        self,
        entity_a: str,
        entity_b: str,
        relation: str,
        *,
        evidence_refs: list[dict[str, Any]],
        expected_revision: int,
    ) -> dict[str, Any]:
        relation = relation.upper()
        if relation not in self.RELATIONS:
            raise OverlayError("INVALID_RELATION", "relation must be SAME_INSTANCE or DIFFERENT_INSTANCE")
        with self._locked(create=True):
            operations = self._read_operations_unlocked()
            self._validate_operation_sources(operations)
            self._check_revision(expected_revision, len(operations))
            a, b = self._validate_pair(entity_a, entity_b, require_same_class=True)
            refs = self._validate_evidence(a, b, evidence_refs)
            active = _active_relations(operations)
            key = _pair_key(a, b)
            for item in active:
                if _pair_key(item["entity_a"], item["entity_b"]) == key:
                    if item["relation"] == relation:
                        raise OverlayError("DUPLICATE_RELATION", "the same active relation already exists", {"operation_id": item["operation_id"]})
                    raise OverlayError("RELATION_CONFLICT", "the opposite relation is already active", {"operation_id": item["operation_id"], "active_relation": item["relation"]})
            if relation == "SAME_INSTANCE":
                self._check_same_instance_conflicts(a, b, active)
            elif self._same_component(a, b, active):
                raise OverlayError("RELATION_CONFLICT", "entities are already connected by active SAME_INSTANCE relations")
            operation = {
                "schema_version": self.SCHEMA_VERSION,
                "operation_id": str(uuid.uuid4()),
                "action": "RELATION",
                "source_run_id": self.run_id,
                "source_video_sha256": self.video_sha256,
                "source_p2a_mapping_sha256": self.mapping_sha256,
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_hash_manifest_sha256": self.source_hash_manifest_sha256,
                "entity_a": a,
                "entity_b": b,
                "relation": relation,
                "provenance": "HUMAN_CONFIRMED",
                "evidence_refs": refs,
                "confirmation_timestamp": _utc_now(),
                "overlay_revision": len(operations) + 1,
                "session_id": f"anonymous-{uuid.uuid4().hex[:12]}",
                "revocation_status": "ACTIVE",
            }
            self._append_operation_unlocked(operation)
            return {"status": "RECORDED", **operation}

    def revoke(self, operation_id: str, *, expected_revision: int) -> dict[str, Any]:
        with self._locked(create=True):
            operations = self._read_operations_unlocked()
            self._validate_operation_sources(operations)
            self._check_revision(expected_revision, len(operations))
            target = next((item for item in operations if item.get("operation_id") == operation_id and item.get("action") == "RELATION"), None)
            if target is None:
                raise OverlayError("OPERATION_NOT_FOUND", "relation operation does not exist", {"operation_id": operation_id})
            revoked = {item.get("target_operation_id") for item in operations if item.get("action") == "REVOKE"}
            if operation_id in revoked:
                raise OverlayError("ALREADY_REVOKED", "relation has already been revoked", {"operation_id": operation_id})
            operation = {
                "schema_version": self.SCHEMA_VERSION,
                "operation_id": str(uuid.uuid4()),
                "action": "REVOKE",
                "target_operation_id": operation_id,
                "source_run_id": self.run_id,
                "source_video_sha256": self.video_sha256,
                "source_p2a_mapping_sha256": self.mapping_sha256,
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_hash_manifest_sha256": self.source_hash_manifest_sha256,
                "confirmation_timestamp": _utc_now(),
                "overlay_revision": len(operations) + 1,
                "session_id": f"anonymous-{uuid.uuid4().hex[:12]}",
                "revocation_status": "REVOKED",
            }
            self._append_operation_unlocked(operation)
            return {"status": "REVOKED", **operation}

    def history(self) -> dict[str, Any]:
        return self.snapshot()["history"]

    def effective_mapping(self) -> dict[str, dict[str, Any]]:
        return self.snapshot()["effective_mapping"]

    def _effective_mapping_for(self, active: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        parent = {entity_id: entity_id for entity_id in self.entity_by_id}

        def find(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for relation in active:
            if relation["relation"] == "SAME_INSTANCE":
                left, right = find(relation["entity_a"]), find(relation["entity_b"])
                if left != right:
                    root = min(left, right)
                    parent[left] = root
                    parent[right] = root
        groups: dict[str, list[str]] = {}
        for entity_id in sorted(parent):
            groups.setdefault(find(entity_id), []).append(entity_id)
        output: dict[str, dict[str, Any]] = {}
        for members in groups.values():
            effective_id = min(members)
            same_instance_ops = [
                item for item in active
                if item["relation"] == "SAME_INSTANCE" and item["entity_a"] in members and item["entity_b"] in members
            ]
            different_instance_ops = [
                item for item in active
                if item["relation"] == "DIFFERENT_INSTANCE"
                and (item["entity_a"] in members or item["entity_b"] in members)
            ]
            relation_ops = same_instance_ops + different_instance_ops
            if len(members) > 1 and same_instance_ops:
                identity_provenance = "HUMAN_CONFIRMED"
            elif different_instance_ops:
                identity_provenance = "HUMAN_REVIEWED_DISTINCT"
            else:
                identity_provenance = "AUTOMATIC_SOURCE"
            for entity_id in members:
                output[entity_id] = {
                    "source_persistent_id": entity_id,
                    "effective_entity_id": effective_id,
                    "source_video_sha256": self.video_sha256,
                    "source_p2a_mapping_sha256": self.mapping_sha256,
                    "source_hash_manifest_sha256": self.source_hash_manifest_sha256,
                    "identity_provenance": identity_provenance,
                    "source_identity_review_status": self.source_summary.get("p2a_r1_identity_validation_status", "UNKNOWN"),
                    "human_confirmation_operation_ids": [item["operation_id"] for item in relation_ops],
                    "human_review_relations": [
                        {"relation": item["relation"], "operation_id": item["operation_id"]}
                        for item in relation_ops
                    ],
                    "source_entity_ids": members,
                }
        return output

    def _validate_pair(self, entity_a: str, entity_b: str, *, require_same_class: bool) -> tuple[str, str]:
        if not isinstance(entity_a, str) or not isinstance(entity_b, str) or not entity_a or not entity_b:
            raise OverlayError("INVALID_ENTITY", "entity IDs must be non-empty strings")
        if entity_a == entity_b:
            raise OverlayError("INVALID_ENTITY", "an entity cannot be related to itself")
        if entity_a not in self.entity_by_id or entity_b not in self.entity_by_id:
            missing = [item for item in (entity_a, entity_b) if item not in self.entity_by_id]
            raise OverlayError("ENTITY_NOT_FOUND", "one or more persistent entities do not exist in the source run", {"entity_ids": missing})
        if self._entity_video_ids[entity_a] != self.video_id or self._entity_video_ids[entity_b] != self.video_id:
            raise OverlayError("CROSS_VIDEO_ENTITY", "both entities must belong to this source video", {"entity_video_ids": {entity_a: self._entity_video_ids[entity_a], entity_b: self._entity_video_ids[entity_b]}, "source_video_id": self.video_id})
        class_a = self.entity_by_id[entity_a].get("class_name")
        class_b = self.entity_by_id[entity_b].get("class_name")
        if require_same_class and class_a != class_b:
            raise OverlayError("CLASS_MISMATCH", "identity relations require the same detected class", {"classes": {entity_a: class_a, entity_b: class_b}})
        return tuple(sorted((entity_a, entity_b)))  # stable undirected identity pair

    def _validate_evidence(self, entity_a: str, entity_b: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(refs, list) or not refs:
            raise OverlayError("EVIDENCE_REQUIRED", "an explicit identity relation requires at least one video evidence reference")
        valid_frames = {item.frame_index for entity_id in (entity_a, entity_b) for item in self._entity_observations[entity_id]}
        normalized = []
        frame_count = self.metadata.get("frame_count")
        for ref in refs:
            if not isinstance(ref, dict):
                raise OverlayError("INVALID_EVIDENCE", "each evidence reference must be an object")
            video_id = ref.get("video_id", self.video_id)
            frame = ref.get("frame_index")
            if video_id != self.video_id:
                raise OverlayError("CROSS_VIDEO_EVIDENCE", "evidence must reference the source video", {"video_id": video_id})
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise OverlayError("INVALID_EVIDENCE", "frame_index must be a non-negative integer")
            if isinstance(frame_count, int) and frame >= frame_count:
                raise OverlayError("INVALID_EVIDENCE", "evidence frame is outside the source video", {"frame_index": frame, "frame_count": frame_count})
            if frame not in valid_frames:
                raise OverlayError("INVALID_EVIDENCE", "evidence frame does not contain an observation from either selected entity", {"frame_index": frame, "entity_ids": [entity_a, entity_b]})
            matching_candidates = [
                item for item in self._entity_observations[entity_a] + self._entity_observations[entity_b]
                if item.frame_index == frame
            ]
            if not matching_candidates:
                raise OverlayError("INVALID_EVIDENCE", "evidence frame has no source observation for either selected entity", {"frame_index": frame})
            supplied_timestamp = ref.get("timestamp")
            if supplied_timestamp is not None and (isinstance(supplied_timestamp, bool) or not isinstance(supplied_timestamp, (int, float))):
                raise OverlayError("INVALID_EVIDENCE", "timestamp must be numeric or null")
            if supplied_timestamp is not None:
                matching_candidates = [
                    item for item in matching_candidates
                    if abs(float(supplied_timestamp) - item.timestamp) <= 1e-6
                ]
                if not matching_candidates:
                    raise OverlayError("INVALID_EVIDENCE", "timestamp disagrees with the source observations at this frame", {"frame_index": frame, "supplied_timestamp": supplied_timestamp})
            evidence_path = ref.get("evidence_path")
            if evidence_path is not None and (not isinstance(evidence_path, str) or not Path(evidence_path).is_file()):
                raise OverlayError("INVALID_EVIDENCE", "evidence_path must refer to an existing rendered frame image", {"evidence_path": evidence_path})
            resolved_evidence_path = Path(evidence_path).resolve() if evidence_path is not None else None
            evidence_sha256 = None
            matching = matching_candidates[0]
            if resolved_evidence_path is not None:
                from recallvision.visualization.evidence import render_evidence_frame

                evidence_sha256 = _sha256(resolved_evidence_path)
                expected_hashes = []
                for candidate in matching_candidates:
                    with tempfile.TemporaryDirectory(prefix="recallvision-identity-evidence-") as temporary_dir:
                        rendered_path = Path(temporary_dir) / "expected.jpg"
                        render_evidence_frame(self.video_path, candidate, rendered_path)
                        expected_hash = _sha256(rendered_path)
                    expected_hashes.append(expected_hash)
                    if evidence_sha256 == expected_hash:
                        matching = candidate
                        break
                else:
                    raise OverlayError(
                        "INVALID_EVIDENCE",
                        "evidence image is not the deterministic rendering of any selected source observation at this frame",
                        {"evidence_path": str(resolved_evidence_path), "expected_sha256": expected_hashes, "actual_sha256": evidence_sha256, "frame_index": frame},
                    )
            timestamp = matching.timestamp
            normalized.append({
                "video_id": self.video_id,
                "video_path": str(self.video_path),
                "video_sha256": self.video_sha256,
                "frame_index": frame,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "source_persistent_ids": [entity_a, entity_b],
                "evidence_path": str(resolved_evidence_path) if resolved_evidence_path else None,
                "evidence_sha256": evidence_sha256,
            })
        self.validate_source_current()
        return normalized

    def _check_same_instance_conflicts(self, entity_a: str, entity_b: str, active: list[dict[str, Any]]) -> None:
        members = self._component_members(entity_a, entity_b, active)
        member_set = set(members)
        for conflict in self._spatial_conflicts:
            if conflict["entity_a"] in member_set and conflict["entity_b"] in member_set:
                raise OverlayError("SPATIAL_CONFLICT", "the proposed identity component contains a recorded simultaneous spatial conflict", {
                    "entity_ids": [conflict["entity_a"], conflict["entity_b"]],
                    "conflict_frames": conflict["frame_indices"],
                    "conflict": conflict,
                })
        for relation in active:
            if relation["relation"] == "DIFFERENT_INSTANCE" and relation["entity_a"] in member_set and relation["entity_b"] in member_set:
                raise OverlayError("RELATION_CONFLICT", "the proposed transitive identity component violates an active DIFFERENT_INSTANCE relation", {
                    "entity_ids": [relation["entity_a"], relation["entity_b"]],
                    "operation_id": relation["operation_id"],
                })

    def _component_members(self, entity_a: str, entity_b: str, active: list[dict[str, Any]]) -> list[str]:
        parent = {item: item for item in self.entity_by_id}
        def find(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item
        for relation in active:
            if relation["relation"] != "SAME_INSTANCE":
                continue
            left, right = find(relation["entity_a"]), find(relation["entity_b"])
            if left != right:
                parent[right] = left
        left, right = find(entity_a), find(entity_b)
        if left != right:
            parent[right] = left
        root = find(entity_a)
        return sorted(item for item in parent if find(item) == root)

    def _same_component(self, entity_a: str, entity_b: str, active: list[dict[str, Any]]) -> bool:
        return entity_b in self._component_members(entity_a, entity_a, active)

    def _check_revision(self, expected: int, actual: int) -> None:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected != actual:
            raise OverlayError("STALE_REVISION", "overlay revision changed; reload the current revision before writing", {"expected_revision": expected, "current_revision": actual})

    def _validate_operation_sources(self, operations: list[dict[str, Any]]) -> None:
        current_artifacts = self._current_source_artifact_hashes()
        hash_manifest_changed = self.source_hash_manifest_path is not None and _sha256(self.source_hash_manifest_path) != self.source_hash_manifest_sha256
        if current_artifacts != self.source_artifact_sha256 or _sha256(self.video_path) != self.video_sha256 or hash_manifest_changed:
            raise OverlayError("STALE_SOURCE", "a frozen source artifact changed during this overlay session", {"expected_artifacts": self.source_artifact_sha256, "current_artifacts": current_artifacts})
        for operation in operations:
            if (operation.get("source_run_id") != self.run_id
                    or operation.get("source_video_sha256") != self.video_sha256
                    or operation.get("source_p2a_mapping_sha256") != self.mapping_sha256
                    or operation.get("source_artifact_sha256") != self.source_artifact_sha256
                    or operation.get("source_hash_manifest_sha256") != self.source_hash_manifest_sha256):
                raise OverlayError("STALE_SOURCE", "overlay operation source hashes do not match this run", {"operation_id": operation.get("operation_id")})

    def validate_source_current(self) -> None:
        """Fail if frozen run artifacts or the video changed after initialization."""
        current_artifacts = self._current_source_artifact_hashes()
        hash_manifest_changed = self.source_hash_manifest_path is not None and _sha256(self.source_hash_manifest_path) != self.source_hash_manifest_sha256
        if current_artifacts != self.source_artifact_sha256 or _sha256(self.video_path) != self.video_sha256 or hash_manifest_changed:
            raise OverlayError(
                "STALE_SOURCE",
                "a frozen source artifact changed during this overlay operation",
                {"expected_artifacts": self.source_artifact_sha256, "current_artifacts": current_artifacts},
            )

    def _active_relations(self) -> list[dict[str, Any]]:
        with self._locked(create=False):
            operations = self._read_operations_unlocked()
        self._validate_operation_sources(operations)
        return _active_relations(operations)

    def _read_operations_unlocked(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        operations = []
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise OverlayError("CORRUPT_LEDGER", f"invalid JSON at line {line_number}") from error
                if not isinstance(value, dict) or value.get("overlay_revision") != len(operations) + 1:
                    raise OverlayError("CORRUPT_LEDGER", f"invalid ledger revision at line {line_number}")
                operations.append(value)
        return operations

    def _append_operation_unlocked(self, operation: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    class _Lock:
        def __init__(self, path: Path, create: bool) -> None:
            self.path = path
            self.create = create
            self.handle = None
        def __enter__(self):
            if self.create:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            elif not self.path.exists():
                return self
            try:
                self.handle = self.path.open("a+" if self.create else "r")
            except FileNotFoundError:
                if not self.create:
                    return self
                raise
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX if self.create else fcntl.LOCK_SH)
            return self
        def __exit__(self, exc_type, exc, tb):
            if self.handle is not None:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()

    def _locked(self, *, create: bool) -> "IdentityOverlay._Lock":
        return self._Lock(self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock"), create)


def _active_relations(operations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(operations)
    revoked = {item.get("target_operation_id") for item in records if item.get("action") == "REVOKE"}
    return [
        item for item in records
        if item.get("action") == "RELATION" and item.get("operation_id") not in revoked
    ]


def _pair_key(entity_a: str, entity_b: str) -> tuple[str, str]:
    return tuple(sorted((entity_a, entity_b)))


def _sample_frames(frames: list[int]) -> list[int]:
    if len(frames) <= 3:
        return frames
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def _sample_indices(length: int) -> list[int]:
    if length <= 3:
        return list(range(length))
    return [0, length // 2, length - 1]


def _observation_ref(observation: Any, video_path: Path, video_sha256: str) -> dict[str, Any]:
    return {
        "video_id": observation.video_id,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "frame_index": observation.frame_index,
        "timestamp": observation.timestamp,
        "persistent_id": observation.persistent_id,
        "track_id": observation.track_id,
        "class_name": observation.class_name,
        "bbox": {
            "x1": observation.bbox.x1,
            "y1": observation.bbox.y1,
            "x2": observation.bbox.x2,
            "y2": observation.bbox.y2,
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OverlayError("SOURCE_INVALID", f"{label} must be a JSON object")
    return value


def _read_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise OverlayError("SOURCE_INVALID", f"{label} must be a JSON list of objects")
    return value


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise OverlayError("SOURCE_INVALID", f"{key} must be a non-empty string")
    return value
