"""Append-only, frame-specific trust reviews over frozen RecallVision runs.

Raw observations and automatic movement events remain untouched. Unreviewed
observations are UNCERTAIN by default; only explicit human decisions contribute
to the trusted evidence view.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recallvision.memory.identity_overlay import IdentityOverlay, OverlayError


class TrustReviewError(ValueError):
    """An explicit, machine-readable trust-review rejection."""

    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.status = status
        self.details = details or {}
        super().__init__(f"{status}: {message}")


class TrustReviewOverlay:
    """Manage reversible observation and event reviews for one frozen run."""

    SCHEMA_VERSION = 1
    TRUST_STATUSES = {"SUPPORTED", "UNCERTAIN", "MISSING"}

    def __init__(
        self,
        run_dir: str | Path,
        ledger_path: str | Path,
        *,
        identity_ledger_path: str | Path,
        source_hash_manifest: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.ledger_path = Path(ledger_path).expanduser().resolve()
        self.identity = IdentityOverlay(self.run_dir, identity_ledger_path, source_hash_manifest)
        self._require_outside_source(self.ledger_path, "trust review ledger")
        self._require_outside_source(self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock"), "trust review lock")
        self.manifest = self.identity.manifest
        self.video_id = self.identity.video_id
        self.video_sha256 = self.identity.video_sha256
        self.run_id = self.identity.run_id
        self.mapping_sha256 = self.identity.mapping_sha256
        self._identity_snapshot = self.identity.snapshot()
        self.observation_refs = self._read_observation_refs()
        self.observation_by_id = {item["raw_observation_id"]: item for item in self.observation_refs}

    def _require_outside_source(self, path: Path, label: str) -> None:
        try:
            path.resolve().relative_to(self.run_dir)
        except ValueError:
            return
        raise TrustReviewError("OUTPUT_IN_SOURCE_RUN", f"{label} must be outside the frozen source run", {"path": str(path)})

    def _read_observation_refs(self) -> list[dict[str, Any]]:
        output = []
        with self.identity.observations_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TrustReviewError("SOURCE_INVALID", "source observations contain an empty line", {"line": line_number})
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TrustReviewError("SOURCE_INVALID", "source observation is invalid JSON", {"line": line_number}) from error
                if not isinstance(record, dict):
                    raise TrustReviewError("SOURCE_INVALID", "source observation must be an object", {"line": line_number})
                digest = hashlib.sha256(f"{line_number}:{line.rstrip()}".encode("utf-8")).hexdigest()
                source_id = record.get("persistent_id")
                mapping = self._identity_snapshot["effective_mapping"].get(source_id)
                if not isinstance(mapping, dict):
                    raise TrustReviewError("SOURCE_MAPPING_MISMATCH", "observation has no effective identity mapping", {"line": line_number, "persistent_id": source_id})
                video_id = record.get("video_id")
                if video_id != self.video_id:
                    raise TrustReviewError("CROSS_VIDEO_OBSERVATION", "observation does not belong to the selected source video", {"line": line_number})
                frame_index = record.get("frame_index")
                timestamp = record.get("timestamp")
                if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                    raise TrustReviewError("SOURCE_INVALID", "observation frame_index must be a non-negative integer", {"line": line_number})
                if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                    raise TrustReviewError("SOURCE_INVALID", "observation timestamp must be numeric", {"line": line_number})
                model = self.manifest.get("model", {})
                model_path = model.get("resolved_path") or model.get("requested_path") if isinstance(model, dict) else None
                tracker = model.get("tracker") if isinstance(model, dict) else None
                algorithm = " + ".join(str(item) for item in (Path(model_path).name if isinstance(model_path, str) else None, tracker) if item)
                output.append({
                    "raw_observation_id": f"obs:{line_number}:{digest[:16]}",
                    "source_line_number": line_number,
                    "raw_record_sha256": digest,
                    "raw_observation": record,
                    "source_entity_id": source_id,
                    "effective_entity_id": mapping["effective_entity_id"],
                    "identity_provenance": mapping["identity_provenance"],
                    "identity_confirmation_operation_ids": mapping.get("human_confirmation_operation_ids", []),
                    "frame_index": frame_index,
                    "timestamp": float(timestamp),
                    "class_name": record.get("class_name"),
                    "track_id": record.get("track_id"),
                    "source_algorithm": algorithm or "UNKNOWN_SOURCE_ALGORITHM",
                })
        return output

    def observation_candidates(self, effective_entity_id: str | None = None) -> list[dict[str, Any]]:
        """List exact source observations, defaulting each unreviewed item to UNCERTAIN."""
        snapshot = self.snapshot()
        output = []
        for item in self.observation_refs:
            if effective_entity_id is not None and item["effective_entity_id"] != effective_entity_id:
                continue
            review = snapshot["observations"].get(item["raw_observation_id"], {})
            output.append({
                **{key: value for key, value in item.items() if key != "raw_observation"},
                "trust_status": review.get("trust_status", "UNCERTAIN"),
                "trust_reason": review.get("reason", "No human review recorded for this exact observation."),
                "review_operation_id": review.get("operation_id"),
                "reviewed_frame_indices": review.get("reviewed_frame_indices", []),
                "review_provenance": review.get("provenance", "UNREVIEWED"),
            })
        return output

    def snapshot(self) -> dict[str, Any]:
        """Return current review decisions while retaining stale history for audit."""
        identity_snapshot = self.identity.snapshot()
        operations = self._read_operations()
        revoked = {item.get("target_operation_id") for item in operations if item.get("action") == "REVOKE"}
        history = []
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        stale_ids = []
        for item in operations:
            view = dict(item)
            if item.get("action") == "REVOKE":
                history.append(view)
                continue
            view["revocation_status"] = "REVOKED" if item.get("operation_id") in revoked else "ACTIVE"
            stale_identity = item.get("source_identity_overlay_revision") != identity_snapshot["overlay_revision"]
            view["source_identity_status"] = "STALE" if stale_identity else "CURRENT"
            if stale_identity:
                stale_ids.append(item.get("operation_id"))
            history.append(view)
            if view["revocation_status"] == "ACTIVE" and not stale_identity:
                key = (str(item.get("action")), str(item.get("observation_id") or item.get("event_id")))
                latest[key] = item

        observations: dict[str, dict[str, Any]] = {
            item["raw_observation_id"]: {
                "trust_status": "UNCERTAIN",
                "reason": "No human review recorded for this exact observation.",
                "operation_id": None,
                "reviewed_frame_indices": [],
                "provenance": "UNREVIEWED",
                "overlay_revision": None,
                "effective_entity_id": item["effective_entity_id"],
                "source_entity_id": item["source_entity_id"],
                "source_algorithm": item["source_algorithm"],
            }
            for item in self.observation_refs
        }
        events: dict[str, dict[str, Any]] = {}
        supported_by_events: set[str] = set()
        for (action, subject_id), item in latest.items():
            if action == "OBSERVATION_REVIEW":
                observations[subject_id] = {
                    "trust_status": item["trust_status"],
                    "reason": item["reason"],
                    "operation_id": item["operation_id"],
                    "reviewed_frame_indices": item["reviewed_frame_indices"],
                    "provenance": item["provenance"],
                    "overlay_revision": item["overlay_revision"],
                }
        for (action, subject_id), item in latest.items():
            if action != "EVENT_REVIEW":
                continue
            source_path = Path(str(item.get("event_file_path", "")))
            source_current = (
                source_path.is_file()
                and _sha256_file(source_path) == item.get("event_file_sha256")
                and _sha256_bytes(_canonical_json(item.get("event_record"))) == item.get("event_record_sha256")
            )
            if not source_current:
                stale_ids.append(item.get("operation_id"))
                continue
            events[subject_id] = {
                "status": item["event_review_status"],
                "reason": item["reason"],
                "operation_id": item["operation_id"],
                "event_record": item["event_record"],
                "event_record_sha256": item["event_record_sha256"],
                "event_file_path": item["event_file_path"],
                "event_file_sha256": item["event_file_sha256"],
                "effective_entity_id": item["effective_entity_id"],
                "reviewed_observation_ids": item["reviewed_observation_ids"],
                "reviewed_frame_indices": item["reviewed_frame_indices"],
                "provenance": item["provenance"],
                "interpretation": item.get("interpretation"),
                "overlay_revision": item["overlay_revision"],
            }
            explicitly_missing = sorted(
                observation_id for observation_id in item["reviewed_observation_ids"]
                if observations.get(observation_id, {}).get("trust_status") == "MISSING"
            )
            if item["event_review_status"] == "SUPPORTED" and explicitly_missing:
                events[subject_id]["review_status"] = "SUPPORTED"
                events[subject_id]["status"] = "BLOCKED_BY_MISSING_OBSERVATION"
                events[subject_id]["evidence_eligibility"] = "BLOCKED"
                events[subject_id]["blocking_observation_ids"] = explicitly_missing
            else:
                events[subject_id]["review_status"] = item["event_review_status"]
                events[subject_id]["evidence_eligibility"] = "ELIGIBLE" if item["event_review_status"] == "SUPPORTED" else "NOT_SUPPORTED"
                if item["event_review_status"] == "SUPPORTED":
                    supported_by_events.update(
                        observation_id for observation_id in item["reviewed_observation_ids"]
                        if observations.get(observation_id, {}).get("trust_status") != "MISSING"
                    )
        supported = {key for key, value in observations.items() if value["trust_status"] == "SUPPORTED"} | supported_by_events
        return {
            "run_id": self.run_id,
            "source_video_sha256": self.video_sha256,
            "source_p2a_mapping_sha256": self.mapping_sha256,
            "source_identity_overlay_revision": identity_snapshot["overlay_revision"],
            "overlay_revision": len(operations),
            "observations": observations,
            "events": events,
            "supported_observation_ids": sorted(supported),
            "stale_operation_ids": sorted(x for x in stale_ids if isinstance(x, str)),
            "history": {
                "overlay_revision": len(operations),
                "operations": history,
                "revoked_operation_ids": sorted(x for x in revoked if isinstance(x, str)),
            },
        }

    def record_observation_review(
        self,
        raw_observation_id: str,
        trust_status: str,
        *,
        reviewed_frame_indices: list[int],
        reason: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        status = str(trust_status).upper()
        if status not in self.TRUST_STATUSES:
            raise TrustReviewError("INVALID_TRUST_STATUS", "trust_status must be SUPPORTED, UNCERTAIN, or MISSING")
        observation = self.observation_by_id.get(raw_observation_id)
        if observation is None:
            raise TrustReviewError("OBSERVATION_NOT_FOUND", "raw observation reference is not present in the frozen source")
        frames = _normalize_frames(reviewed_frame_indices)
        expected_frame = observation["frame_index"]
        if frames != [expected_frame]:
            raise TrustReviewError("REVIEW_FRAME_REQUIRED", "review must record the exact single frame shown for this observation", {"expected_frame_indices": [expected_frame], "received": frames})
        if not isinstance(reason, str) or not reason.strip():
            raise TrustReviewError("REVIEW_REASON_REQUIRED", "a short review reason is required")
        with self._locked(create=True):
            operations = self._read_operations_unlocked()
            self._check_revision(expected_revision, len(operations))
            identity_snapshot = self.identity.snapshot()
            self._validate_source_current()
            operation = self._operation_base("OBSERVATION_REVIEW", len(operations) + 1, identity_snapshot)
            operation.update({
                "observation_id": raw_observation_id,
                "source_line_number": observation["source_line_number"],
                "raw_observation_sha256": observation["raw_record_sha256"],
                "source_entity_id": observation["source_entity_id"],
                "effective_entity_id": observation["effective_entity_id"],
                "identity_provenance": observation["identity_provenance"],
                "source_algorithm": observation["source_algorithm"],
                "trust_status": status,
                "reviewed_frame_indices": frames,
                "reviewed_interval": {"frame_indices": frames, "coverage": "EXACT_REVIEWED_FRAMES_ONLY"},
                "reason": reason.strip(),
                "provenance": "HUMAN_REVIEW",
            })
            self._append_unlocked(operation)
        return {"status": "RECORDED", **operation}

    def record_event_review(
        self,
        event_id: str,
        event_review_status: str,
        *,
        reviewed_observation_ids: list[str],
        reviewed_frame_indices: list[int],
        reason: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        status = str(event_review_status).upper()
        if status not in self.TRUST_STATUSES:
            raise TrustReviewError("INVALID_TRUST_STATUS", "event review status must be SUPPORTED, UNCERTAIN, or MISSING")
        events_path = self.run_dir / "p2d" / "move_object_events.json"
        if not events_path.is_file():
            raise TrustReviewError("EVENT_SOURCE_MISSING", "the frozen run has no MOVE_OBJECT event candidates")
        event_file = _read_json(events_path)
        event_record = next((item for item in event_file if isinstance(item, dict) and item.get("event_id") == event_id), None)
        if event_record is None or event_record.get("event_type") != "MOVE_OBJECT":
            raise TrustReviewError("EVENT_NOT_FOUND", "MOVE_OBJECT candidate is not present in the frozen source event file", {"event_id": event_id})
        if any(
            not isinstance(evidence, dict) or evidence.get("video_id") != self.video_id
            for evidence in event_record.get("evidence", [])
        ):
            raise TrustReviewError("CROSS_VIDEO_EVENT", "every reviewed event evidence item must reference this source video", {"event_id": event_id})
        expected_file_hash = _manifest_output_hash(self.manifest, events_path)
        if expected_file_hash is None or expected_file_hash != _sha256_file(events_path):
            raise TrustReviewError("EVENT_SOURCE_UNPINNED", "MOVE_OBJECT candidate file is not pinned to the source run manifest", {"path": str(events_path)})
        evidence_frames = sorted({
            int(frame)
            for evidence in event_record.get("evidence", []) if isinstance(evidence, dict)
            for frame in evidence.get("frame_indices", [])
            if isinstance(frame, int) and not isinstance(frame, bool)
        })
        frames = _normalize_frames(reviewed_frame_indices)
        if not evidence_frames or frames != evidence_frames:
            raise TrustReviewError("EVENT_EVIDENCE_INCOMPLETE", "review must name every exact evidence frame for this event", {"required_frame_indices": evidence_frames, "received": frames})
        ids = sorted(set(reviewed_observation_ids))
        if len(ids) != len(reviewed_observation_ids) or not ids:
            raise TrustReviewError("EVENT_EVIDENCE_INCOMPLETE", "event review must include unique source observation references")
        refs = [self.observation_by_id.get(item) for item in ids]
        if any(item is None for item in refs):
            raise TrustReviewError("OBSERVATION_NOT_FOUND", "event evidence includes an unknown source observation reference")
        source_entity_ids = event_record.get("object_ids", [])
        mapped_entities = {
            self._identity_snapshot["effective_mapping"][entity_id]["effective_entity_id"]
            for entity_id in source_entity_ids
            if entity_id in self._identity_snapshot["effective_mapping"]
        }
        ref_entities = {item["effective_entity_id"] for item in refs if item is not None}
        if len(mapped_entities) != 1 or ref_entities != mapped_entities:
            raise TrustReviewError("EVENT_TARGET_MISMATCH", "reviewed observations and event must resolve to the same effective object", {"event_entities": sorted(mapped_entities), "observation_entities": sorted(ref_entities)})
        observed_frames = {item["frame_index"] for item in refs if item is not None}
        if not set(evidence_frames).issubset(observed_frames):
            raise TrustReviewError("EVENT_EVIDENCE_INCOMPLETE", "each event evidence frame needs a referenced source observation", {"required_frame_indices": evidence_frames, "observed_frame_indices": sorted(observed_frames)})
        if not isinstance(reason, str) or not reason.strip():
            raise TrustReviewError("REVIEW_REASON_REQUIRED", "a short event review reason is required")
        identity_snapshot = self.identity.snapshot()
        record_hash = _sha256_bytes(_canonical_json(event_record))
        event_file_hash = _sha256_file(events_path)
        with self._locked(create=True):
            operations = self._read_operations_unlocked()
            self._check_revision(expected_revision, len(operations))
            self._validate_source_current()
            revoked_ids = {item.get("target_operation_id") for item in operations if item.get("action") == "REVOKE"}
            latest_observation_reviews: dict[str, dict[str, Any]] = {}
            for prior in operations:
                if (
                    prior.get("action") == "OBSERVATION_REVIEW"
                    and prior.get("operation_id") not in revoked_ids
                    and prior.get("source_identity_overlay_revision") == identity_snapshot["overlay_revision"]
                ):
                    latest_observation_reviews[str(prior.get("observation_id"))] = prior
            rejected_boundary = next((
                observation_id for observation_id in ids
                if latest_observation_reviews.get(observation_id, {}).get("trust_status") == "MISSING"
                and self.observation_by_id[observation_id]["frame_index"] in evidence_frames
            ), None)
            if status == "SUPPORTED" and rejected_boundary is not None:
                raise TrustReviewError(
                    "EVENT_BOUNDARY_OBSERVATION_REJECTED",
                    "a movement event cannot be supported when one of its boundary observations is marked MISSING",
                    {"event_id": event_id, "observation_id": rejected_boundary},
                )
            operation = self._operation_base("EVENT_REVIEW", len(operations) + 1, identity_snapshot)
            operation.update({
                "event_id": event_id,
                "event_record": event_record,
                "event_record_sha256": record_hash,
                "event_file_path": str(events_path),
                "event_file_sha256": event_file_hash,
                "effective_entity_id": next(iter(mapped_entities)),
                "reviewed_observation_ids": ids,
                "reviewed_frame_indices": frames,
                "supporting_time_range": {"start_time": event_record.get("start_time"), "end_time": event_record.get("end_time")},
                "reason": reason.strip(),
                "event_review_status": status,
                "interpretation": "POSITION_CHANGE_BETWEEN_TRUSTED_OBSERVATIONS" if status == "SUPPORTED" else None,
                "provenance": "HUMAN_REVIEW",
            })
            self._append_unlocked(operation)
        return {"status": "RECORDED", **operation}

    def revoke(self, operation_id: str, *, expected_revision: int) -> dict[str, Any]:
        with self._locked(create=True):
            operations = self._read_operations_unlocked()
            self._check_revision(expected_revision, len(operations))
            target = next((item for item in operations if item.get("operation_id") == operation_id and item.get("action") in {"OBSERVATION_REVIEW", "EVENT_REVIEW"}), None)
            if target is None:
                raise TrustReviewError("OPERATION_NOT_FOUND", "trust review operation does not exist", {"operation_id": operation_id})
            revoked = {item.get("target_operation_id") for item in operations if item.get("action") == "REVOKE"}
            if operation_id in revoked:
                raise TrustReviewError("ALREADY_REVOKED", "trust review operation has already been revoked", {"operation_id": operation_id})
            identity_snapshot = self.identity.snapshot()
            self._validate_source_current()
            operation = self._operation_base("REVOKE", len(operations) + 1, identity_snapshot)
            operation.update({"target_operation_id": operation_id, "revocation_status": "REVOKED"})
            self._append_unlocked(operation)
        return {"status": "REVOKED", **operation}

    def _operation_base(self, action: str, revision: int, identity_snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "operation_id": str(uuid.uuid4()),
            "action": action,
            "source_run_id": self.run_id,
            "source_video_sha256": self.video_sha256,
            "source_p2a_mapping_sha256": self.mapping_sha256,
            "source_identity_overlay_revision": identity_snapshot["overlay_revision"],
            "confirmation_timestamp": _utc_now(),
            "overlay_revision": revision,
            "session_id": f"anonymous-{uuid.uuid4().hex[:12]}",
        }

    def _validate_source_current(self) -> None:
        self.identity.validate_source_current()
        snapshot = self.identity.snapshot()
        if snapshot["overlay_revision"] != self._identity_snapshot["overlay_revision"]:
            raise TrustReviewError("STALE_IDENTITY_REVISION", "identity mapping changed while this trust review was open", {"expected": self._identity_snapshot["overlay_revision"], "current": snapshot["overlay_revision"]})

    @staticmethod
    def _check_revision(expected: int, actual: int) -> None:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected != actual:
            raise TrustReviewError("STALE_REVISION", "trust review revision changed; reload before saving", {"expected_revision": expected, "current_revision": actual})

    def _append_unlocked(self, operation: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_operations(self) -> list[dict[str, Any]]:
        with self._locked(create=False):
            return self._read_operations_unlocked()

    def _read_operations_unlocked(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        operations = []
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    operation = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TrustReviewError("LEDGER_INVALID", "trust review ledger contains invalid JSON", {"line": line_number}) from error
                if not isinstance(operation, dict) or operation.get("source_run_id") != self.run_id or operation.get("source_video_sha256") != self.video_sha256 or operation.get("source_p2a_mapping_sha256") != self.mapping_sha256:
                    raise TrustReviewError("LEDGER_SOURCE_MISMATCH", "trust review ledger contains an operation from another source run", {"line": line_number})
                operations.append(operation)
        return operations

    def _locked(self, *, create: bool):
        return _FileLock(self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock"), create=create)


class _FileLock:
    def __init__(self, path: Path, *, create: bool) -> None:
        self.path = path
        self.create = create
        self.handle = None

    def __enter__(self):
        if not self.create and not self.path.exists():
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _normalize_frames(values: list[int]) -> list[int]:
    if not isinstance(values, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise TrustReviewError("INVALID_REVIEW_FRAMES", "reviewed_frame_indices must be non-negative integers")
    return sorted(set(values))


def _read_json(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrustReviewError("EVENT_SOURCE_INVALID", "frozen MOVE_OBJECT candidate file is unreadable", {"path": str(path)}) from error
    if not isinstance(value, list):
        raise TrustReviewError("EVENT_SOURCE_INVALID", "frozen MOVE_OBJECT candidates must be a JSON array", {"path": str(path)})
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_output_hash(manifest: dict[str, Any], path: Path) -> str | None:
    target = path.resolve()
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for record in stage.get("outputs", []):
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            candidate = Path(record["path"])
            candidate = candidate if candidate.is_absolute() else path.parents[1] / candidate
            if candidate.resolve() == target:
                value = record.get("sha256")
                return value if isinstance(value, str) else None
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
