"""Fail-closed authorization from F2 validated interpretation to derived event."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.vlm_f1.hashing import package_sha256, sha256_file
from recallvision.vlm_f1.schema import OUTPUT_SCHEMA_VERSION, PACKAGE_VERSION
from recallvision.vlm_f2.schema import (
    ATTRIBUTION_SCOPE_EVENT_LOCAL,
    ATTRIBUTION_SCOPE_VIDEO_LOCAL,
    ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
    VALIDATED_INTERPRETATION_SCHEMA_VERSION,
    VALIDATION_REPORT_SCHEMA_VERSION,
)

from .errors import VlmF3InputError
from .fingerprint import event_fingerprint
from .schema import (
    AUTHORIZATION_REPORT_SCHEMA_VERSION,
    AUTHORIZED,
    BLOCKED_ALREADY_ATTRIBUTED,
    BLOCKED_IDENTITY_GATE,
    BLOCKED_INVALID_INTERPRETATION,
    BLOCKED_NOT_ATTRIBUTION_READY,
    BLOCKED_SOURCE_MISMATCH,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    AgentAttributionAuthorization,
)
from .source import event_video_id, load_event


def authorize_attribution(
    *,
    source_events_path: str | Path,
    event_id: str,
    f1_package: str | Path,
    f2_sidecar_path: str | Path,
    f2_report_path: str | Path,
) -> dict[str, Any]:
    """Authorize one F2 result without mutating the source event artifact."""

    source_path = Path(source_events_path)
    package_path = Path(f1_package)
    sidecar_path = Path(f2_sidecar_path)
    report_path = Path(f2_report_path)
    source_event = load_event(source_path, event_id)
    source_event_hash = event_fingerprint(source_event)
    source_file_hash = sha256_file(source_path)
    sidecar = _load_object(sidecar_path, "F2 validated interpretation")
    report = _load_object(report_path, "F2 validation report")
    manifest = _load_manifest(package_path)
    package_hash = package_sha256(package_path)
    sidecar_hash = sha256_file(sidecar_path)
    provenance = _provenance(
        source_path=source_path,
        source_event=source_event,
        source_event_hash=source_event_hash,
        source_file_hash=source_file_hash,
        package_path=package_path,
        package_hash=package_hash,
        manifest=manifest,
        sidecar_path=sidecar_path,
        sidecar_hash=sidecar_hash,
        report_path=report_path,
        report_hash=sha256_file(report_path),
    )

    if source_event.person_ids:
        return _result(
            provenance,
            status=BLOCKED_ALREADY_ATTRIBUTED,
            reason="source MOVE_OBJECT already contains person_ids; F3 will not overwrite or merge",
            sidecar=sidecar,
        )
    if source_event.event_type != EventType.MOVE_OBJECT:
        return _result(
            provenance,
            status=BLOCKED_SOURCE_MISMATCH,
            reason="source event is not EventType.MOVE_OBJECT",
            sidecar=sidecar,
        )
    source_video_id = event_video_id(source_event)
    if source_video_id is None:
        return _result(provenance, status=BLOCKED_SOURCE_MISMATCH, reason="source event has no video ID", sidecar=sidecar)

    source_artifacts = manifest.get("source_artifacts")
    event_artifact = source_artifacts.get("events") if isinstance(source_artifacts, dict) else None
    if not isinstance(event_artifact, dict) or Path(event_artifact.get("path", "")).name != source_path.name:
        return _result(
            provenance,
            status=BLOCKED_SOURCE_MISMATCH,
            reason="source event artifact path does not match the supplied P2D event file",
            sidecar=sidecar,
        )
    if event_artifact.get("sha256") != source_file_hash:
        return _result(
            provenance,
            status=BLOCKED_SOURCE_MISMATCH,
            reason="source event artifact hash (SHA256) does not match the F1 provenance manifest",
            sidecar=sidecar,
        )
    binding_mismatch = _source_binding_mismatch(source_event, source_video_id, sidecar, manifest)
    if binding_mismatch:
        return _result(provenance, status=BLOCKED_SOURCE_MISMATCH, reason=binding_mismatch, sidecar=sidecar)
    if sidecar.get("package_sha256") != package_hash or sidecar.get("package_version") != PACKAGE_VERSION:
        return _result(
            provenance,
            status=BLOCKED_INVALID_INTERPRETATION,
            reason="F2 sidecar package binding does not match the supplied F1 package",
            sidecar=sidecar,
        )

    invalid_reason = _f2_validation_failure(sidecar, report, manifest)
    if invalid_reason:
        if (
            sidecar.get("validated_status") == "VALIDATED_SUPPORTED_AGENT"
            and _is_identity_gate_failure(sidecar, report)
        ):
            status = BLOCKED_IDENTITY_GATE
        elif sidecar.get("attribution_ready") is False or report.get("attribution_ready") is False:
            status = BLOCKED_NOT_ATTRIBUTION_READY
        elif sidecar.get("source_status") in {
            "AMBIGUOUS_AGENT",
            "NO_SUPPORTED_AGENT",
            "INSUFFICIENT_VISUAL_EVIDENCE",
        }:
            status = BLOCKED_NOT_ATTRIBUTION_READY
        else:
            status = BLOCKED_INVALID_INTERPRETATION
        return _result(provenance, status=status, reason=invalid_reason, sidecar=sidecar)

    derived = copy.deepcopy(source_event)
    derived.person_ids = [sidecar["mapped_persistent_person_id"]]
    return _result(
        provenance,
        status=AUTHORIZED,
        reason="F2 validated supported agent passed all F3 authorization preconditions",
        sidecar=sidecar,
        derived_event=derived,
    )


def write_artifacts(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write an F3 report, source manifest, and derived event when authorized."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record = copy.deepcopy(result["authorization_record"])
    derived = result.get("derived_event")
    derived_path = destination / "attributed_move_object_event.json"
    derived_hash: str | None = None
    if isinstance(derived, Mapping):
        _write_json(derived_path, derived)
        derived_hash = sha256_file(derived_path)
    elif derived_path.exists():
        # This is a generated F3 artifact, and keeping it beside a blocked
        # report would make a failed authorization look usable.
        derived_path.unlink()
        derived_path = None
    else:
        derived_path = None
    record["derived_event_hash"] = derived_hash
    authorization_report = {
        "authorization_report_schema_version": AUTHORIZATION_REPORT_SCHEMA_VERSION,
        **record,
        "vlm_inference": False,
        "provider_dependency_added": False,
        "source_event_mutated": False,
        "who_moved_object_implemented": False,
    }
    authorization_path = destination / "authorization_report.json"
    _write_json(authorization_path, authorization_report)
    authorization_hash = sha256_file(authorization_path)
    provenance = copy.deepcopy(result["provenance"])
    source_manifest = {
        "source_manifest_schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "authorization_version": record["authorization_version"],
        "source_event": {
            "event_id": record["source_event_id"],
            "canonical_event_sha256": record["source_event_hash"],
            "source_file_sha256": record["source_event_file_sha256"],
            "path": provenance["source_event_path"],
        },
        "f1_package": provenance["f1_package"],
        "f1_source_artifacts": provenance["f1_source_artifacts"],
        "f2_validated_sidecar": provenance["f2_validated_sidecar"],
        "f2_validation_report": provenance["f2_validation_report"],
        "derived_event": {
            "file": derived_path.name if derived_path is not None else None,
            "sha256": derived_hash,
        },
        "authorization_report": {
            "file": authorization_path.name,
            "sha256": authorization_hash,
        },
    }
    manifest_path = destination / "source_manifest.json"
    _write_json(manifest_path, source_manifest)
    return {
        "output_dir": str(destination),
        "derived_event_path": str(derived_path) if derived_path is not None else None,
        "authorization_report_path": str(authorization_path),
        "source_manifest_path": str(manifest_path),
        "derived_event_sha256": derived_hash,
        "authorization_report_sha256": authorization_hash,
        "source_manifest_sha256": sha256_file(manifest_path),
        "authorization_status": record["authorization_status"],
    }


def _result(
    provenance: dict[str, Any],
    *,
    status: str,
    reason: str,
    sidecar: Mapping[str, Any],
    derived_event: EventRecord | None = None,
) -> dict[str, Any]:
    assertion = sidecar.get("identity_assertion")
    record = AgentAttributionAuthorization(
        source_event_id=provenance["source_event_id"],
        source_event_hash=provenance["source_event_hash"],
        source_event_file_sha256=provenance["source_event_file_sha256"],
        f1_package_sha256=provenance["f1_package"]["sha256"],
        f2_sidecar_sha256=provenance["f2_validated_sidecar"]["sha256"],
        source_status=sidecar.get("source_status") if isinstance(sidecar.get("source_status"), str) else None,
        candidate_label=sidecar.get("candidate_label") if isinstance(sidecar.get("candidate_label"), str) else None,
        mapped_persistent_person_id=(
            sidecar.get("mapped_persistent_person_id")
            if isinstance(sidecar.get("mapped_persistent_person_id"), str)
            else None
        ),
        identity_level=assertion.get("identity_level") if isinstance(assertion, dict) else None,
        identity_scope=assertion.get("attribution_scope") if isinstance(assertion, dict) else None,
        identity_gate_status=sidecar.get("identity_gate_status") if isinstance(sidecar.get("identity_gate_status"), str) else None,
        attribution_ready=sidecar.get("attribution_ready") is True,
        authorization_status=status,
        authorization_reason=reason,
    )
    return {
        "authorization_status": status,
        "authorization_reason": reason,
        "authorization_record": record.to_record(),
        "provenance": provenance,
        "derived_event": derived_event.to_record() if derived_event is not None else None,
    }


def _f2_validation_failure(
    sidecar: Mapping[str, Any], report: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str | None:
    shape_failure = _validated_sidecar_shape_failure(sidecar)
    if shape_failure:
        return shape_failure
    if report.get("validation_report_schema_version") != VALIDATION_REPORT_SCHEMA_VERSION:
        return "F2 validation report schema version is unsupported"
    if sidecar.get("validated_interpretation_schema_version") != VALIDATED_INTERPRETATION_SCHEMA_VERSION:
        return "F2 validated interpretation schema version is unsupported"
    required_report_passes = {
        "package_binding_status": "PASS",
        "source_schema_status": "PASS",
        "candidate_validation_status": "PASS",
        "evidence_validation_status": "PASS",
    }
    for field, expected in required_report_passes.items():
        if report.get(field) != expected:
            return f"F2 validation report {field} is not {expected}"
    for field in (
        "vlm_inference",
        "provider_dependency",
        "event_mutation",
        "person_ids_populated",
        "who_moved_object_implemented",
    ):
        if report.get(field) is not False:
            return f"F2 validation report safety flag {field} is not false"
    if sidecar.get("source_status") != "SUPPORTED_AGENT":
        if sidecar.get("source_status") in {
            "AMBIGUOUS_AGENT",
            "NO_SUPPORTED_AGENT",
            "INSUFFICIENT_VISUAL_EVIDENCE",
        }:
            return "F2 abstention status cannot authorize attribution"
        return "F2 source_status is not supported"
    if sidecar.get("validated_status") != "VALIDATED_SUPPORTED_AGENT":
        return "F2 validated_status is not VALIDATED_SUPPORTED_AGENT"
    if sidecar.get("identity_gate_status") != report.get("identity_gate_status"):
        return "F2 identity gate status disagrees between sidecar and report"
    if report.get("attribution_ready") is not True or sidecar.get("attribution_ready") is not True:
        return "F2 attribution_ready is not true in all validation records"
    candidate_label = sidecar.get("candidate_label")
    if not isinstance(candidate_label, str):
        return "F2 candidate_label is not a string"
    if candidate_label not in manifest.get("candidate_mapping", {}):
        return "F2 candidate_label is absent from the F1 closed mapping"
    mapped = sidecar.get("mapped_persistent_person_id")
    if not isinstance(mapped, str) or not mapped:
        return "F2 mapped_persistent_person_id is missing"
    if manifest["candidate_mapping"][sidecar["candidate_label"]] != mapped:
        return "F2 mapped person ID disagrees with the F1 candidate mapping"
    evidence_frames = sidecar.get("evidence_frames")
    if not isinstance(evidence_frames, list) or not evidence_frames or any(
        isinstance(frame, bool) or not isinstance(frame, int) for frame in evidence_frames
    ):
        return "F2 evidence_frames is invalid or empty"
    if not set(evidence_frames).issubset(set(manifest["selected_frame_indices"])):
        return "F2 evidence_frames is not a subset of F1 selected_frame_indices"
    if sidecar.get("causal_support") is not True:
        return "F2 causal_support is not true"
    assertion = sidecar.get("identity_assertion")
    if not isinstance(assertion, dict):
        return "F2 identity assertion is missing"
    if assertion.get("ready_for_attribution") is not True:
        return "F2 identity assertion is not ready_for_attribution"
    if assertion.get("persistent_person_id") != mapped:
        return "F2 identity assertion person does not match the mapped person"
    if assertion.get("video_id") != manifest["video_id"]:
        return "F2 identity assertion video scope does not match the F1 package"
    gate = sidecar.get("identity_gate_status")
    if assertion.get("identity_level") == ID_LEVEL_2_EVENT_LOCAL_VALIDATED:
        if assertion.get("attribution_scope") != ATTRIBUTION_SCOPE_EVENT_LOCAL:
            return "F2 event-local identity assertion has the wrong scope"
        if assertion.get("move_object_event_id") != manifest["move_object_event_id"]:
            return "F2 event-local identity assertion has the wrong event"
        if gate != "PASS_EVENT_LOCAL":
            return "F2 event-local identity gate did not pass"
    elif assertion.get("identity_level") == ID_LEVEL_3_VIDEO_LOCAL_VALIDATED:
        if assertion.get("attribution_scope") != ATTRIBUTION_SCOPE_VIDEO_LOCAL:
            return "F2 video-local identity assertion has the wrong scope"
        if gate != "PASS_VIDEO_LOCAL":
            return "F2 video-local identity gate did not pass"
    else:
        return "F2 identity level is not attribution-valid"
    return None


def _validated_sidecar_shape_failure(sidecar: Mapping[str, Any]) -> str | None:
    required = {
        "validated_interpretation_schema_version",
        "package_version",
        "package_sha256",
        "video_id",
        "move_object_event_id",
        "target_object_id",
        "interpretation_source",
        "source_status",
        "validated_status",
        "candidate_label",
        "mapped_persistent_person_id",
        "interaction_supported",
        "causal_support",
        "evidence_frames",
        "brief_reason",
        "identity_gate_status",
        "attribution_ready",
        "identity_assertion",
    }
    missing = required - set(sidecar)
    if missing:
        return f"F2 validated sidecar is missing fields: {sorted(missing)}"
    for field in (
        "package_version",
        "package_sha256",
        "video_id",
        "move_object_event_id",
        "target_object_id",
        "interpretation_source",
        "source_status",
        "validated_status",
        "identity_gate_status",
        "brief_reason",
    ):
        if not isinstance(sidecar[field], str) or not sidecar[field].strip():
            return f"F2 validated sidecar field {field} must be a non-empty string"
    if not isinstance(sidecar["interaction_supported"], bool):
        return "F2 validated sidecar interaction_supported must be boolean"
    if not isinstance(sidecar["causal_support"], bool):
        return "F2 validated sidecar causal_support must be boolean"
    if not isinstance(sidecar["attribution_ready"], bool):
        return "F2 validated sidecar attribution_ready must be boolean"
    frames = sidecar["evidence_frames"]
    if not isinstance(frames, list) or any(
        isinstance(frame, bool) or not isinstance(frame, int) for frame in frames
    ):
        return "F2 validated sidecar evidence_frames must be a list of integers"
    assertion = sidecar["identity_assertion"]
    if assertion is not None and not isinstance(assertion, dict):
        return "F2 validated sidecar identity_assertion must be an object or null"
    if isinstance(assertion, dict):
        assertion_required = {
            "video_id",
            "move_object_event_id",
            "persistent_person_id",
            "identity_level",
            "attribution_scope",
            "ready_for_attribution",
            "validation_source",
        }
        missing_assertion = assertion_required - set(assertion)
        if missing_assertion:
            return f"F2 identity assertion is missing fields: {sorted(missing_assertion)}"
        for field in (
            "video_id",
            "move_object_event_id",
            "persistent_person_id",
            "identity_level",
            "attribution_scope",
            "validation_source",
        ):
            if not isinstance(assertion[field], str) or not assertion[field].strip():
                return f"F2 identity assertion field {field} must be a non-empty string"
        if not isinstance(assertion["ready_for_attribution"], bool):
            return "F2 identity assertion ready_for_attribution must be boolean"
    return None


def _is_identity_gate_failure(sidecar: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    gate = sidecar.get("identity_gate_status") or report.get("identity_gate_status")
    if isinstance(gate, str) and gate.startswith("FAILED_"):
        return True
    assertion = sidecar.get("identity_assertion")
    if not isinstance(assertion, dict):
        return False
    if assertion.get("identity_level") == ID_LEVEL_2_EVENT_LOCAL_VALIDATED:
        return (
            assertion.get("attribution_scope") != ATTRIBUTION_SCOPE_EVENT_LOCAL
            or assertion.get("move_object_event_id") != sidecar.get("move_object_event_id")
        )
    if assertion.get("identity_level") == ID_LEVEL_3_VIDEO_LOCAL_VALIDATED:
        return assertion.get("attribution_scope") != ATTRIBUTION_SCOPE_VIDEO_LOCAL
    return True


def _source_binding_mismatch(
    source_event: EventRecord,
    source_video_id: str,
    sidecar: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str | None:
    if source_event.event_id != manifest.get("move_object_event_id"):
        return "source event ID does not match the F1 package event ID"
    if source_video_id != manifest.get("video_id"):
        return "source event video ID does not match the F1 package video ID"
    if manifest.get("target_object_id") not in source_event.object_ids:
        return "F1 target object is absent from the source event"
    for field in ("video_id", "move_object_event_id", "target_object_id"):
        if sidecar.get(field) != manifest.get(field):
            return f"F2 sidecar {field} does not match the F1 package"
    return None


def _provenance(
    *,
    source_path: Path,
    source_event: EventRecord,
    source_event_hash: str,
    source_file_hash: str,
    package_path: Path,
    package_hash: str,
    manifest: Mapping[str, Any],
    sidecar_path: Path,
    sidecar_hash: str,
    report_path: Path,
    report_hash: str,
) -> dict[str, Any]:
    return {
        "source_event_id": source_event.event_id,
        "video_id": manifest.get("video_id"),
        "source_event_hash": source_event_hash,
        "source_event_file_sha256": source_file_hash,
        "source_event_path": str(source_path),
        "f1_package": {"path": str(package_path), "sha256": package_hash},
        "f1_source_artifacts": manifest.get("source_artifacts", {}),
        "f2_validated_sidecar": {"path": str(sidecar_path), "sha256": sidecar_hash},
        "f2_validation_report": {"path": str(report_path), "sha256": report_hash},
    }


def _load_manifest(package: Path) -> dict[str, Any]:
    path = package / "manifest.json"
    if not package.is_dir() or not path.is_file():
        raise VlmF3InputError(f"F1 package manifest does not exist: {path}")
    value = _load_object(path, "F1 package manifest")
    required = {
        "package_version",
        "output_schema_version",
        "video_id",
        "move_object_event_id",
        "target_object_id",
        "candidate_labels",
        "candidate_mapping",
        "selected_frame_indices",
        "source_artifacts",
    }
    if not required.issubset(value):
        raise VlmF3InputError(f"F1 package manifest is missing fields: {sorted(required - set(value))}")
    if value["package_version"] != PACKAGE_VERSION or value["output_schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise VlmF3InputError("F1 package manifest version is unsupported")
    for field in ("video_id", "move_object_event_id", "target_object_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise VlmF3InputError(f"F1 package manifest {field} must be a non-empty string")
    labels = value["candidate_labels"]
    mapping = value["candidate_mapping"]
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not re.fullmatch(r"P[1-9][0-9]*", label) for label in labels
    ) or len(set(labels)) != len(labels):
        raise VlmF3InputError("F1 package manifest candidate_labels is invalid")
    if not isinstance(mapping, dict) or set(mapping) != set(labels):
        raise VlmF3InputError("F1 package manifest candidate_mapping does not match labels")
    if any(
        not isinstance(label, str)
        or not isinstance(person_id, str)
        or not person_id.strip()
        for label, person_id in mapping.items()
    ):
        raise VlmF3InputError("F1 package manifest candidate_mapping values are invalid")
    frames = value["selected_frame_indices"]
    if not isinstance(frames, list) or any(
        isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames
    ):
        raise VlmF3InputError("F1 package manifest selected_frame_indices is invalid")
    source_artifacts = value["source_artifacts"]
    events_artifact = source_artifacts.get("events") if isinstance(source_artifacts, dict) else None
    if not isinstance(events_artifact, dict) or not isinstance(events_artifact.get("path"), str) or not events_artifact["path"].strip():
        raise VlmF3InputError("F1 package manifest source_artifacts.events is invalid")
    if not isinstance(events_artifact.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", events_artifact["sha256"]
    ):
        raise VlmF3InputError("F1 package manifest source_artifacts.events.sha256 is invalid")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VlmF3InputError(f"unable to load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise VlmF3InputError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
