"""Validate an external F1 interpretation and write a deterministic F2 sidecar."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recallvision.vlm_f1.errors import VlmOutputValidationError
from recallvision.vlm_f1.hashing import package_sha256, sha256_file
from recallvision.vlm_f1.schema import PACKAGE_VERSION, OUTPUT_SCHEMA_VERSION
from recallvision.vlm_f1.validator import validate_vlm_output

from .errors import VlmF2ValidationError
from .schema import (
    ATTRIBUTION_SCOPE_EVENT_LOCAL,
    ATTRIBUTION_SCOPE_VIDEO_LOCAL,
    ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
    IDENTITY_GATE_FAILED_MISSING,
    IDENTITY_GATE_NOT_REQUIRED,
    IdentityValidationAssertion,
    PASS_EVENT_LOCAL,
    PASS_VIDEO_LOCAL,
    VALIDATED_INTERPRETATION_SCHEMA_VERSION,
    VALIDATION_REPORT_SCHEMA_VERSION,
)

_SOURCE_FIELDS = {
    "package_version",
    "package_sha256",
    "video_id",
    "move_object_event_id",
    "target_object_id",
    "interpretation_source",
    "interpretation",
}
_MANIFEST_FIELDS = {
    "package_version",
    "output_schema_version",
    "video_id",
    "move_object_event_id",
    "target_object_id",
    "candidate_labels",
    "candidate_mapping",
    "selected_frame_indices",
}


def ingest_interpretation(
    package_root: str | Path,
    source_document: Mapping[str, Any],
    *,
    identity_assertion: Mapping[str, Any] | IdentityValidationAssertion | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate one external interpretation against one immutable F1 package.

    The returned dictionaries are in-memory sidecar contents. No event record,
    package file, provider, model, or persistent identity is changed here.
    """

    package = Path(package_root)
    manifest = _load_manifest(package)
    source = _validate_source_document(source_document)
    package_hash = package_sha256(package)
    _validate_package_binding(source, manifest, package_hash)

    interpretation = source["interpretation"]
    candidate_labels = manifest["candidate_labels"]
    selected_frames = manifest["selected_frame_indices"]
    try:
        normalized = validate_vlm_output(
            interpretation,
            candidate_labels=candidate_labels,
            supplied_frames=selected_frames,
        )
    except VlmOutputValidationError as error:
        raise VlmF2ValidationError(str(error)) from error
    _validate_cross_field_semantics(normalized)

    status = normalized["status"]
    candidate_label = normalized["candidate_label"]
    is_supported = status == "SUPPORTED_AGENT"
    mapped_person_id = manifest["candidate_mapping"].get(candidate_label) if is_supported else None
    if is_supported and mapped_person_id is None:
        raise VlmF2ValidationError(
            f"candidate_label {candidate_label!r} has no persistent ID in the F1 manifest"
        )

    assertion = _coerce_identity_assertion(identity_assertion)
    if not is_supported:
        identity_gate_status = IDENTITY_GATE_NOT_REQUIRED
        attribution_ready = False
    else:
        identity_gate_status = _identity_gate_status(
            assertion,
            video_id=manifest["video_id"],
            event_id=manifest["move_object_event_id"],
            mapped_person_id=mapped_person_id,
        )
        attribution_ready = identity_gate_status in {PASS_EVENT_LOCAL, PASS_VIDEO_LOCAL}

    validated_status = (
        "VALIDATED_SUPPORTED_AGENT" if attribution_ready else "SUPPORTED_AGENT_IDENTITY_BLOCKED"
    ) if is_supported else status
    validated = {
        "validated_interpretation_schema_version": VALIDATED_INTERPRETATION_SCHEMA_VERSION,
        "package_version": manifest["package_version"],
        "package_sha256": package_hash,
        "video_id": manifest["video_id"],
        "move_object_event_id": manifest["move_object_event_id"],
        "target_object_id": manifest["target_object_id"],
        "interpretation_source": source["interpretation_source"],
        "source_status": status,
        "validated_status": validated_status,
        "candidate_label": candidate_label,
        "mapped_persistent_person_id": mapped_person_id,
        "interaction_supported": normalized["interaction_supported"],
        "causal_support": normalized["causal_support"],
        "evidence_frames": list(normalized["evidence_frames"]),
        "brief_reason": normalized["brief_reason"],
        "identity_gate_status": identity_gate_status,
        "attribution_ready": attribution_ready,
        "identity_assertion": assertion.to_record() if assertion is not None else None,
    }
    report = {
        "validation_report_schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "package_binding_status": "PASS",
        "source_schema_status": "PASS",
        "candidate_validation_status": "PASS" if is_supported else "NOT_REQUIRED",
        "evidence_validation_status": "PASS" if normalized["evidence_frames"] else "NOT_REQUIRED",
        "identity_gate_status": identity_gate_status,
        "attribution_ready": attribution_ready,
        "vlm_inference": False,
        "provider_dependency": False,
        "event_mutation": False,
        "person_ids_populated": False,
        "who_moved_object_implemented": False,
    }
    return {
        "source_interpretation": dict(source),
        "validated_agent_interpretation": validated,
        "validation_report": report,
    }


def write_sidecar(result: Mapping[str, Mapping[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    """Write the three F2 JSON sidecars with stable formatting and return paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "source_interpretation": "source_interpretation.json",
        "validated_agent_interpretation": "validated_agent_interpretation.json",
        "validation_report": "validation_report.json",
    }
    paths: dict[str, Path] = {}
    for key, filename in names.items():
        value = result.get(key)
        if not isinstance(value, Mapping):
            raise VlmF2ValidationError(f"sidecar result is missing {key}")
        path = destination / filename
        path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[key + "_path"] = path
    validated_path = paths["validated_agent_interpretation_path"]
    return {
        "sidecar_dir": str(destination),
        **{key: str(path) for key, path in paths.items()},
        "sidecar_sha256": sha256_file(validated_path),
        "attribution_ready": bool(result["validated_agent_interpretation"]["attribution_ready"]),
    }


def _load_manifest(package: Path) -> dict[str, Any]:
    if not package.is_dir():
        raise VlmF2ValidationError(f"F1 package directory does not exist: {package}")
    path = package / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VlmF2ValidationError(f"unable to load F1 manifest: {path}: {error}") from error
    if not isinstance(value, dict):
        raise VlmF2ValidationError("F1 manifest must be an object")
    missing = _MANIFEST_FIELDS - set(value)
    if missing:
        raise VlmF2ValidationError(f"F1 manifest is missing fields: {sorted(missing)}")
    if value["package_version"] != PACKAGE_VERSION:
        raise VlmF2ValidationError("F1 manifest package_version is not supported")
    if value["output_schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise VlmF2ValidationError("F1 manifest output_schema_version is not supported")
    for field in ("video_id", "move_object_event_id", "target_object_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise VlmF2ValidationError(f"F1 manifest {field} must be a non-empty string")
    labels = value["candidate_labels"]
    mapping = value["candidate_mapping"]
    if not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels):
        raise VlmF2ValidationError("F1 manifest candidate_labels must be a list of strings")
    if not isinstance(mapping, dict) or set(mapping) != set(labels):
        raise VlmF2ValidationError("F1 manifest candidate_mapping must match candidate_labels")
    if any(not isinstance(label, str) or not isinstance(person_id, str) or not person_id for label, person_id in mapping.items()):
        raise VlmF2ValidationError("F1 manifest candidate_mapping must map labels to IDs")
    frames = value["selected_frame_indices"]
    if not isinstance(frames, list) or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames):
        raise VlmF2ValidationError("F1 manifest selected_frame_indices must be a list of integers")
    return value


def _validate_source_document(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise VlmF2ValidationError("source interpretation must be an object")
    unknown = set(source) - _SOURCE_FIELDS
    missing = _SOURCE_FIELDS - set(source)
    if unknown:
        raise VlmF2ValidationError(f"source interpretation has unknown fields: {sorted(unknown)}")
    if missing:
        raise VlmF2ValidationError(f"source interpretation is missing fields: {sorted(missing)}")
    for field in (
        "package_version",
        "package_sha256",
        "video_id",
        "move_object_event_id",
        "target_object_id",
        "interpretation_source",
    ):
        if not isinstance(source[field], str) or not source[field].strip():
            raise VlmF2ValidationError(f"source interpretation {field} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source["package_sha256"]):
        raise VlmF2ValidationError("source interpretation package_sha256 must be a SHA256 hex string")
    if not isinstance(source["interpretation"], Mapping):
        raise VlmF2ValidationError("source interpretation interpretation must be an object")
    return dict(source)


def _validate_package_binding(source: Mapping[str, Any], manifest: Mapping[str, Any], package_hash: str) -> None:
    if source["package_sha256"] != package_hash:
        raise VlmF2ValidationError(
            f"package SHA256 mismatch: supplied {source['package_sha256']}, actual {package_hash}"
        )
    for field in ("package_version", "video_id", "move_object_event_id", "target_object_id"):
        if source[field] != manifest[field]:
            raise VlmF2ValidationError(f"package binding mismatch for {field}")


def _validate_cross_field_semantics(result: Mapping[str, Any]) -> None:
    if result["status"] == "SUPPORTED_AGENT":
        if not result["causal_support"]:
            raise VlmF2ValidationError("SUPPORTED_AGENT requires causal_support")
        if not result["evidence_frames"]:
            raise VlmF2ValidationError("SUPPORTED_AGENT requires non-empty evidence_frames")


def _coerce_identity_assertion(
    assertion: Mapping[str, Any] | IdentityValidationAssertion | None,
) -> IdentityValidationAssertion | None:
    if assertion is None:
        return None
    if isinstance(assertion, IdentityValidationAssertion):
        # Re-parse even typed instances because dataclasses can be constructed
        # directly with invalid runtime values (for example 1 instead of True).
        return IdentityValidationAssertion.from_mapping(assertion.to_record())
    return IdentityValidationAssertion.from_mapping(assertion)


def _identity_gate_status(
    assertion: IdentityValidationAssertion | None,
    *,
    video_id: str,
    event_id: str,
    mapped_person_id: str,
) -> str:
    if assertion is None:
        return IDENTITY_GATE_FAILED_MISSING
    if assertion.video_id != video_id:
        return "FAILED_VIDEO_SCOPE"
    if assertion.persistent_person_id != mapped_person_id:
        return "FAILED_PERSON_MISMATCH"
    if not assertion.ready_for_attribution:
        return "FAILED_NOT_READY"
    if assertion.identity_level == ID_LEVEL_2_EVENT_LOCAL_VALIDATED:
        if assertion.attribution_scope != ATTRIBUTION_SCOPE_EVENT_LOCAL:
            return "FAILED_SCOPE"
        if assertion.move_object_event_id != event_id:
            return "FAILED_EVENT_SCOPE"
        return PASS_EVENT_LOCAL
    if assertion.identity_level == ID_LEVEL_3_VIDEO_LOCAL_VALIDATED:
        if assertion.attribution_scope != ATTRIBUTION_SCOPE_VIDEO_LOCAL:
            return "FAILED_SCOPE"
        return PASS_VIDEO_LOCAL
    return "FAILED_LEVEL"
