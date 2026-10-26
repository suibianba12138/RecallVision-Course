"""Prospective EVAL-D0 provenance records and frozen-method enforcement."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ANNOTATION_SCHEMA_VERSION, is_sha256

FROZEN_V1_METHOD_HEAD = "bc9ca6370962df108bd44d4a50cbdf2132234e8a"
FROZEN_EVALUATION_PROTOCOL_HEAD = "e3d66acea82033e7d1875b87cf2504e928a43d84"
PROVENANCE_SCHEMA_VERSION = "eval_d0_provenance_v1"
REQUIRED_ARTIFACT_KEYS = (
    "observations",
    "persistent_memory",
    "state_timeline",
    "events",
    "f1",
    "f2",
    "f3",
    "query_output",
)


class EvaluationProvenanceError(ValueError):
    """Raised when a prospective evaluation record cannot be trusted."""


def build_provenance_record(
    *,
    video_path: str | Path,
    video_id: str,
    split: str,
    method_head: str,
    evaluation_protocol_head: str,
    tooling_head: str,
    configuration_hashes: Mapping[str, str],
    artifact_hashes: Mapping[str, str | None],
    annotation_schema_version: str,
) -> dict[str, Any]:
    """Build a complete record and fail closed if the V1 source head differs."""

    _check_method_head(method_head)
    if evaluation_protocol_head != FROZEN_EVALUATION_PROTOCOL_HEAD:
        raise EvaluationProvenanceError("evaluation protocol HEAD does not match the frozen protocol")
    path = Path(video_path)
    if not path.is_file():
        raise EvaluationProvenanceError(f"video path does not exist: {path}")
    record = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "video_id": video_id,
        "split": split,
        "video_sha256": sha256_file(path),
        "method_head": method_head,
        "evaluation_protocol_head": evaluation_protocol_head,
        "tooling_head": tooling_head,
        "configuration_hashes": dict(configuration_hashes),
        "artifact_hashes": dict(artifact_hashes),
        "annotation_schema_version": annotation_schema_version,
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    validate_provenance_record(record)
    return record


def validate_provenance_record(value: Mapping[str, Any]) -> None:
    """Validate all required source, method, protocol, and artifact hashes."""

    required = {
        "provenance_schema_version",
        "video_id",
        "split",
        "video_sha256",
        "method_head",
        "evaluation_protocol_head",
        "tooling_head",
        "configuration_hashes",
        "artifact_hashes",
        "annotation_schema_version",
        "captured_at_utc",
    }
    _mapping(value, "provenance")
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise EvaluationProvenanceError(f"provenance is missing fields: {sorted(missing)}")
    if unknown:
        raise EvaluationProvenanceError(f"provenance has unknown fields: {sorted(unknown)}")
    if value["provenance_schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise EvaluationProvenanceError("provenance schema version is not supported")
    _text(value["video_id"], "video_id")
    if value["split"] not in {"DEV", "EVAL"}:
        raise EvaluationProvenanceError("provenance split is not allowed")
    if not is_sha256(value["video_sha256"]):
        raise EvaluationProvenanceError("video_sha256 must be a SHA-256 hex digest")
    _check_method_head(value["method_head"])
    if value["evaluation_protocol_head"] != FROZEN_EVALUATION_PROTOCOL_HEAD:
        raise EvaluationProvenanceError("evaluation protocol HEAD does not match the frozen protocol")
    _commit(value["tooling_head"], "tooling_head")
    _hash_mapping(value["configuration_hashes"], "configuration_hashes", allow_none=False)
    artifacts = value["artifact_hashes"]
    if not isinstance(artifacts, Mapping):
        raise EvaluationProvenanceError("artifact_hashes must be an object")
    if set(artifacts) != set(REQUIRED_ARTIFACT_KEYS):
        raise EvaluationProvenanceError(
            "artifact_hashes must include exactly observations, persistent_memory, state_timeline, events, f1, f2, f3, query_output"
        )
    _hash_mapping(artifacts, "artifact_hashes", allow_none=True)
    if value["annotation_schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise EvaluationProvenanceError("annotation_schema_version is not supported")
    captured = value["captured_at_utc"]
    if not isinstance(captured, str) or not captured.endswith("Z"):
        raise EvaluationProvenanceError("captured_at_utc must be an ISO-8601 UTC timestamp")


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks without loading a video into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_method_head(value: Any) -> None:
    if value != FROZEN_V1_METHOD_HEAD:
        raise EvaluationProvenanceError(
            "frozen method HEAD mismatch; refuse to evaluate modified V1 method"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationProvenanceError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationProvenanceError(f"{label} must be a non-empty string")


def _commit(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise EvaluationProvenanceError(f"{label} must be a 40-character lowercase Git commit")


def _hash_mapping(value: Any, label: str, *, allow_none: bool) -> None:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationProvenanceError(f"{label} must be a non-empty object")
    for key, digest in value.items():
        _text(key, f"{label} key")
        if digest is None and allow_none:
            continue
        if not is_sha256(digest):
            raise EvaluationProvenanceError(f"{label}.{key} must be a SHA-256 hex digest or null")
