"""Versioned F3 authorization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

AUTHORIZATION_VERSION = "VLM-F3-AUTHORIZATION-2026-10-04-v1"
AUTHORIZATION_REPORT_SCHEMA_VERSION = "VLM-F3-AUTHORIZATION-REPORT-2026-10-04-v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "VLM-F3-SOURCE-MANIFEST-2026-10-04-v1"

AUTHORIZED = "AUTHORIZED"
BLOCKED_NOT_ATTRIBUTION_READY = "BLOCKED_NOT_ATTRIBUTION_READY"
BLOCKED_SOURCE_MISMATCH = "BLOCKED_SOURCE_MISMATCH"
BLOCKED_IDENTITY_GATE = "BLOCKED_IDENTITY_GATE"
BLOCKED_INVALID_INTERPRETATION = "BLOCKED_INVALID_INTERPRETATION"
BLOCKED_ALREADY_ATTRIBUTED = "BLOCKED_ALREADY_ATTRIBUTED"

AUTHORIZATION_STATUSES = (
    AUTHORIZED,
    BLOCKED_NOT_ATTRIBUTION_READY,
    BLOCKED_SOURCE_MISMATCH,
    BLOCKED_IDENTITY_GATE,
    BLOCKED_INVALID_INTERPRETATION,
    BLOCKED_ALREADY_ATTRIBUTED,
)


@dataclass(frozen=True)
class AgentAttributionAuthorization:
    """Provider-independent authorization decision and provenance summary."""

    source_event_id: str
    source_event_hash: str
    source_event_file_sha256: str
    f1_package_sha256: str
    f2_sidecar_sha256: str
    source_status: str | None
    candidate_label: str | None
    mapped_persistent_person_id: str | None
    identity_level: str | None
    identity_scope: str | None
    identity_gate_status: str | None
    attribution_ready: bool
    authorization_status: str
    authorization_reason: str
    derived_event_hash: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "authorization_version": AUTHORIZATION_VERSION,
            "source_event_id": self.source_event_id,
            "source_event_hash": self.source_event_hash,
            "source_event_file_sha256": self.source_event_file_sha256,
            "f1_package_sha256": self.f1_package_sha256,
            "f2_sidecar_sha256": self.f2_sidecar_sha256,
            "source_status": self.source_status,
            "candidate_label": self.candidate_label,
            "mapped_persistent_person_id": self.mapped_persistent_person_id,
            "identity_level": self.identity_level,
            "identity_scope": self.identity_scope,
            "identity_gate_status": self.identity_gate_status,
            "attribution_ready": self.attribution_ready,
            "authorization_status": self.authorization_status,
            "authorization_reason": self.authorization_reason,
            "derived_event_hash": self.derived_event_hash,
        }
