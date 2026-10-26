"""Versioned, provider-independent F2 schemas and identity assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import VlmF2ValidationError

VALIDATED_INTERPRETATION_SCHEMA_VERSION = "VLM-F2-VALIDATED-INTERPRETATION-2026-10-04-v1"
VALIDATION_REPORT_SCHEMA_VERSION = "VLM-F2-VALIDATION-REPORT-2026-10-04-v1"

ID_LEVEL_0_INVALID = "ID_LEVEL_0_INVALID"
ID_LEVEL_1_UNCERTAIN = "ID_LEVEL_1_UNCERTAIN"
ID_LEVEL_2_EVENT_LOCAL_VALIDATED = "ID_LEVEL_2_EVENT_LOCAL_VALIDATED"
ID_LEVEL_3_VIDEO_LOCAL_VALIDATED = "ID_LEVEL_3_VIDEO_LOCAL_VALIDATED"
IDENTITY_LEVELS = (
    ID_LEVEL_0_INVALID,
    ID_LEVEL_1_UNCERTAIN,
    ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
)

ATTRIBUTION_SCOPE_EVENT_LOCAL = "EVENT_LOCAL"
ATTRIBUTION_SCOPE_VIDEO_LOCAL = "VIDEO_LOCAL"
ATTRIBUTION_SCOPES = (ATTRIBUTION_SCOPE_EVENT_LOCAL, ATTRIBUTION_SCOPE_VIDEO_LOCAL)

PASS_EVENT_LOCAL = "PASS_EVENT_LOCAL"
PASS_VIDEO_LOCAL = "PASS_VIDEO_LOCAL"
IDENTITY_GATE_NOT_REQUIRED = "NOT_REQUIRED"
IDENTITY_GATE_FAILED_MISSING = "FAILED_MISSING"


@dataclass(frozen=True)
class IdentityValidationAssertion:
    """External identity evidence; F2 validates it but never creates it."""

    video_id: str
    move_object_event_id: str
    persistent_person_id: str
    identity_level: str
    attribution_scope: str
    ready_for_attribution: bool
    validation_source: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IdentityValidationAssertion":
        if not isinstance(value, Mapping):
            raise VlmF2ValidationError("identity assertion must be an object")
        required = {
            "video_id",
            "move_object_event_id",
            "persistent_person_id",
            "identity_level",
            "attribution_scope",
            "ready_for_attribution",
            "validation_source",
        }
        unknown = set(value) - required
        missing = required - set(value)
        if unknown:
            raise VlmF2ValidationError(f"identity assertion has unknown fields: {sorted(unknown)}")
        if missing:
            raise VlmF2ValidationError(f"identity assertion is missing fields: {sorted(missing)}")
        text_fields = (
            "video_id",
            "move_object_event_id",
            "persistent_person_id",
            "identity_level",
            "attribution_scope",
            "validation_source",
        )
        for field in text_fields:
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise VlmF2ValidationError(f"identity assertion {field} must be a non-empty string")
        if value["identity_level"] not in IDENTITY_LEVELS:
            raise VlmF2ValidationError(f"identity_level is not allowed: {value['identity_level']!r}")
        if value["attribution_scope"] not in ATTRIBUTION_SCOPES:
            raise VlmF2ValidationError(
                f"attribution_scope is not allowed: {value['attribution_scope']!r}"
            )
        if not isinstance(value["ready_for_attribution"], bool):
            raise VlmF2ValidationError("identity assertion ready_for_attribution must be boolean")
        return cls(
            video_id=value["video_id"],
            move_object_event_id=value["move_object_event_id"],
            persistent_person_id=value["persistent_person_id"],
            identity_level=value["identity_level"],
            attribution_scope=value["attribution_scope"],
            ready_for_attribution=value["ready_for_attribution"],
            validation_source=value["validation_source"],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "move_object_event_id": self.move_object_event_id,
            "persistent_person_id": self.persistent_person_id,
            "identity_level": self.identity_level,
            "attribution_scope": self.attribution_scope,
            "ready_for_attribution": self.ready_for_attribution,
            "validation_source": self.validation_source,
        }
