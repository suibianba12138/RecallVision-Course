"""Stable constants for the dependency-free EVAL-D0 contracts."""

from __future__ import annotations

import re

MANIFEST_SCHEMA_VERSION = "eval_d0_manifest_v1"
ANNOTATION_SCHEMA_VERSION = "evaluation_annotation_v1"

QUERY_TYPES = (
    "WHERE_OBJECT",
    "WHEN_OBJECT_MOVED",
    "WHERE_OBJECT_BEFORE",
    "WHO_MOVED_OBJECT",
)
QUERY_TYPE_SET = frozenset(QUERY_TYPES)

SPLITS = frozenset({"DEV", "EVAL"})
ANSWERABILITY_STATUSES = frozenset({"ANSWERABLE", "UNANSWERABLE", "AMBIGUOUS"})
EXPECTED_QUERY_STATUSES = frozenset({"ANSWER", "ABSTAIN"})
TARGET_REFERENCE_STATUSES = frozenset({"KNOWN_OBJECT", "UNKNOWN_OBJECT"})
AGENT_ATTRIBUTION_STATUSES = frozenset(
    {"SUPPORTED", "AMBIGUOUS", "NO_VISIBLE_AGENT", "INSUFFICIENT"}
)
MOVEMENT_SUPPORT_STATUSES = frozenset(
    {"SUPPORTED", "NO_SUPPORTED_EVENT", "INSUFFICIENT"}
)

HARD_NEGATIVE_TAGS = frozenset(
    {
        "SAME_ZONE_MOVEMENT",
        "STATIONARY_INTERACTION",
        "MULTIPLE_PEOPLE",
        "POST_EVENT_CANDIDATE",
        "DETECTOR_GAP_EXPECTED",
        "OCCLUSION",
        "TRACK_FRAGMENTATION_EXPECTED",
        "MULTIPLE_MOVE_EVENTS",
        "OBJECT_DISAPPEARS",
        "NO_VISIBLE_AGENT",
        "AMBIGUOUS_AGENT",
    }
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def is_sha256(value: object) -> bool:
    """Return whether *value* is a 64-character hexadecimal SHA-256 digest."""

    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
