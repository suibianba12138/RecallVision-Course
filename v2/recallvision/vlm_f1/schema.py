"""Versioned, provider-independent F1 prompt and output contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_VERSION = "VLM-F1-PACKAGE-1.0"
PROMPT_VERSION = "VLM-F1-PROMPT-2026-10-04-v1"
OUTPUT_SCHEMA_VERSION = "VLM-F1-OUTPUT-SCHEMA-2026-10-04-v1"

ALLOWED_OUTPUT_STATUSES = (
    "SUPPORTED_AGENT",
    "AMBIGUOUS_AGENT",
    "NO_SUPPORTED_AGENT",
    "INSUFFICIENT_VISUAL_EVIDENCE",
)

PROMPT_TEXT = """A structured visual-memory system has already identified a supported MOVE_OBJECT event for a target object. The model is not asked to decide whether the object moved, when it moved, whether it exists, or which persistent entities exist.

Using only the supplied event-local raw images, the closed supplied candidate set, and any supplied structured temporal facts, determine whether one supplied candidate is visually supported as causing the target object's displacement.

Presence, proximity, touching, holding, or a stationary grasp alone are insufficient. Choose a candidate only when the sequence supports candidate-caused displacement. Abstain when the evidence is ambiguous, insufficient, or unsupported.

Return only a JSON object matching the supplied output schema. candidate_label must be one supplied label or null. Do not invent labels or persistent IDs. The downstream RecallVision package maps a selected candidate label to a persistent entity and applies a separate identity-validation gate.

Target object: {target_object_id}
Supplied candidate labels: {candidate_labels}
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": OUTPUT_SCHEMA_VERSION,
    "title": "RecallVision VLM-F1 output",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "candidate_label",
        "interaction_supported",
        "causal_support",
        "evidence_frames",
        "brief_reason",
    ],
    "properties": {
        "status": {"type": "string", "enum": list(ALLOWED_OUTPUT_STATUSES)},
        "candidate_label": {"type": ["string", "null"], "pattern": "^P[1-9][0-9]*$"},
        "interaction_supported": {"type": "boolean"},
        "causal_support": {"type": "boolean"},
        "evidence_frames": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "brief_reason": {"type": "string", "minLength": 1},
    },
}


def write_contract_files(package_root: Path) -> None:
    """Write versioned generic prompt and strict output schema files."""

    (package_root / "prompt.txt").write_text(PROMPT_TEXT, encoding="utf-8")
    (package_root / "output_schema.json").write_text(
        json.dumps(OUTPUT_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
