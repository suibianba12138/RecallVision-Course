"""Strict validation for future provider outputs; no semantic attribution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .errors import VlmOutputValidationError
from .schema import ALLOWED_OUTPUT_STATUSES

_REQUIRED_FIELDS = {
    "status",
    "candidate_label",
    "interaction_supported",
    "causal_support",
    "evidence_frames",
    "brief_reason",
}
_ABSTENTION_STATUSES = {
    "AMBIGUOUS_AGENT",
    "NO_SUPPORTED_AGENT",
    "INSUFFICIENT_VISUAL_EVIDENCE",
}


def validate_vlm_output(
    result: Mapping[str, Any],
    *,
    candidate_labels: Sequence[str],
    supplied_frames: Iterable[int],
) -> dict[str, Any]:
    """Validate a closed VLM result without mapping it to a persistent ID."""

    if not isinstance(result, Mapping):
        raise VlmOutputValidationError("VLM output must be an object")
    unknown = set(result) - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - set(result)
    if unknown:
        raise VlmOutputValidationError(f"unknown fields: {sorted(unknown)}")
    if missing:
        raise VlmOutputValidationError(f"missing fields: {sorted(missing)}")

    status = result["status"]
    if status not in ALLOWED_OUTPUT_STATUSES:
        raise VlmOutputValidationError(f"status is not allowed: {status!r}")
    candidate = result["candidate_label"]
    allowed_candidates = set(candidate_labels)
    if candidate is not None and candidate not in allowed_candidates:
        raise VlmOutputValidationError(
            f"candidate_label {candidate!r} is not in the closed candidate set"
        )
    if status == "SUPPORTED_AGENT" and candidate is None:
        raise VlmOutputValidationError("SUPPORTED_AGENT requires a non-null candidate_label")
    if status in _ABSTENTION_STATUSES and candidate is not None:
        raise VlmOutputValidationError("abstention candidate_label must be null")

    if not isinstance(result["interaction_supported"], bool):
        raise VlmOutputValidationError("interaction_supported must be boolean")
    if not isinstance(result["causal_support"], bool):
        raise VlmOutputValidationError("causal_support must be boolean")
    if not isinstance(result["evidence_frames"], list) or any(
        isinstance(frame, bool) or not isinstance(frame, int) for frame in result["evidence_frames"]
    ):
        raise VlmOutputValidationError("evidence_frames must be a list of integers")
    allowed_frames = set(supplied_frames)
    if any(frame not in allowed_frames for frame in result["evidence_frames"]):
        raise VlmOutputValidationError("evidence_frames contains a frame absent from the package")
    if not isinstance(result["brief_reason"], str) or not result["brief_reason"].strip():
        raise VlmOutputValidationError("brief_reason must be a non-empty string")

    return dict(result)
