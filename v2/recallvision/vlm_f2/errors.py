"""Fail-closed errors for VLM-F2."""

from __future__ import annotations


class VlmF2ValidationError(ValueError):
    """Raised when an external interpretation cannot be validated."""
