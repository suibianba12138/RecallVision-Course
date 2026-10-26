"""Fail-closed errors for VLM-F1 package construction and validation."""


class VlmF1InputError(ValueError):
    """Raised when structured inputs cannot safely produce an F1 package."""


class VlmOutputValidationError(ValueError):
    """Raised when a future VLM result violates the closed output contract."""
