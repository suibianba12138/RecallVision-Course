"""Backend-independent VLM-F1 event-local evidence packaging."""

from .builder import BuiltEvidencePackage, VideoInfo, build_evidence_package
from .validator import validate_vlm_output

__all__ = [
    "BuiltEvidencePackage",
    "VideoInfo",
    "build_evidence_package",
    "validate_vlm_output",
]
