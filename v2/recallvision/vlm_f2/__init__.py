"""Provider-independent VLM-F2 interpretation ingestion."""

from .ingest import ingest_interpretation, write_sidecar

__all__ = ["ingest_interpretation", "write_sidecar"]
