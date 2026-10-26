"""Provider-independent VLM-F3 attribution authorization."""

from .authorization import authorize_attribution, write_artifacts

__all__ = ["authorize_attribution", "write_artifacts"]
