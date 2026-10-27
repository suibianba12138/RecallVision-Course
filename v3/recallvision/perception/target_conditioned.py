"""Model-neutral records for prompt-conditioned video detections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ModelProvenance:
    repository: str
    revision: str
    checkpoint_sha256: str
    runtime_name: str
    runtime_version: str

    def __post_init__(self) -> None:
        for name in ("repository", "revision", "runtime_name", "runtime_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a 64-character SHA-256 digest")


@dataclass(frozen=True)
class TargetDetection:
    video_id: str
    video_sha256: str
    target_description: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    frame_index: int
    timestamp_seconds: float
    model: ModelProvenance
    raw_model_label: str

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not _SHA256_RE.fullmatch(self.video_sha256):
            raise ValueError("video_sha256 must be a 64-character SHA-256 digest")
        if not self.target_description.strip():
            raise ValueError("target_description must not be empty")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coordinates")
        if any(not math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("bbox_xyxy coordinates must be finite")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive area")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0.0:
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        if not self.raw_model_label.strip():
            raise ValueError("raw_model_label must not be empty")

    def as_record(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_sha256": self.video_sha256,
            "target_description": self.target_description,
            "bbox_xyxy": list(self.bbox_xyxy),
            "score": self.score,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "raw_model_label": self.raw_model_label,
            "model_repository": self.model.repository,
            "model_revision": self.model.revision,
            "checkpoint_sha256": self.model.checkpoint_sha256,
            "runtime_name": self.model.runtime_name,
            "runtime_version": self.model.runtime_version,
        }


class TargetConditionedDetector(Protocol):
    """Common interface for detectors conditioned on a user target description."""

    def detect(
        self,
        image: Any,
        *,
        video_id: str,
        video_sha256: str,
        target_description: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[TargetDetection]: ...


def normalize_target_prompt(description: str) -> str:
    """Normalize a target phrase without adding semantic words."""
    normalized = " ".join(description.strip().lower().rstrip(".").split())
    if not normalized:
        raise ValueError("target_description must not be empty")
    return normalized + "."


def make_target_detections(
    *,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    video_id: str,
    video_sha256: str,
    target_description: str,
    frame_index: int,
    timestamp_seconds: float,
    model: ModelProvenance,
) -> list[TargetDetection]:
    """Convert backend box/score/label arrays to stable, validated records."""
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("boxes, scores, and labels must have the same length")
    return [
        TargetDetection(
            video_id=video_id,
            video_sha256=video_sha256,
            target_description=target_description,
            bbox_xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
            score=float(score),
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            model=model,
            raw_model_label=str(label),
        )
        for box, score, label in zip(boxes, scores, labels)
    ]
