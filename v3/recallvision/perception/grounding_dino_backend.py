"""Offline, target-conditioned Grounding DINO adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from pathlib import Path
from typing import Any

from .target_conditioned import (
    ModelProvenance,
    TargetDetection,
    make_target_detections,
    normalize_target_prompt,
)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class GroundingDinoTargetDetector:
    """Local-only Transformers implementation of TargetConditionedDetector.

    The checkpoint is hash-pinned and every `from_pretrained` call is forced
    offline, so runtime inference cannot silently fetch a different model.
    """

    repository = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        *,
        asset_root: Path,
        revision: str,
        expected_checkpoint_sha256: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        asset_root = Path(asset_root)
        checkpoint = asset_root / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not _REVISION_RE.fullmatch(revision):
            raise ValueError("revision must be a pinned 40-character commit SHA")
        if not _SHA256_RE.fullmatch(expected_checkpoint_sha256):
            raise ValueError("expected_checkpoint_sha256 must be a 64-character SHA-256 digest")
        if not math.isfinite(box_threshold) or not 0.0 <= box_threshold <= 1.0:
            raise ValueError("box_threshold must be between 0 and 1")
        if not math.isfinite(text_threshold) or not 0.0 <= text_threshold <= 1.0:
            raise ValueError("text_threshold must be between 0 and 1")
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256.lower() != expected_checkpoint_sha256.lower():
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected {expected_checkpoint_sha256}, found {actual_sha256}"
            )

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.asset_root = asset_root
        self.revision = revision
        self.checkpoint_sha256 = actual_sha256
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(asset_root), local_files_only=True
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(asset_root), local_files_only=True
        ).to(device)
        self.model.eval()
        self.provenance = ModelProvenance(
            repository=self.repository,
            revision=revision,
            checkpoint_sha256=actual_sha256,
            runtime_name="transformers",
            runtime_version=importlib.metadata.version("transformers"),
        )

    def detect(
        self,
        image: Any,
        *,
        video_id: str,
        video_sha256: str,
        target_description: str,
        frame_index: int,
        timestamp_seconds: float,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[TargetDetection]:
        effective_box_threshold = self.box_threshold if box_threshold is None else float(box_threshold)
        effective_text_threshold = self.text_threshold if text_threshold is None else float(text_threshold)
        if not math.isfinite(effective_box_threshold) or not 0.0 <= effective_box_threshold <= 1.0:
            raise ValueError("box_threshold must be between 0 and 1")
        if not math.isfinite(effective_text_threshold) or not 0.0 <= effective_text_threshold <= 1.0:
            raise ValueError("text_threshold must be between 0 and 1")
        caption = normalize_target_prompt(target_description)
        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if self._torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=effective_box_threshold,
            text_threshold=effective_text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = _to_list(processed.get("boxes", []))
        scores = _to_list(processed.get("scores", []))
        if not boxes:
            return []
        labels = processed.get("text_labels") or processed.get("labels") or []
        labels = [str(label) for label in labels]
        if not labels and boxes:
            labels = [target_description] * len(boxes)
        return make_target_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            video_id=video_id,
            video_sha256=video_sha256,
            target_description=target_description,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            model=self.provenance,
        )
