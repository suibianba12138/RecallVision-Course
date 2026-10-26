"""Build deterministic raw-plus-candidate-reference F1 packages."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import BoundingBox, Observation

from .errors import VlmF1InputError
from .frames import VideoInfo, crop_reference_card, extract_raw_frame
from .hashing import package_sha256, sha256_file
from .schema import (
    OUTPUT_SCHEMA_VERSION,
    PACKAGE_VERSION,
    PROMPT_VERSION,
    write_contract_files,
)

FrameExtractor = Callable[[Path, int, Path], None]
ReferenceCardCropper = Callable[[Path, BoundingBox, VideoInfo, Path], dict[str, Any]]


@dataclass(frozen=True)
class BuiltEvidencePackage:
    """Filesystem result plus the canonical manifest."""

    root: Path
    manifest: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def package_hash(self) -> str:
        return package_sha256(self.root)


def select_minimal_sequence(
    context_start_frame: int,
    from_boundary_frame: int,
    to_boundary_frame: int,
    context_end_frame: int,
) -> tuple[int, ...]:
    """Resolve the nine conceptual positions with upper-half tie rounding."""

    if not (
        0 <= context_start_frame <= from_boundary_frame <= to_boundary_frame <= context_end_frame
    ):
        raise VlmF1InputError("context and MOVE_OBJECT boundary frames are not ordered")

    pre_last = max(context_start_frame, from_boundary_frame - 1)
    post_first = min(context_end_frame, to_boundary_frame + 1)

    def upper_half(start: int, end: int) -> int:
        return (start + end + 1) // 2

    conceptual = (
        context_start_frame,
        upper_half(context_start_frame, pre_last),
        pre_last,
        from_boundary_frame,
        upper_half(from_boundary_frame, to_boundary_frame),
        to_boundary_frame,
        post_first,
        upper_half(post_first, context_end_frame),
        context_end_frame,
    )
    return tuple(dict.fromkeys(conceptual))


def build_evidence_package(
    *,
    event: EventRecord,
    event_diagnostic: Mapping[str, Any],
    object_observations: Sequence[Observation],
    person_observations: Sequence[Observation],
    video_path: Path,
    output_dir: Path,
    repository_head: str,
    persistent_entity_ids: Iterable[str] | None = None,
    video_info: VideoInfo | None = None,
    frame_extractor: FrameExtractor | None = None,
    reference_card_cropper: ReferenceCardCropper | None = None,
    input_artifacts: Mapping[str, Path] | None = None,
) -> BuiltEvidencePackage:
    """Build one model-independent package for an already-supported event."""

    video_id = _validate_event(event, event_diagnostic)
    object_id = event.object_ids[0]
    if not video_path.is_file():
        raise VlmF1InputError(f"raw video is unavailable: {video_path}")
    if not repository_head:
        raise VlmF1InputError("repository_head is required for provenance")

    if persistent_entity_ids is not None and object_id not in set(persistent_entity_ids):
        raise VlmF1InputError(f"target object ID is absent from persistent entities: {object_id}")
    object_local = [
        item
        for item in object_observations
        if item.video_id == video_id and item.persistent_id == object_id and item.entity_type == "object"
    ]
    if not object_local:
        raise VlmF1InputError(f"target object has no persistent observations: {object_id}")

    info = video_info
    if info is None:
        from .frames import probe_video

        info = probe_video(video_path)
    event_frames = _event_frames(event)
    from_frame, to_frame = min(event_frames), max(event_frames)
    support_frames = _support_frames(event_diagnostic)
    if not support_frames:
        horizon = _support_horizon(event_diagnostic)
        support_frames = [
            item.frame_index
            for item in object_local
            if event.start_time - horizon <= item.timestamp <= event.end_time + horizon
        ]
    support_frames.extend(event_frames)
    if not support_frames:
        raise VlmF1InputError("MOVE_OBJECT has no valid temporal/frame support")
    context_start_frame, context_end_frame = min(support_frames), max(support_frames)
    _validate_frame_range(info, context_start_frame)
    _validate_frame_range(info, context_end_frame)
    if from_frame < context_start_frame or to_frame > context_end_frame:
        raise VlmF1InputError("MOVE_OBJECT boundary falls outside its event-local context")
    support_horizon = _support_horizon(event_diagnostic)
    selected_frames = select_minimal_sequence(
        context_start_frame,
        from_frame,
        to_frame,
        context_end_frame,
    )
    for frame_index in selected_frames:
        _validate_frame_range(info, frame_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    cards_dir = output_dir / "candidate_reference_cards"
    frames_dir.mkdir(exist_ok=True)
    cards_dir.mkdir(exist_ok=True)
    extractor = frame_extractor or extract_raw_frame
    cropper = reference_card_cropper or crop_reference_card

    raw_frame_files: dict[str, str] = {}
    raw_frame_hashes: dict[str, str] = {}
    coverage: dict[str, dict[str, bool]] = {}
    object_frames = {item.frame_index for item in object_local}
    for frame_index in selected_frames:
        destination = frames_dir / f"frame_{frame_index:04d}.jpg"
        extractor(video_path, frame_index, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise VlmF1InputError(f"frame extractor did not produce frame {frame_index}")
        key = str(frame_index)
        raw_frame_files[key] = f"frames/frame_{frame_index:04d}.jpg"
        raw_frame_hashes[key] = sha256_file(destination)
        coverage[key] = {
            "target_object_observation_available": frame_index in object_frames,
            "interpolated": False,
        }

    candidate_groups = _candidate_groups(
        person_observations,
        video_id=video_id,
        context_start_frame=context_start_frame,
        context_end_frame=context_end_frame,
    )
    candidate_labels = [f"P{index}" for index in range(1, len(candidate_groups) + 1)]
    candidate_mapping = {
        label: persistent_id for label, (persistent_id, _items) in zip(candidate_labels, candidate_groups)
    }
    candidate_facts: dict[str, dict[str, Any]] = {}
    candidate_cards: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="recallvision_vlm_f1_") as temporary:
        temporary_root = Path(temporary)
        for label, (persistent_id, items) in zip(candidate_labels, candidate_groups):
            ordered = sorted(items, key=lambda item: (item.timestamp, item.frame_index, item.track_id))
            first, last = ordered[0], ordered[-1]
            candidate_facts[label] = {
                "persistent_person_id": persistent_id,
                "first_observed_frame": first.frame_index,
                "first_observed_time": first.timestamp,
                "last_observed_frame": last.frame_index,
                "last_observed_time": last.timestamp,
                "observation_count_in_context": len(ordered),
                "present_at_from_boundary": any(item.frame_index == from_frame for item in ordered),
                "present_at_to_boundary": any(item.frame_index == to_frame for item in ordered),
                "overlaps_event_boundary": any(from_frame <= item.frame_index <= to_frame for item in ordered),
            }
            usable = [item for item in ordered if _usable_bbox(item.bbox)]
            if not usable:
                candidate_cards[label] = {
                    "status": "MISSING",
                    "persistent_person_id": persistent_id,
                    "reference_frame": None,
                    "reference_timestamp": None,
                    "source_observation_ref": None,
                    "crop_bbox": None,
                    "crop_metadata": None,
                    "card_file": None,
                    "card_sha256": None,
                    "missing_reason": "no_usable_event_local_canonical_observation",
                    "median_tie_policy": "lower_observation_on_even_count",
                }
                continue
            reference = usable[(len(usable) - 1) // 2]
            card_path = cards_dir / f"{label}.jpg"
            source_frame = temporary_root / f"reference_{reference.frame_index:04d}.jpg"
            extractor(video_path, reference.frame_index, source_frame)
            if not source_frame.is_file() or source_frame.stat().st_size == 0:
                raise VlmF1InputError(f"reference source frame {reference.frame_index} was not produced")
            crop_metadata = cropper(source_frame, reference.bbox, info, card_path)
            if not card_path.is_file() or card_path.stat().st_size == 0:
                raise VlmF1InputError(f"reference card was not produced for {label}")
            candidate_cards[label] = {
                "status": "AVAILABLE",
                "persistent_person_id": persistent_id,
                "reference_frame": reference.frame_index,
                "reference_timestamp": reference.timestamp,
                "source_observation_ref": _observation_ref(reference),
                "crop_bbox": _bbox_record(reference.bbox),
                "crop_metadata": crop_metadata,
                "card_file": f"candidate_reference_cards/{label}.jpg",
                "card_sha256": sha256_file(card_path),
                "median_tie_policy": "lower_observation_on_even_count",
            }

    source_artifacts = {
        "video": {"path": video_path.name, "sha256": sha256_file(video_path)}
    }
    for name, path in sorted((input_artifacts or {}).items()):
        source_artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    manifest: dict[str, Any] = {
        "package_version": PACKAGE_VERSION,
        "video_id": video_id,
        "video_hash": sha256_file(video_path),
        "video_metadata": {
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
        },
        "move_object_event_id": event.event_id,
        "target_object_id": object_id,
        "event_from_frame": from_frame,
        "event_to_frame": to_frame,
        "event_start_time": event.start_time,
        "event_end_time": event.end_time,
        "support_horizon_seconds": support_horizon,
        "context_start_time": context_start_frame / info.fps,
        "context_end_time": context_end_frame / info.fps,
        "context_start_frame": context_start_frame,
        "context_end_frame": context_end_frame,
        "frame_selection_policy": "MINIMAL_SEQUENCE_V1",
        "frame_rounding_policy": "upper_integer_on_half_tie",
        "selected_frame_indices": list(selected_frames),
        "raw_frame_files": raw_frame_files,
        "raw_frame_hashes": raw_frame_hashes,
        "detector_coverage": coverage,
        "candidate_labels": candidate_labels,
        "candidate_mapping": candidate_mapping,
        "candidate_temporal_facts": candidate_facts,
        "candidate_reference_cards": candidate_cards,
        "identity_validation_metadata": {
            "status": "UNKNOWN",
            "scope": None,
            "source": "not_supplied_to_f1_core",
        },
        "prompt_version": PROMPT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "repository_head": repository_head,
        "source_artifacts": source_artifacts,
        "creation_configuration": {
            "raw_event_frames": True,
            "structured_overlay_used": False,
            "short_clip_used": False,
            "full_video_vlm_used": False,
            "vlm_inference": False,
            "provider_dependency": False,
            "person_detection_rerun": False,
            "bbox_interpolation": False,
            "reference_bbox_policy": "canonical_xyxy_floor_ceil_clip_no_expansion",
        },
    }
    write_contract_files(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BuiltEvidencePackage(root=output_dir, manifest=manifest)


def _validate_event(event: EventRecord, diagnostic: Mapping[str, Any]) -> str:
    if event.event_type is not EventType.MOVE_OBJECT:
        raise VlmF1InputError("VLM-F1 requires an existing MOVE_OBJECT event")
    if not event.event_id or not event.object_ids or len(event.object_ids) != 1:
        raise VlmF1InputError("MOVE_OBJECT must contain exactly one target object ID")
    if not math.isfinite(event.start_time) or not math.isfinite(event.end_time) or event.end_time < event.start_time:
        raise VlmF1InputError("MOVE_OBJECT has invalid temporal bounds")
    diagnostic_event_id = diagnostic.get("event_id")
    if diagnostic_event_id is not None and diagnostic_event_id != event.event_id:
        raise VlmF1InputError("event diagnostic does not match MOVE_OBJECT event")
    video_ids = {item.video_id for item in event.evidence}
    if not video_ids or len(video_ids) != 1:
        raise VlmF1InputError("MOVE_OBJECT evidence must identify exactly one video")
    if any(frame < 0 for frame in _event_frames(event)):
        raise VlmF1InputError("MOVE_OBJECT evidence contains a negative frame")
    return next(iter(video_ids))


def _event_frames(event: EventRecord) -> list[int]:
    frames = sorted({frame for item in event.evidence for frame in item.frame_indices})
    if not frames:
        raise VlmF1InputError("MOVE_OBJECT has no frame evidence")
    return frames


def _support_frames(diagnostic: Mapping[str, Any]) -> list[int]:
    frames: list[int] = []
    for field in ("left_support_frames", "right_support_frames"):
        value = diagnostic.get(field, [])
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise VlmF1InputError(f"{field} must be a list of integer frames")
        frames.extend(value)
    return sorted(set(frames))


def _support_horizon(diagnostic: Mapping[str, Any]) -> float:
    value = diagnostic.get("support_horizon_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise VlmF1InputError("support_horizon_seconds must be non-negative and finite")
    return float(value)


def _validate_frame_range(info: VideoInfo, frame_index: int) -> None:
    if frame_index < 0 or (info.frame_count is not None and frame_index >= info.frame_count):
        raise VlmF1InputError(f"frame {frame_index} is outside the raw video")


def _candidate_groups(
    observations: Sequence[Observation],
    *,
    video_id: str,
    context_start_frame: int,
    context_end_frame: int,
) -> list[tuple[str, list[Observation]]]:
    groups: dict[str, list[Observation]] = {}
    for item in observations:
        if (
            item.video_id == video_id
            and item.entity_type == "person"
            and item.persistent_id
            and context_start_frame <= item.frame_index <= context_end_frame
        ):
            groups.setdefault(item.persistent_id, []).append(item)
    ordered = sorted(
        groups.items(),
        key=lambda pair: (
            min(item.timestamp for item in pair[1]),
            pair[0],
        ),
    )
    if not ordered:
        raise VlmF1InputError("candidate-person memory has no observation in the event-local context")
    return [(persistent_id, sorted(items, key=lambda item: (item.timestamp, item.frame_index, item.track_id))) for persistent_id, items in ordered]


def _observation_ref(observation: Observation) -> str:
    return f"{observation.video_id}#frame={observation.frame_index}#{observation.persistent_id}"


def _bbox_record(bbox: BoundingBox) -> list[float]:
    return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]


def _usable_bbox(bbox: BoundingBox) -> bool:
    values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return all(math.isfinite(value) for value in values) and bbox.x2 > bbox.x1 and bbox.y2 > bbox.y1
