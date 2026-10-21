"""Run P2A-R1 duplicate-aware entity consolidation over frozen observations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recallvision.memory.appearance import HSVHistogramConfig, extract_hsv_histogram
from recallvision.memory.consolidation import (
    EntityBuild,
    EntityConsolidationConfig,
    EntityFragment,
    build_entity_fragments,
    build_persistent_entities,
    detect_spatial_conflicts,
    find_continuation_candidates,
    load_entity_consolidation_config,
    select_continuation_edges,
)
from recallvision.memory.persistent_objects import enrich_observations
from recallvision.memory.relations import (
    CanonicalObservation,
    DuplicateCandidate,
    DuplicateGroup,
    build_duplicate_groups,
    canonicalize_observations,
    find_duplicate_candidates,
)
from recallvision.memory.store import ObservationMemoryStore
from recallvision.memory.tracklets import TrackletKey, TrackletSummary, build_tracklet_summaries
from recallvision.perception.ultralytics_backend import read_video_metadata, sha256_file
from recallvision.query.object_location import answer_where_object
from recallvision.query.parser import parse_query


@dataclass(frozen=True)
class R1RunResult:
    output_dir: Path
    summary: dict[str, Any]


def run_p2a_r1(
    *,
    input_memory: Path,
    input_video: Path,
    output_dir: Path,
    config_path: Path,
) -> R1RunResult:
    """Run R1 without loading or executing a detector/tracker."""

    observations = ObservationMemoryStore(input_memory).load_all()
    if not observations:
        raise ValueError(f"R1 input memory is empty: {input_memory}")
    metadata = read_video_metadata(input_video)
    config = load_entity_consolidation_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = build_tracklet_summaries(
        observations,
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
    )
    descriptors = _extract_tracklet_descriptors(input_video, summaries, config)
    duplicate_candidates = find_duplicate_candidates(
        summaries,
        observations,
        descriptors,
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
        config=config,
    )
    duplicate_groups = build_duplicate_groups(summaries, duplicate_candidates)
    canonical = canonicalize_observations(observations, duplicate_groups)
    canonical_appearances = _extract_canonical_appearances(
        input_video,
        canonical,
        config,
    )
    fragments = build_entity_fragments(
        summaries,
        duplicate_groups,
        observations,
        canonical_appearances,
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
        window_size=config.boundary_window_size,
    )
    conflicts = detect_spatial_conflicts(
        fragments,
        config=config,
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
    )
    continuation_candidates = find_continuation_candidates(fragments, config=config)
    continuation_selection = select_continuation_edges(
        continuation_candidates,
        conflicts=conflicts,
        config=config,
    )
    entity_build = build_persistent_entities(
        fragments,
        observations,
        canonical,
        continuation_selection.accepted_edges,
        duplicate_candidates,
    )
    enriched = enrich_observations(observations, entity_build.track_to_persistent)

    _write_json(output_dir / "baseline_failure_audit.json", _baseline_audit())
    _write_json(output_dir / "duplicate_candidates.json", [item.to_record() for item in duplicate_candidates])
    _write_json(output_dir / "duplicate_edges.json", [item.to_record() for item in duplicate_candidates])
    _write_json(output_dir / "duplicate_groups.json", [item.to_record() for item in duplicate_groups])
    _write_json(output_dir / "continuation_candidates.json", continuation_selection.to_records())
    _write_json(output_dir / "continuation_edges.json", continuation_selection.to_records())
    _write_json(output_dir / "entity_conflicts.json", [item.to_record() for item in conflicts])
    _write_json(output_dir / "persistent_entities.json", [item.to_record() for item in entity_build.entities])
    _write_json(output_dir / "track_to_persistent_mapping.json", {
        "|".join(key): value for key, value in sorted(entity_build.track_to_persistent.items())
    })
    _write_json(output_dir / "tracklet_summaries.json", [item.to_record() for item in summaries])

    raw_path = output_dir / "observations_with_persistent_ids.jsonl"
    _write_observations(raw_path, enriched)
    consolidated_path = output_dir / "observations_consolidated.jsonl"
    _write_consolidated(consolidated_path, canonical, entity_build.fragment_to_persistent)

    _write_r1_contact_sheet(input_video, summaries, duplicate_groups, output_dir / "bottle_tracklet_contact_sheet_r1.jpg")
    _write_fullframe_audit(input_video, observations, output_dir / "bottle_fullframe_identity_audit.jpg")
    summary = _build_summary(
        observations=observations,
        enriched=enriched,
        canonical=canonical,
        summaries=summaries,
        duplicate_candidates=duplicate_candidates,
        duplicate_groups=duplicate_groups,
        fragments=fragments,
        continuation_selection=continuation_selection,
        conflicts=conflicts,
        entity_build=entity_build,
        metadata=metadata,
        input_memory=input_memory,
        input_video=input_video,
        config_path=config_path,
        config=config,
        consolidated_path=consolidated_path,
    )
    _write_json(output_dir / "entity_consolidation_summary.json", summary)
    return R1RunResult(output_dir=output_dir, summary=summary)


def _extract_tracklet_descriptors(
    video_path: Path,
    summaries: list[TrackletSummary],
    config: EntityConsolidationConfig,
) -> dict[TrackletKey, tuple[float, ...]]:
    import cv2

    histogram_config = HSVHistogramConfig(config.appearance_h_bins, config.appearance_s_bins)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cache: dict[int, Any] = {}
    try:
        descriptors: dict[TrackletKey, tuple[float, ...]] = {}
        for summary in summaries:
            representative = summary.representative_frames[1]
            frame = _read_frame(capture, representative.frame_index, video_path, cache)
            box = representative.bbox
            descriptors[summary.key] = extract_hsv_histogram(
                frame,
                (box.x1, box.y1, box.x2, box.y2),
                config=histogram_config,
            )
        return descriptors
    finally:
        capture.release()


def _extract_canonical_appearances(
    video_path: Path,
    canonical: list[CanonicalObservation],
    config: EntityConsolidationConfig,
) -> dict[tuple[str, int], tuple[float, ...]]:
    import cv2

    histogram_config = HSVHistogramConfig(config.appearance_h_bins, config.appearance_s_bins)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cache: dict[int, Any] = {}
    try:
        appearances: dict[tuple[str, int], tuple[float, ...]] = {}
        for item in canonical:
            observation = item.observation
            frame = _read_frame(capture, observation.frame_index, video_path, cache)
            box = observation.bbox
            appearances[(item.group_id, observation.frame_index)] = extract_hsv_histogram(
                frame,
                (box.x1, box.y1, box.x2, box.y2),
                config=histogram_config,
            )
        return appearances
    finally:
        capture.release()


def _read_frame(capture: Any, frame_index: int, video_path: Path, cache: dict[int, Any]) -> Any:
    import cv2

    if frame_index not in cache:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        cache[frame_index] = frame
    return cache[frame_index]


def _write_observations(path: Path, observations: list[Any]) -> None:
    path.unlink(missing_ok=True)
    store = ObservationMemoryStore(path)
    for observation in observations:
        store.append(observation)


def _write_consolidated(
    path: Path,
    canonical: list[CanonicalObservation],
    fragment_to_persistent: dict[str, str],
) -> None:
    lines = []
    for item in canonical:
        fragment_id = f"fragment:{item.group_id}"
        if fragment_id not in fragment_to_persistent:
            raise KeyError(f"missing persistent entity for {fragment_id}")
        lines.append(
            json.dumps(
                item.to_record(persistent_id=fragment_to_persistent[fragment_id]),
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(
    *,
    observations: list[Any],
    enriched: list[Any],
    canonical: list[CanonicalObservation],
    summaries: list[TrackletSummary],
    duplicate_candidates: list[DuplicateCandidate],
    duplicate_groups: list[DuplicateGroup],
    fragments: list[EntityFragment],
    continuation_selection: Any,
    conflicts: list[Any],
    entity_build: EntityBuild,
    metadata: dict[str, Any],
    input_memory: Path,
    input_video: Path,
    config_path: Path,
    config: EntityConsolidationConfig,
    consolidated_path: Path,
) -> dict[str, Any]:
    bottle_observations = [item for item in observations if item.class_name == "bottle"]
    bottle_summaries = [item for item in summaries if item.class_name == "bottle"]
    bottle_groups = [item for item in duplicate_groups if item.class_name == "bottle"]
    bottle_fragments = [item for item in fragments if item.class_name == "bottle"]
    bottle_entities = [item for item in entity_build.entities if item.class_name == "bottle"]
    bottle_continuations = [
        item for item in continuation_selection.accepted_edges if item.class_name == "bottle"
    ]
    query = parse_query("Where is the bottle?", video_id=bottle_observations[0].video_id, question_id="p2a-r1-query")
    consolidated_observations = ObservationMemoryStore(consolidated_path).load_all()
    answer = answer_where_object(query, consolidated_observations)
    selected = next(
        (
            item for item in consolidated_observations
            if item.frame_index == answer.frame_index
            and item.class_name == "bottle"
            and item.persistent_id == answer.target_entity_id
        ),
        None,
    )
    duplicate_16_17 = next(
        (item for item in duplicate_candidates if {item.source_track_id, item.target_track_id} == {"16", "17"}),
        None,
    )
    continuation_1_58 = next(
        (
            item for item in continuation_selection.all_edges
            if _edge_contains_tracks(item, fragments, "1", "58")
        ),
        None,
    )
    return {
        "p2a_r1_pipeline_status": "PASS",
        "p2a_r1_identity_validation_status": "PENDING_HUMAN_REVIEW",
        "input_memory": str(input_memory),
        "input_video": str(input_video),
        "video_sha256": sha256_file(input_video),
        "video_metadata": metadata,
        "config_file": str(config_path),
        "config": config.to_record(),
        "raw_bottle_observations": len(bottle_observations),
        "raw_bottle_tracklets": len(bottle_summaries),
        "raw_track_ids": [item.track_id for item in bottle_summaries],
        "p2a_baseline_persistent_objects": 4,
        "p2a_r1_duplicate_group_count": len(bottle_groups),
        "p2a_r1_fragment_count": len(bottle_fragments),
        "p2a_r1_continuation_edge_count": len(bottle_continuations),
        "p2a_r1_persistent_objects": len(bottle_entities),
        "duplicate_groups": [item.to_record() for item in bottle_groups],
        "persistent_entities": [item.to_record() for item in bottle_entities],
        "track_to_persistent_mapping": {
            "|".join(key): value for key, value in sorted(entity_build.track_to_persistent.items())
            if key[1] == "bottle"
        },
        "relation_16_17": {
            "relation": duplicate_16_17.relation_type.value if duplicate_16_17 else "NOT_FOUND",
            "shared_frames": duplicate_16_17.shared_frame_count if duplicate_16_17 else None,
            "median_iou": duplicate_16_17.median_iou if duplicate_16_17 else None,
            "median_ios": duplicate_16_17.median_ios if duplicate_16_17 else None,
            "center_distance": duplicate_16_17.median_center_distance if duplicate_16_17 else None,
            "appearance": duplicate_16_17.appearance_similarity if duplicate_16_17 else None,
            "decision": duplicate_16_17.accepted if duplicate_16_17 else False,
        },
        "relation_1_58": {
            "relation": continuation_1_58.relation_type.value if continuation_1_58 else "NOT_FOUND",
            "temporal_gap": continuation_1_58.temporal_gap if continuation_1_58 else None,
            "center_distance": continuation_1_58.center_distance if continuation_1_58 else None,
            "area_ratio": continuation_1_58.area_ratio if continuation_1_58 else None,
            "appearance": continuation_1_58.appearance_similarity if continuation_1_58 else None,
            "continuation_score": continuation_1_58.score if continuation_1_58 else None,
            "decision": continuation_1_58.accepted if continuation_1_58 else False,
            "rejection_reason": continuation_1_58.rejection_reason if continuation_1_58 else None,
        },
        "raw_enriched_memory": "observations_with_persistent_ids.jsonl",
        "consolidated_memory": "observations_consolidated.jsonl",
        "raw_observation_count": len(enriched),
        "consolidated_observation_count": len(canonical),
        "query": {
            "question": "Where is the bottle?",
            "answer": answer.answer,
            "zone": answer.zone,
            "timestamp": answer.timestamp,
            "frame_index": answer.frame_index,
            "track_id": selected.track_id if selected else None,
            "persistent_id": selected.persistent_id if selected else None,
        },
        "fullframe_audit_image": "bottle_fullframe_identity_audit.jpg",
        "r1_contact_sheet": "bottle_tracklet_contact_sheet_r1.jpg",
        "original_observation_preservation": _preserves_original_fields(observations, enriched),
        "human_review_required": True,
        "training_run": False,
        "deep_reid_used": False,
        "vlm_used": False,
    }


def _edge_contains_tracks(edge: Any, fragments: list[EntityFragment], left: str, right: str) -> bool:
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    source = by_id.get(edge.source_fragment_id)
    target = by_id.get(edge.target_fragment_id)
    return bool(source and target and left in source.boundary.track_ids and right in target.boundary.track_ids)


def _preserves_original_fields(original: list[Any], enriched: list[Any]) -> bool:
    if len(original) != len(enriched):
        return False
    return all(
        left.video_id == right.video_id
        and left.frame_index == right.frame_index
        and left.timestamp == right.timestamp
        and left.track_id == right.track_id
        and left.entity_type == right.entity_type
        and left.class_name == right.class_name
        and left.bbox == right.bbox
        and left.confidence == right.confidence
        and left.zone == right.zone
        for left, right in zip(original, enriched)
    )


def _baseline_audit() -> dict[str, Any]:
    canonical = Path("outputs/p2a_r1/pixabay_bottle/baseline_failure_audit.json")
    if canonical.exists():
        return json.loads(canonical.read_text(encoding="utf-8"))
    raise FileNotFoundError("baseline_failure_audit.json must be generated before R1 formal run")


def _write_r1_contact_sheet(
    video_path: Path,
    summaries: list[TrackletSummary],
    groups: list[DuplicateGroup],
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    bottle_summaries = [summary for summary in summaries if summary.class_name == "bottle"]
    group_by_key = {key: group for group in groups for key in group.tracklet_keys}
    cell_width, cell_height = 380, 310
    canvas = np.full((len(bottle_summaries) * cell_height, 3 * cell_width, 3), 245, dtype=np.uint8)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cache: dict[int, Any] = {}
    try:
        for row, summary in enumerate(bottle_summaries):
            group = group_by_key[summary.key]
            for column, representative in enumerate(summary.representative_frames):
                frame = _read_frame(capture, representative.frame_index, video_path, cache)
                crop = _crop(frame, representative.bbox)
                if crop is None:
                    raise RuntimeError(f"Could not crop track {summary.track_id} frame {representative.frame_index}")
                cell = _fit_crop(crop, cell_width, cell_height)
                y0, x0 = row * cell_height, column * cell_width
                canvas[y0:y0 + cell_height, x0:x0 + cell_width] = cell
                cv2.putText(canvas, f"track={summary.track_id} group={group.group_id}", (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"frame={representative.frame_index} t={representative.timestamp:.3f}s", (x0 + 8, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    finally:
        capture.release()
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write {output_path}")


def _write_fullframe_audit(video_path: Path, observations: list[Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    frames = [69, 70, 74, 75, 78, 79, 104, 111, 113, 126, 151, 152]
    metadata = read_video_metadata(video_path)
    by_frame = {frame: [item for item in observations if item.class_name == "bottle" and item.frame_index == frame] for frame in frames}
    cell_width, cell_height = 640, 420
    canvas = np.full((4 * cell_height, 3 * cell_width, 3), 245, dtype=np.uint8)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        for index, frame_index in enumerate(frames):
            frame = _read_frame(capture, frame_index, video_path, {})
            display = cv2.resize(frame, (cell_width, 360), interpolation=cv2.INTER_AREA)
            sx, sy = cell_width / metadata["width"], 360 / metadata["height"]
            detections = by_frame[frame_index]
            for item in detections:
                x1, y1, x2, y2 = [int(round(value)) for value in (item.bbox.x1 * sx, item.bbox.y1 * sy, item.bbox.x2 * sx, item.bbox.y2 * sy)]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 130, 255), 2)
                cv2.putText(display, f"track={item.track_id} conf={item.confidence:.2f}", (max(2, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 130, 255), 1, cv2.LINE_AA)
            cell = np.full((cell_height, cell_width, 3), 245, dtype=np.uint8)
            cell[:360] = display
            cv2.putText(cell, f"frame={frame_index} t={frame_index / metadata['fps']:.3f}s", (8, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(cell, f"bottle_detections={len(detections)}" if detections else "NO_BOTTLE_OBSERVATION", (8, 407), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 100, 0) if detections else (0, 0, 180), 1, cv2.LINE_AA)
            row, column = divmod(index, 3)
            canvas[row * cell_height:(row + 1) * cell_height, column * cell_width:(column + 1) * cell_width] = cell
    finally:
        capture.release()
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write {output_path}")


def _crop(frame: Any, bbox: Any) -> Any:
    height, width = frame.shape[:2]
    left, top = max(0, min(width, int(bbox.x1))), max(0, min(height, int(bbox.y1)))
    right, bottom = max(0, min(width, int(bbox.x2 + 0.999999))), max(0, min(height, int(bbox.y2 + 0.999999)))
    return None if right <= left or bottom <= top else frame[top:bottom, left:right]


def _fit_crop(crop: Any, width: int, height: int) -> Any:
    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    crop_height, crop_width = crop.shape[:2]
    scale = min((width - 16) / crop_width, (height - 58) / crop_height)
    resized = cv2.resize(crop, (max(1, int(round(crop_width * scale))), max(1, int(round(crop_height * scale)))), interpolation=cv2.INTER_AREA)
    y = 52 + max(0, (height - 52 - resized.shape[0]) // 2)
    x = max(0, (width - resized.shape[1]) // 2)
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-memory", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_p2a_r1(
        input_memory=arguments.input_memory,
        input_video=arguments.input_video,
        output_dir=arguments.output_dir,
        config_path=arguments.config,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
