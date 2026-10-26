"""Run the P2A association baseline over frozen P1B observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recallvision.memory.appearance import HSVHistogramConfig, extract_hsv_histogram
from recallvision.memory.association import (
    AssociationConfig,
    AssociationSelection,
    associate_tracklets,
    load_association_config,
)
from recallvision.memory.persistent_objects import (
    PersistentObjectBuild,
    build_persistent_objects,
    enrich_observations,
)
from recallvision.memory.store import ObservationMemoryStore
from recallvision.memory.tracklets import TrackletSummary, build_tracklet_summaries
from recallvision.perception.ultralytics_backend import read_video_metadata, sha256_file
from recallvision.query.object_location import answer_where_object
from recallvision.query.parser import parse_query


@dataclass(frozen=True)
class P2ARunResult:
    """Paths and summary emitted by one deterministic formal run."""

    output_dir: Path
    summary: dict[str, Any]


def run_p2a(
    *,
    input_memory: Path,
    input_video: Path,
    output_dir: Path,
    config_path: Path,
) -> P2ARunResult:
    """Build all P2A artifacts without running a detector or model."""

    observations = ObservationMemoryStore(input_memory).load_all()
    if not observations:
        raise ValueError(f"P2A input memory is empty: {input_memory}")
    metadata = read_video_metadata(input_video)
    config = load_association_config(config_path)
    summaries = build_tracklet_summaries(
        observations,
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "tracklet_summaries.json",
        [summary.to_record() for summary in summaries],
    )

    descriptors = _extract_descriptors(
        input_video,
        summaries,
        config=HSVHistogramConfig(
            h_bins=config.appearance_h_bins,
            s_bins=config.appearance_s_bins,
        ),
    )
    selection = associate_tracklets(summaries, descriptors, config=config)
    build = build_persistent_objects(summaries, observations, selection.accepted_edges)
    enriched = enrich_observations(observations, build.mapping)

    _write_json(output_dir / "association_candidates.json", selection.to_records())
    _write_json(output_dir / "association_edges.json", selection.to_records())
    _write_json(output_dir / "persistent_objects.json", [obj.to_record() for obj in build.objects])
    enriched_path = output_dir / "observations_with_persistent_ids.jsonl"
    if enriched_path.exists():
        enriched_path.unlink()
    enriched_store = ObservationMemoryStore(enriched_path)
    for observation in enriched:
        enriched_store.append(observation)

    contact_sheet = output_dir / "bottle_tracklet_contact_sheet.jpg"
    _write_contact_sheet(input_video, summaries, contact_sheet)
    summary = _build_summary(
        observations=observations,
        enriched=enriched,
        summaries=summaries,
        selection=selection,
        build=build,
        metadata=metadata,
        config=config,
        input_memory=input_memory,
        input_video=input_video,
        config_path=config_path,
    )
    _write_json(output_dir / "association_summary.json", summary)
    return P2ARunResult(output_dir=output_dir, summary=summary)


def _extract_descriptors(
    video_path: Path,
    summaries: list[TrackletSummary],
    *,
    config: HSVHistogramConfig,
) -> dict[tuple[str, str, str], tuple[float, ...]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for P2A appearance extraction: {video_path}")
    try:
        descriptors: dict[tuple[str, str, str], tuple[float, ...]] = {}
        for summary in summaries:
            representative = summary.representative_frames[len(summary.representative_frames) // 2]
            frame = _read_frame(capture, representative.frame_index, video_path)
            bbox = representative.bbox
            descriptors[summary.key] = extract_hsv_histogram(
                frame,
                (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                config=config,
            )
        return descriptors
    finally:
        capture.release()


def _write_contact_sheet(
    video_path: Path,
    summaries: list[TrackletSummary],
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    bottle_summaries = [summary for summary in summaries if summary.class_name == "bottle"]
    if not bottle_summaries:
        raise ValueError("P2A contact sheet requires at least one bottle tracklet")
    cell_width, cell_height = 360, 300
    canvas = np.full(
        (len(bottle_summaries) * cell_height, 3 * cell_width, 3),
        245,
        dtype=np.uint8,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for contact sheet: {video_path}")
    try:
        for row, summary in enumerate(bottle_summaries):
            for column, representative in enumerate(summary.representative_frames):
                frame = _read_frame(capture, representative.frame_index, video_path)
                crop = _crop(frame, representative.bbox)
                if crop is None:
                    raise RuntimeError(
                        f"Could not crop tracklet {summary.track_id} at frame {representative.frame_index}"
                    )
                cell = _fit_crop(crop, cell_width, cell_height)
                y0 = row * cell_height
                x0 = column * cell_width
                canvas[y0 : y0 + cell_height, x0 : x0 + cell_width] = cell
                cv2.putText(
                    canvas,
                    f"track={summary.track_id} frame={representative.frame_index}",
                    (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"t={representative.timestamp:.3f}s",
                    (x0 + 8, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def _read_frame(capture: Any, frame_index: int, video_path: Path) -> Any:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _crop(frame: Any, bbox: Any) -> Any:
    height, width = frame.shape[:2]
    left = max(0, min(width, int(bbox.x1)))
    top = max(0, min(height, int(bbox.y1)))
    right = max(0, min(width, int(bbox.x2 + 0.999999)))
    bottom = max(0, min(height, int(bbox.y2 + 0.999999)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _fit_crop(crop: Any, width: int, height: int) -> Any:
    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    crop_height, crop_width = crop.shape[:2]
    scale = min((width - 16) / crop_width, (height - 58) / crop_height)
    resized = cv2.resize(
        crop,
        (max(1, int(round(crop_width * scale))), max(1, int(round(crop_height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    y = 52 + max(0, (height - 52 - resized.shape[0]) // 2)
    x = max(0, (width - resized.shape[1]) // 2)
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _build_summary(
    *,
    observations: list[Any],
    enriched: list[Any],
    summaries: list[TrackletSummary],
    selection: AssociationSelection,
    build: PersistentObjectBuild,
    metadata: dict[str, float | int],
    config: AssociationConfig,
    input_memory: Path,
    input_video: Path,
    config_path: Path,
) -> dict[str, Any]:
    bottle_observations = [item for item in observations if item.class_name == "bottle"]
    bottle_summaries = [item for item in summaries if item.class_name == "bottle"]
    bottle_objects = [item for item in build.objects if item.class_name == "bottle"]
    bottle_edges = [item for item in selection.all_edges if item.class_name == "bottle"]
    bottle_last_seen = max(item.timestamp for item in bottle_observations)
    query = parse_query(
        "Where is the bottle?",
        video_id=observations[0].video_id,
        question_id="p2a-query",
    )
    answer = answer_where_object(query, enriched)
    selected = next(
        (
            item
            for item in enriched
            if item.frame_index == answer.frame_index
            and item.class_name == "bottle"
            and item.persistent_id == answer.target_entity_id
        ),
        None,
    )
    return {
        "p2a_pipeline_status": "PASS",
        "p2a_identity_validation_status": "PENDING_HUMAN_REVIEW",
        "input_memory": str(input_memory),
        "input_video": str(input_video),
        "input_video_sha256": sha256_file(input_video),
        "video_metadata": metadata,
        "video_end_timestamp": metadata["duration"],
        "bottle_observations": len(bottle_observations),
        "bottle_tracklets_before": len(bottle_summaries),
        "input_bottle_track_ids": [item.track_id for item in bottle_summaries],
        "association_method": "class/temporal/spatial/scale/HSV-appearance baseline",
        "appearance_descriptor": {
            "type": "normalized HSV histogram",
            "h_bins": config.appearance_h_bins,
            "s_bins": config.appearance_s_bins,
            "value_channel_used": False,
        },
        "config_file": str(config_path),
        "config": config.to_record(),
        "candidate_edge_count": len(selection.all_edges),
        "bottle_candidate_edge_count": len(bottle_edges),
        "hard_gate_pass_count": sum(edge.hard_gate_pass for edge in selection.all_edges),
        "bottle_hard_gate_pass_count": sum(edge.hard_gate_pass for edge in bottle_edges),
        "accepted_edge_count": len(selection.accepted_edges),
        "bottle_accepted_edge_count": sum(edge.accepted for edge in bottle_edges),
        "bottle_persistent_objects_after": len(bottle_objects),
        "persistent_object_chains": [
            {
                "persistent_id": item.persistent_id,
                "class_name": item.class_name,
                "track_ids": list(item.track_ids),
            }
            for item in build.objects
        ],
        "query": {
            "question": "Where is the bottle?",
            "answer": answer.answer,
            "timestamp": answer.timestamp,
            "frame_index": answer.frame_index,
            "track_id": selected.track_id if selected else None,
            "persistent_id": selected.persistent_id if selected else None,
        },
        "last_bottle_observation": bottle_last_seen,
        "bottle_staleness": float(metadata["duration"]) - bottle_last_seen,
        "original_observation_preservation": _preserves_original_fields(observations, enriched),
        "query_compatibility": answer.status.value == "OK",
        "training_run": False,
        "deep_reid_used": False,
        "vlm_used": False,
        "human_review_required": True,
    }


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-memory", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_p2a(
        input_memory=arguments.input_memory,
        input_video=arguments.input_video,
        output_dir=arguments.output_dir,
        config_path=arguments.config,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
