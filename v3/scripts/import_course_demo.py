#!/usr/bin/env python3
"""Restore a course demo's saved visual memory for a hash-matched local video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recallvision.video_ai.persistent_memory import PersistentMemoryStore

PACK_FORMAT = "recallvision-course-memory-pack"
PACK_VERSION = 1


class DemoPackError(ValueError):
    """Raised when a demo pack or its source media cannot be trusted."""


def _read_pack(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        pack = source
    else:
        try:
            pack = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DemoPackError(f"Could not read the course demo memory pack: {exc}") from exc
    if not isinstance(pack, dict) or pack.get("format") != PACK_FORMAT or pack.get("version") != PACK_VERSION:
        raise DemoPackError("Unsupported course demo memory pack format or version.")
    if not isinstance(pack.get("pack_id"), str) or not pack["pack_id"].strip():
        raise DemoPackError("The course demo pack has no pack_id.")
    if not isinstance(pack.get("videos"), list):
        raise DemoPackError("The course demo pack has no videos list.")
    return pack


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(path: Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise DemoPackError("The selected source video could not be opened by OpenCV.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    if not math.isfinite(duration) or duration <= 0 or frame_count <= 0:
        raise DemoPackError("The selected source video has no valid duration or frames.")
    return {"duration": duration, "fps": fps, "frame_count": frame_count, "width": width, "height": height}


def _validate_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if not isinstance(expected, dict):
        raise DemoPackError("The course demo record has no source video metadata.")
    for key in ("duration", "fps", "frame_count", "width", "height"):
        if key not in expected:
            raise DemoPackError(f"The course demo record is missing source metadata: {key}.")
    if int(actual["frame_count"]) != int(expected["frame_count"]):
        raise DemoPackError("Decoded frame count does not match the course demo source record.")
    if (int(actual["width"]), int(actual["height"])) != (int(expected["width"]), int(expected["height"])):
        raise DemoPackError("Decoded frame size does not match the course demo source record.")
    if abs(float(actual["fps"]) - float(expected["fps"])) > 0.1:
        raise DemoPackError("Decoded frame rate does not match the course demo source record.")
    if abs(float(actual["duration"]) - float(expected["duration"])) > 0.1:
        raise DemoPackError("Decoded duration does not match the course demo source record.")


def _decode_evidence(
    video_path: Path, video: dict[str, Any], *, fps: float,
) -> list[dict[str, Any]]:
    import cv2

    rows = video.get("evidence_frames", [])
    if not isinstance(rows, list):
        raise DemoPackError("Evidence frame references must be a list.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise DemoPackError("The source video could not be decoded for evidence restoration.")
    decoded: list[dict[str, Any]] = []
    seen: set[int] = set()
    try:
        for row in rows:
            try:
                frame_index = int(row["frame_index"])
                seconds = float(row["time_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DemoPackError("An evidence frame reference is malformed.") from exc
            if frame_index < 0 or frame_index in seen or not math.isfinite(seconds) or seconds < 0:
                raise DemoPackError("An evidence frame index or timestamp is invalid.")
            expected_index = round(seconds * fps)
            if frame_index != expected_index:
                raise DemoPackError(
                    f"Evidence frame {frame_index} does not match its timestamp ({seconds:.3f} s); "
                    f"the source frame should be {expected_index}."
                )
            seen.add(frame_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise DemoPackError(f"Could not decode source evidence frame {frame_index}.")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise DemoPackError(f"Could not encode source evidence frame {frame_index}.")
            decoded.append({"frame_index": frame_index, "time_seconds": seconds, "jpeg_bytes": encoded.tobytes()})
    finally:
        capture.release()
    return decoded


def _entry_for_video(pack: dict[str, Any], digest: str) -> dict[str, Any]:
    for entry in pack["videos"]:
        if isinstance(entry, dict) and entry.get("sha256") == digest:
            return entry
    raise DemoPackError("The selected video does not match the course demo source hash.")


def _validated_saved_answers(records: list[Any], video_hash: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    validated: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            raise DemoPackError("A saved course demo answer is malformed.")
        question = str(record.get("question") or "").strip()
        answer = record.get("answer")
        if not question or not isinstance(answer, dict):
            raise DemoPackError("A saved course demo answer is malformed.")
        if answer.get("video_hash") not in (None, video_hash):
            raise DemoPackError("A saved course demo answer points to another video hash.")
        validated.append((question, answer, record.get("provenance", {})))
    return validated


def _restore_is_complete(
    store: PersistentMemoryStore,
    video_hash: str,
    pack: dict[str, Any],
    entry: dict[str, Any],
    evidence_frames: list[dict[str, Any]],
    saved_answers: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> bool:
    video = store.get_video(video_hash)
    if (
        video is None
        or video["analysis_status"] != "complete"
        or video.get("analysis_metadata", {}).get("course_demo_pack_id") != pack["pack_id"]
        or len(store.list_memories(video_hash)) != len(entry["memories"])
    ):
        return False

    existing_frames = {
        (int(frame["frame_index"]), round(float(frame["time_seconds"]), 3))
        for frame in store.list_evidence_frames(video_hash)
    }
    expected_frames = {
        (int(frame["frame_index"]), round(float(frame["time_seconds"]), 3))
        for frame in evidence_frames
    }
    if existing_frames != expected_frames:
        return False

    cache_context = store.answer_cache_context(
        video_hash,
        verification_config_fingerprint=str(
            pack.get("verification_config_fingerprint") or "course-demo-v1"
        ),
    )
    return all(
        store.get_saved_answer(video_hash, question, cache_context=cache_context) is not None
        for question, _, _ in saved_answers
    )


def _restore_visual_observations(
    store: PersistentMemoryStore, video_hash: str, items: list[dict[str, Any]],
) -> None:
    for item in items:
        if item.get("kind") != "visual_observation":
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        store.save_visual_observation(
            video_hash,
            label=str(item["label"]),
            target_description=str(payload.get("target_description") or item["label"]),
            timestamp_seconds=float(item["start_seconds"]),
            result={
                "verification_status": item.get("verification_status", "unknown"),
                "visibility_state": item.get("visibility_state", "unknown"),
                "identity_status": item.get("identity_status", "unverified"),
                "uncertainty_reason": item.get("uncertainty_reason"),
                "visual_evidence": item.get("visual_evidence", []),
            },
            model_fingerprint=str(item.get("model_fingerprint") or "course-demo"),
            verifier_fingerprint=str(item.get("verification_model_fingerprint") or "course-demo-verifier"),
            config_fingerprint=str(item.get("verification_config_fingerprint") or "course-demo-config"),
        )


def restore_demo_video(
    video_path: str | Path,
    source_pack: str | Path | dict[str, Any],
    *,
    store: PersistentMemoryStore,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Restore persisted memories and source evidence without loading or running models."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    pack = _read_pack(source_pack)
    digest = _source_digest(path)
    entry = _entry_for_video(pack, digest)
    if not isinstance(entry.get("memories"), list) or not isinstance(entry.get("saved_answers"), list):
        raise DemoPackError("The course demo source record must contain memories and saved answers.")
    saved_answers = _validated_saved_answers(entry["saved_answers"], digest)
    actual_metadata = _source_metadata(path)
    _validate_metadata(actual_metadata, entry.get("metadata"))
    evidence_frames = _decode_evidence(path, entry, fps=float(actual_metadata["fps"]))

    existing = store.get_video(digest)
    existing_metadata = existing.get("analysis_metadata", {}) if existing else {}
    if existing and existing["analysis_status"] == "complete":
        if existing_metadata.get("course_demo_pack_id") == pack["pack_id"] and _restore_is_complete(
            store, digest, pack, entry, evidence_frames, saved_answers
        ):
            return {
                "video_hash": digest,
                "source_name": existing["source_name"],
                "memory_count": len(store.list_memories(digest)),
                "saved_answer_count": len(saved_answers),
                "evidence_frame_count": len(evidence_frames),
                "already_restored": True,
                "inference_performed": False,
            }
        if not replace_existing:
            if existing_metadata.get("course_demo_pack_id") != pack["pack_id"]:
                raise DemoPackError(
                    "This video already has memory from another source. Pass --replace-existing to replace it with the course demo memory."
                )

    source_name = str(entry.get("source_name") or path.name)
    model_fingerprint = str(entry.get("model_fingerprint") or "course-demo-memory")
    metadata = entry["metadata"]
    try:
        stored_video = store.register_video(
            path, source_name=source_name, metadata=metadata, model_fingerprint=model_fingerprint
        )
        memories = entry["memories"]
        semantic_memories = [item for item in memories if item.get("kind") != "visual_observation"]
        analysis_metadata = dict(entry.get("analysis_metadata") or {})
        analysis_metadata.update({
            "course_demo_pack_id": pack["pack_id"],
            "course_demo_source_url": entry.get("source_url"),
            "course_demo_license_name": entry.get("license_name"),
            "restore_performed_inference": False,
            "analysis_origin": "previously completed real-video run",
        })
        store.replace_memories(
            digest,
            semantic_memories,
            raw_response="Previously generated real-video memory restored from the course demo pack; no inference was run during restore.",
            analysis_metadata=analysis_metadata,
            complete=True,
        )
        _restore_visual_observations(store, digest, memories)
        store.clear_evidence_frames(digest)
        for frame in evidence_frames:
            store.save_evidence_frame(
                digest,
                frame_index=frame["frame_index"],
                time_seconds=frame["time_seconds"],
                jpeg_bytes=frame["jpeg_bytes"],
            )

        cache_context = store.answer_cache_context(
            digest,
            verification_config_fingerprint=str(pack.get("verification_config_fingerprint") or "course-demo-v1"),
        )
        for question, answer, provenance in saved_answers:
            store.save_answer(
                digest,
                question,
                answer,
                provenance={
                    "source": "course_demo_memory_pack",
                    "pack_id": pack["pack_id"],
                    "historical_provenance": provenance,
                },
                cache_context=cache_context,
            )
    except Exception as exc:
        if store.get_video(digest) is not None:
            store.mark_analysis_failed(digest, f"Course demo restore incomplete: {exc}")
        raise

    return {
        "video_hash": digest,
        "source_name": stored_video["source_name"],
        "memory_count": len(store.list_memories(digest)),
        "saved_answer_count": len(saved_answers),
        "evidence_frame_count": len(evidence_frames),
        "already_restored": False,
        "inference_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Locally downloaded Mixkit source video")
    parser.add_argument(
        "--pack", type=Path,
        default=Path(__file__).resolve().parents[1] / "demo" / "course_memory_pack_v1.json",
        help="Path to the course memory pack JSON",
    )
    parser.add_argument("--memory-dir", type=Path, help="Persistent store directory (defaults to the app store)")
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="Replace a completed memory for the same source video with the course demo memory",
    )
    args = parser.parse_args()
    store = PersistentMemoryStore(args.memory_dir)
    result = restore_demo_video(args.video, args.pack, store=store, replace_existing=args.replace_existing)
    action = "Already restored" if result["already_restored"] else "Restored"
    print(
        f"{action} {result['source_name']}: {result['memory_count']} memory items, "
        f"{result['evidence_frame_count']} evidence frames, {result['saved_answer_count']} saved answers. "
        "No model inference was run."
    )
    print(f"SHA-256: {result['video_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
