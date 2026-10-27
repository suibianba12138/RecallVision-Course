from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recallvision.video_ai.product_logic import (
    build_adaptive_visibility_probe_times,
    build_event_probe_times,
    build_visibility_probe_times,
    classify_event_claim,
    extract_visibility_observations,
    resolve_event_disagreement,
    resolve_visibility_observations,
    route_question,
    visibility_target,
)
from recallvision.video_ai.persistent_memory import (
    MEMORY_GENERATION_VERSION,
    MEMORY_SCHEMA_VERSION,
    PersistentMemoryStore,
    answer_from_memory,
    is_event_summary_question,
    is_object_inventory_question,
    normalize_memory_document,
    parse_memory_response,
    saved_answer_requires_visual_recheck,
)
from recallvision.video_ai.visual_evidence import (
    combine_detector_crop_reviews,
    filter_target_candidates,
    review_detector_frame,
)
from recallvision.video_ai.runtime import (
    DEFAULT_MODEL_PATHS,
    analyze_memory_frames,
    ask_target_candidate_crops,
    ask_event_recheck_frames,
    ask_frames,
    ask_video,
    ask_visibility_frames,
)
from recallvision.video_ai.video_cache import VideoAnalysisCache


UPLOAD_ROOT = Path(tempfile.gettempdir()) / "recallvision-v3-video-memory" / "uploads"
CACHE_ROOT = Path(os.environ.get(
    "RECALLVISION_VIDEO_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "recallvision-v3-video-memory" / "cache"),
)).expanduser()
VIDEO_CACHE = VideoAnalysisCache(CACHE_ROOT)
MAX_VIDEO_BYTES = 250 * 1024 * 1024
MAX_VIDEO_SECONDS = 180
QWEN_FINGERPRINT = "qwen3.5-2b:aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
TIMELENS_FINGERPRINT = "timelens2-2b:70facd614adbacf7d09bd6e6389ce11ece256e9bb0dcc52194c28882c71c8588"
CACHE_SCHEMA = "video-memory-v3"
MEMORY_FINGERPRINT = f"{QWEN_FINGERPRINT}:{MEMORY_GENERATION_VERSION}"
GROUNDING_DINO_PATH = Path(os.environ.get(
    "RECALLVISION_GROUNDING_DINO_PATH",
    "~/.cache/recallvision-v3/models/grounding-dino-tiny-a2bb814dd30d776dcf7e30523b00659f4f141c71",
)).expanduser()
GROUNDING_DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_DINO_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
GROUNDING_DINO_FINGERPRINT = f"grounding-dino-tiny:{GROUNDING_DINO_REVISION}:{GROUNDING_DINO_SHA256}"
VERIFICATION_MODEL_FINGERPRINT = f"{GROUNDING_DINO_FINGERPRINT}+{QWEN_FINGERPRINT}"
VERIFICATION_CONFIG_FINGERPRINT = f"adaptive-visibility-probe-v5:reference-crop-identity-v1:frame-local-crop-agreement-v2:{GROUNDING_DINO_FINGERPRINT}:qwen-crop-id-v2"
MEMORY_STORE = PersistentMemoryStore()


def _save_upload(uploaded: Any) -> tuple[Path, str]:
    data = uploaded.getvalue()
    if not data:
        raise ValueError("The selected video is empty.")
    if len(data) > MAX_VIDEO_BYTES:
        raise ValueError("Videos must be 250 MB or smaller.")
    digest = hashlib.sha256(data).hexdigest()
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_ROOT / f"{digest}{Path(uploaded.name).suffix.lower()}"
    if not path.is_file():
        path.write_bytes(data)
    return path, digest


def _video_info(path: Path, video_hash: str | None = None) -> dict[str, Any]:
    if video_hash:
        cached = VIDEO_CACHE.get_metadata(video_hash)
        if cached and all(key in cached for key in ("fps", "frame_count", "width", "height", "duration")):
            return cached
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("This video could not be opened by the local decoder.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    if duration <= 0 or duration > MAX_VIDEO_SECONDS:
        raise ValueError(f"Video duration must be between 0 and {MAX_VIDEO_SECONDS} seconds.")
    metadata = {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}
    if video_hash:
        VIDEO_CACHE.put_metadata(video_hash, metadata)
    return metadata


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_intervals(raw: str, duration: float) -> list[list[float]]:
    text = re.sub(r"```(?:json)?|```", "", raw.strip(), flags=re.IGNORECASE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    intervals: list[list[float]] = []
    if not isinstance(parsed, list):
        return intervals
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            a, b = sorted((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b > duration:
            continue
        intervals.append([a, b])
    return intervals


def _extract_evidence_times(value: dict[str, Any] | None, duration: float) -> list[float]:
    if not value:
        return []
    raw_times = value.get("evidence_seconds", [])
    if isinstance(raw_times, str):
        try:
            raw_times = json.loads(raw_times)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_times, list):
        return []
    times: list[float] = []
    for item in raw_times:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                times.extend((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
        else:
            try:
                times.append(float(item))
            except (TypeError, ValueError):
                continue
    return sorted({round(t, 2) for t in times if math.isfinite(t) and 0 <= t <= duration})[:8]


def _visibility_query(question: str) -> str | None:
    target = visibility_target(question)
    if target:
        return (
            f"{target} is physically visible in the video. Return only intervals "
            "where the object itself appears; exclude times after it leaves the view."
        )
    return None


def _source_frame(path: Path, seconds: float, video_hash: str | None = None, metadata: dict[str, Any] | None = None) -> Any:
    info = metadata or _video_info(path, video_hash)
    fps = float(info.get("fps", 0.0))
    frame_count = int(info.get("frame_count", 0))
    frame_index = max(0, round(seconds * fps)) if fps > 0 else 0
    if frame_count > 0:
        frame_index = min(frame_index, frame_count - 1)

    def decode_jpeg() -> bytes:
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                return b""
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return encoded.tobytes() if ok else b""
        finally:
            capture.release()

    try:
        if video_hash:
            encoded = VIDEO_CACHE.get_or_create_frame(video_hash, frame_index, decode_jpeg)
        else:
            encoded = decode_jpeg()
        if not encoded:
            return None
        import numpy as np

        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        actual_seconds = frame_index / fps if fps > 0 else seconds
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), actual_seconds
    except (OSError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def _load_target_detector() -> Any:
    from recallvision.perception.grounding_dino_backend import GroundingDinoTargetDetector

    return GroundingDinoTargetDetector(
        asset_root=GROUNDING_DINO_PATH,
        revision=GROUNDING_DINO_REVISION,
        expected_checkpoint_sha256=GROUNDING_DINO_SHA256,
        box_threshold=0.4,
        text_threshold=0.3,
        device="cpu",
    )


def _review_target_memory_frames(
    *, store: PersistentMemoryStore, video_hash: str, video_path: Path,
    metadata: dict[str, Any], target: str, memories: list[dict[str, Any]], focus_end: bool = False,
) -> dict[str, Any]:
    """Review saved samples, then spend a few cached source-frame checks on large gaps."""
    from PIL import Image

    target_terms = set(re.findall(r"[\w]+", target.casefold()))
    matching = [item for item in memories if item["kind"] == "entity_observation" and (
        target.casefold() in (item["label"] + " " + item["claim"]).casefold()
        or bool(target_terms & set(re.findall(r"[\w]+", (item["label"] + " " + item["claim"]).casefold())))
    )]
    requested = sorted({
        float(seconds)
        for item in matching
        for seconds in item.get("evidence_seconds", [item["start_seconds"]])
    })
    if focus_end:
        cutoff = max(0.0, float(metadata["duration"]) - 2.1)
        requested = [seconds for seconds in requested if seconds >= cutoff]
    if not requested:
        duration = float(metadata["duration"])
        fps = max(float(metadata.get("fps", 1.0)), 1.0)
        if focus_end:
            requested = [max(0.0, duration - offset) for offset in (2.0, 1.0, 1.0 / fps)]
        else:
            requested = [round(max(0.0, duration - 1.0 / fps) * index / 5, 3) for index in range(6)]
    requested = requested[-3:] if focus_end else requested[:8]
    detector: Any | None = None
    cache_hits = 0
    analysis_lookups = 0
    detector_load_seconds: float | None = None

    def detect_at(rgb: Any, actual_seconds: float, *, box_threshold: float, text_threshold: float) -> tuple[list[dict[str, Any]], int, str, int]:
        nonlocal detector, detector_load_seconds, cache_hits, analysis_lookups
        frame_index = max(0, round(actual_seconds * float(metadata.get("fps", 0.0))))
        frame_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        config = {
            "frame_index": frame_index,
            "timestamp_seconds": round(actual_seconds, 3),
            "frame_sha256": frame_sha256,
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "candidate_filter": "target-label-nms-v1",
            "schema": CACHE_SCHEMA,
        }

        def infer() -> dict[str, Any]:
            nonlocal detector, detector_load_seconds
            import time

            before_load = time.perf_counter()
            if detector is None:
                detector = _load_target_detector()
                detector_load_seconds = time.perf_counter() - before_load
            return {"detections": [item.as_record() for item in detector.detect(
                Image.fromarray(rgb), video_id=video_hash[:16], video_sha256=video_hash,
                target_description=target, frame_index=frame_index,
                timestamp_seconds=actual_seconds, box_threshold=box_threshold,
                text_threshold=text_threshold,
            )]}

        analysis_lookups += 1
        result = _cached_analysis(
            video_hash, "target_detection", target, GROUNDING_DINO_FINGERPRINT, config, infer,
        )
        cache_hits += int(bool(result.get("cache_hit")))
        return list(result.get("detections", [])), frame_index, frame_sha256, int(bool(result.get("cache_hit")))

    def collect_review(rgb: Any, actual_seconds: float, detections: list[dict[str, Any]], *, minimum_score: float) -> tuple[dict[str, Any], list[tuple[Any, float]], list[dict[str, Any]]]:
        frame_index = max(0, round(actual_seconds * float(metadata.get("fps", 0.0))))
        frame_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        target_candidates = filter_target_candidates(detections, target)
        for rank, candidate in enumerate(target_candidates):
            candidate["candidate_id"] = f"{frame_index}:{rank}"
        review = review_detector_frame(
            target_candidates, width=int(metadata["width"]), height=int(metadata["height"]),
            frame_index=frame_index, timestamp_seconds=actual_seconds,
            frame_sha256=frame_sha256, minimum_score=minimum_score,
        )
        crops: list[tuple[Any, float]] = []
        crop_records: list[dict[str, Any]] = []
        height, width = rgb.shape[:2]
        # Multiple distinct target-like regions remain ambiguous unless each crop is
        # independently reviewed. IDs keep those decisions attached to the right box.
        for candidate in review["visual_evidence"][0]["detections"]:
            box = candidate.get("bbox_xyxy")
            if not box or len(box) != 4 or candidate.get("rejection_reason") == "box or score is outside the source frame bounds":
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            pad_x, pad_y = max(8, int((x2 - x1) * 0.12)), max(8, int((y2 - y1) * 0.12))
            left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
            right, bottom = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if right <= left or bottom <= top:
                continue
            crop = Image.fromarray(rgb[top:bottom, left:right])
            crops.append((crop, actual_seconds))
            crop_records.append({
                "candidate_id": candidate["candidate_id"],
                "frame_index": frame_index,
                "timestamp_seconds": round(actual_seconds, 3),
                "frame_sha256": frame_sha256,
                "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                "bbox_xyxy": [x1, y1, x2, y2],
                "detector_score": candidate.get("score"),
            })
        return review, crops, crop_records

    def review_crops(
        crops: list[tuple[Any, float]], records: list[dict[str, Any]], *, prompt_version: str,
        reference_frame: tuple[Any, float] | None = None,
        reference_info: dict[str, Any] | None = None,
        route_name: str = "target_crop_review",
    ) -> dict[str, dict[str, Any]]:
        nonlocal cache_hits, analysis_lookups
        if not crops:
            return {}
        ids = [str(record["candidate_id"]) for record in records]
        crop_config = {
            "candidates": records, "schema": CACHE_SCHEMA, "prompt": prompt_version,
            "target": target,
        }
        if reference_frame is not None:
            crop_config["reference"] = reference_info or {
                "timestamp_seconds": round(float(reference_frame[1]), 3),
            }
        analysis_lookups += 1
        crop_result = _cached_analysis(
            video_hash, route_name, target, QWEN_FINGERPRINT, crop_config,
            lambda: ask_target_candidate_crops(
                crops, target, candidate_ids=ids, reference_frame=reference_frame, max_new_tokens=256,
            ),
        )
        cache_hits += int(bool(crop_result.get("cache_hit")))
        by_id: dict[str, dict[str, Any]] = {}
        observations = extract_visibility_observations(crop_result.get("raw_response", ""))
        for observation in observations:
            candidate_id = observation.get("candidate_id")
            if candidate_id is None:
                # Older model replies may omit IDs. Timestamp fallback is safe only if
                # precisely one crop was shown at that time.
                try:
                    seconds = float(observation.get("seconds"))
                except (TypeError, ValueError):
                    continue
                same_time = [record for record in records if abs(float(record["timestamp_seconds"]) - seconds) <= 0.05]
                if len(same_time) != 1:
                    continue
                candidate_id = same_time[0]["candidate_id"]
            candidate_id = str(candidate_id)
            if candidate_id not in ids:
                continue
            by_id[candidate_id] = {
                **observation, "model_fingerprint": QWEN_FINGERPRINT,
                "crop_sha256": next(record["crop_sha256"] for record in records if record["candidate_id"] == candidate_id),
            }
        return by_id

    def finalize_reviews(reviews: list[dict[str, Any]], crop_reviews: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [combine_detector_crop_reviews(review, {
            str(candidate.get("candidate_id")): crop_reviews[str(candidate.get("candidate_id"))]
            for candidate in review["visual_evidence"][0].get("detections", [])
            if str(candidate.get("candidate_id")) in crop_reviews
        }) for review in reviews]

    frame_reviews: list[dict[str, Any]] = []
    primary_crops: list[tuple[Any, float]] = []
    primary_crop_records: list[dict[str, Any]] = []
    for seconds in requested:
        source = _source_frame(video_path, seconds, video_hash, metadata)
        if source is None:
            continue
        rgb, actual_seconds = source
        detections, _frame_index, _frame_sha256, detection_hit = detect_at(
            rgb, actual_seconds, box_threshold=0.4, text_threshold=0.3,
        )
        review, crops, records = collect_review(rgb, actual_seconds, detections, minimum_score=0.4)
        review["cache_hit"] = bool(detection_hit)
        frame_reviews.append(review)
        primary_crops.extend(crops)
        primary_crop_records.extend(records)

    primary_crop_reviews = review_crops(primary_crops, primary_crop_records, prompt_version="target-crop-id-v2")
    combined_reviews = finalize_reviews(frame_reviews, primary_crop_reviews)

    # Only refine gaps after an exact-frame crop has supported a sighting. A detector
    # miss cannot become an absence anchor and no target-specific time is injected.
    adaptive_times = [] if focus_end else build_adaptive_visibility_probe_times(
        combined_reviews, duration=float(metadata["duration"]), fps=float(metadata.get("fps", 1.0)),
        max_extra_frames=3,
    )
    supplemental_reviews: list[dict[str, Any]] = []
    supplemental_crops: list[tuple[Any, float]] = []
    supplemental_crop_records: list[dict[str, Any]] = []
    for seconds in adaptive_times:
        source = _source_frame(video_path, seconds, video_hash, metadata)
        if source is None:
            continue
        rgb, actual_seconds = source
        detections, _frame_index, _frame_sha256, detection_hit = detect_at(
            rgb, actual_seconds, box_threshold=0.2, text_threshold=0.2,
        )
        review, crops, records = collect_review(rgb, actual_seconds, detections, minimum_score=0.2)
        review["adaptive_probe"] = True
        review["cache_hit"] = bool(detection_hit)
        supplemental_reviews.append(review)
        supplemental_crops.extend(crops)
        supplemental_crop_records.extend(records)

    supplemental_crop_reviews = review_crops(
        supplemental_crops, supplemental_crop_records, prompt_version="target-crop-adaptive-id-v1",
    )

    # A candidate may be another backpack-shaped object. Compare any supplemental
    # crop accepted as the requested category to the exact crop that anchored the
    # target identity earlier in this clip. Ambiguous comparisons stay unresolved.
    supported_anchors = [
        {"review": review, "candidate": candidate}
        for review in combined_reviews
        for candidate in review["visual_evidence"][0].get("detections", [])
        if candidate.get("target_presence_corroborated")
    ]
    if supported_anchors and supplemental_crops:
        anchor = max(
            supported_anchors,
            key=lambda entry: float(entry["review"]["visual_evidence"][0]["timestamp_seconds"]),
        )
        anchor_candidate_id = str(anchor["candidate"].get("candidate_id", ""))
        anchor_index = next((index for index, record in enumerate(primary_crop_records)
                             if str(record.get("candidate_id")) == anchor_candidate_id), None)
        if anchor_index is not None:
            identity_indices = [index for index, record in enumerate(supplemental_crop_records)
                                if supplemental_crop_reviews.get(str(record.get("candidate_id")), {}).get("visibility") == "present"]
            if identity_indices:
                anchor_info = primary_crop_records[anchor_index]
                identity_records = [supplemental_crop_records[index] for index in identity_indices]
                identity_frames = [supplemental_crops[index] for index in identity_indices]
                identity_reviews = review_crops(
                    identity_frames, identity_records,
                    prompt_version="target-crop-reference-match-v1",
                    reference_frame=primary_crops[anchor_index], reference_info=anchor_info,
                    route_name="target_identity_crop_review",
                )
                # If the identity model omits a crop, do not fall back to its broader
                # category-only decision as though it had matched the reference.
                for record in identity_records:
                    candidate_id = str(record["candidate_id"])
                    supplemental_crop_reviews[candidate_id] = {
                        **supplemental_crop_reviews.get(candidate_id, {}),
                        "visibility": "uncertain", "reference_match": "uncertain",
                    }
                for candidate_id, identity_review in identity_reviews.items():
                    if identity_review.get("reference_match") not in {"same", "different", "uncertain"}:
                        identity_reviews[candidate_id] = {
                            **identity_review, "visibility": "uncertain", "reference_match": "uncertain",
                        }
                supplemental_crop_reviews.update(identity_reviews)
    supplemental_reviews = finalize_reviews(supplemental_reviews, supplemental_crop_reviews)

    for review in combined_reviews:
        seconds = round(float(review["visual_evidence"][0]["timestamp_seconds"]), 3)
        for item in matching:
            if any(abs(float(value) - seconds) <= 0.05 for value in item.get("evidence_seconds", [])):
                store.update_visual_verification(
                    item["item_id"], review, verifier_fingerprint=VERIFICATION_MODEL_FINGERPRINT,
                    config_fingerprint=VERIFICATION_CONFIG_FINGERPRINT,
                )

    combined_supplemental: list[dict[str, Any]] = []
    for review in supplemental_reviews:
        record = review["visual_evidence"][0]
        candidates = record.get("detections", [])
        label = next((str(item.get("raw_model_label", target)) for item in candidates), target)
        seconds = round(float(record["timestamp_seconds"]), 3)
        store.save_visual_observation(
            video_hash, label=label, target_description=target, timestamp_seconds=seconds,
            result=review, model_fingerprint=MEMORY_FINGERPRINT,
            verifier_fingerprint=VERIFICATION_MODEL_FINGERPRINT,
            config_fingerprint=f"{VERIFICATION_CONFIG_FINGERPRINT}:adaptive-probe-v1",
        )
        combined_supplemental.append(review)

    combined_reviews = sorted(
        [*combined_reviews, *combined_supplemental],
        key=lambda review: float(review["visual_evidence"][0]["timestamp_seconds"]),
    )
    if combined_supplemental:
        _persist_evidence_frames(
            store, video_hash, video_path, metadata,
            [float(review["visual_evidence"][0]["timestamp_seconds"]) for review in combined_supplemental],
        )

    supported: list[dict[str, Any]] = []
    for review in combined_reviews:
        candidates = review["visual_evidence"][0]["detections"]
        accepted = [candidate for candidate in candidates if candidate.get("target_presence_corroborated")]
        if review["verification_status"] == "corroborated" and len(accepted) == 1:
            supported.append({"review": review, "candidate": accepted[0]})
    return {
        "frame_reviews": combined_reviews,
        "supported_candidates": supported,
        "cache_hits": cache_hits,
        "analysis_lookups": analysis_lookups,
        "candidate_frame_count": len(combined_reviews),
        "primary_frame_count": len(frame_reviews),
        "supplemental_frame_count": len(combined_supplemental),
        "supplemental_times": [round(float(review["visual_evidence"][0]["timestamp_seconds"]), 3) for review in combined_supplemental],
        "cache_hit": cache_hits == analysis_lookups,
        "detector_loaded": detector is not None,
        "detector_load_seconds": detector_load_seconds,
    }


def _persist_evidence_frames(
    store: PersistentMemoryStore, video_hash: str, video_path: Path,
    metadata: dict[str, Any], seconds: list[float], *, limit: int = 24,
) -> list[Path]:
    saved: list[Path] = []
    for requested in sorted({round(float(value), 3) for value in seconds})[:limit]:
        source = _source_frame(video_path, requested, video_hash, metadata)
        if source is None:
            continue
        rgb, actual = source
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            continue
        index = max(0, round(actual * float(metadata.get("fps", 1.0))))
        saved.append(store.save_evidence_frame(
            video_hash, frame_index=index, time_seconds=actual, jpeg_bytes=encoded.tobytes()
        ))
    return saved


def _memory_source_frames(
    video_path: Path, video_hash: str, metadata: dict[str, Any], *, max_frames: int = 32
) -> list[tuple[Any, float]]:
    from PIL import Image
    import numpy as np

    duration = float(metadata["duration"])
    fps = float(metadata.get("fps", 0.0))
    count = min(max_frames, max(4, math.ceil(duration / 1.5)))
    last_time = max(0.0, duration - (1.0 / fps if fps > 0 else 0.0))
    requested_times = [last_time * index / max(1, count - 1) for index in range(count)]
    frame_times: dict[int, float] = {}
    for requested in requested_times:
        frame_index = max(0, round(requested * fps)) if fps > 0 else 0
        frame_count = int(metadata.get("frame_count", 0))
        if frame_count > 0:
            frame_index = min(frame_index, frame_count - 1)
        frame_times.setdefault(frame_index, frame_index / fps if fps > 0 else requested)

    def decode_many(frame_indices: list[int]) -> dict[int, bytes]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return {}
        result: dict[int, bytes] = {}
        try:
            for frame_index in sorted(frame_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = capture.read()
                if not ok:
                    continue
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if encoded_ok:
                    result[frame_index] = encoded.tobytes()
        finally:
            capture.release()
        return result

    encoded_frames = VIDEO_CACHE.get_or_create_frames(video_hash, list(frame_times), decode_many)
    frames: list[tuple[Any, float]] = []
    for frame_index, encoded in sorted(encoded_frames.items()):
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        frames.append((Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), frame_times[frame_index]))
    if len(frames) < 2:
        raise ValueError("Could not decode enough timestamped source frames to build visual memory.")
    return frames


def _generate_persistent_memory(
    store: PersistentMemoryStore, video_hash: str, video_path: Path, metadata: dict[str, Any],
    progress: Any | None = None,
) -> dict[str, Any]:
    if progress:
        progress("Decoding and reusing timestamped source frames…")
    frames = _memory_source_frames(video_path, video_hash, metadata)
    if progress:
        progress(f"Analyzing {len(frames)} sampled frames with the local video model…")
    result = analyze_memory_frames(frames, max_new_tokens=1024)
    raw = str(result.get("raw_response", ""))
    try:
        document = parse_memory_response(raw)
    except ValueError as exc:
        store.mark_analysis_failed(video_hash, str(exc), raw_response=raw)
        raise
    memories = normalize_memory_document(
        document, duration=float(metadata["duration"]), model_fingerprint=MEMORY_FINGERPRINT,
        sampled_timestamps=result.get("source_timestamps", [seconds for _, seconds in frames]),
    )
    if not memories:
        store.mark_analysis_failed(
            video_hash, "No timestamped visual memories passed validation.", raw_response=raw
        )
        raise ValueError("The model returned no timestamped visual memories that passed validation.")
    store.replace_memories(
        video_hash, memories, raw_response=raw,
        analysis_metadata={
            **{key: value for key, value in result.items() if key != "raw_response"},
            "partial_output": bool(document.get("_partial")),
        },
        complete=not bool(document.get("_partial")),
    )
    if progress:
        progress(f"Saving {len(memories)} source-linked memory records and evidence frames…")
    evidence_times = [seconds for item in memories for seconds in item["evidence_seconds"]]
    _persist_evidence_frames(store, video_hash, video_path, metadata, evidence_times)
    return {"result": result, "memory_count": len(memories)}


def _cached_analysis(
    video_hash: str, route: str, query: str, model: str, config: dict[str, Any], call: Any
) -> dict[str, Any]:
    cached = VIDEO_CACHE.get_analysis(video_hash, route, query, model, config)
    if cached is not None:
        return {**cached, "cache_hit": True}
    result = call()
    VIDEO_CACHE.put_analysis(video_hash, route, query, model, result, config)
    return {**result, "cache_hit": False}


def _parsed_response(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    return _parse_json_object(str(result.get("raw_response", "")))


def _model_fps(duration: float, route: str, preferred_fps: float) -> float:
    target = preferred_fps
    frame_budget = 32 if route == "qwen" else 48
    return max(0.25, min(target, frame_budget / max(duration, 1.0)))


st.set_page_config(page_title="RecallVision Video Memory", page_icon="🧠", layout="wide")
st.title("RecallVision · Video Memory")
st.caption("Upload a clip, explore source-backed moments, then ask a question. Local analysis stays on this machine.")

with st.sidebar:
    st.subheader("Local models")
    st.markdown("**Video understanding**  \nQwen3.5-2B")
    st.markdown("**Temporal grounding**  \nTimeLens2-2B")
    st.caption("CPU analysis may take a few minutes. Repeated questions reuse the local video and model cache.")

uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "m4v", "avi", "webm"])
library = MEMORY_STORE.list_videos()
video_library = {item["video_hash"]: item for item in library}
selected_hash = st.selectbox(
    "Or reopen a saved video",
    options=[""] + list(video_library),
    key="saved_video_selector",
    format_func=lambda value: (
        "Select a saved video…" if not value else
        f"{'Course demo · ' if video_library[value].get('analysis_metadata', {}).get('course_demo_pack_id') else ''}"
        f"{video_library[value]['source_name']} · {video_library[value]['metadata'].get('duration', 0):.1f} s · {value[:10]}"
    ),
)
if uploaded is None and not selected_hash:
    if not library:
        st.info("Upload a short clip. RecallVision will build its visual memory automatically, then you can explore or ask questions.")
    else:
        st.info("Choose a saved video above to reopen its persistent visual memory, or upload another clip.")
    st.stop()

try:
    if uploaded is not None:
        staged_path, staged_hash = _save_upload(uploaded)
        metadata = _video_info(staged_path, staged_hash)
        video_record = MEMORY_STORE.register_video(
            staged_path, source_name=uploaded.name, metadata=metadata,
            model_fingerprint=MEMORY_FINGERPRINT,
        )
        video_hash = video_record["video_hash"]
        video_path = Path(video_record["source_path"])
    else:
        video_hash = selected_hash
        video_record = MEMORY_STORE.ensure_analysis_version(video_hash, MEMORY_FINGERPRINT)
        if video_record is None:
            raise ValueError("The selected saved video is no longer in the local memory database.")
        video_path = Path(video_record["source_path"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Saved source video is missing: {video_path}")
        metadata = video_record["metadata"]
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Persistent source video could not be opened: {video_path}")
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

generation_error = None
generation_error_detail = None
retry_memory = False
if video_record["analysis_status"] in {"failed", "partial"}:
    if video_record["analysis_status"] == "failed":
        st.error(f"Automatic memory generation failed: {video_record.get('analysis_error') or 'unknown error'}")
    else:
        st.warning("Automatic memory generation returned a partial structured record set.")
    retry_memory = st.button("Retry automatic memory generation", key=f"retry-memory-{video_hash}")
if video_record["analysis_status"] == "pending" or retry_memory:
    with st.status("Building persistent visual memory…", expanded=True) as analysis_status:
        try:
            _generate_persistent_memory(
                MEMORY_STORE, video_hash, video_path, metadata, progress=analysis_status.write
            )
            analysis_status.update(label="Visual memory saved", state="complete", expanded=False)
        except Exception as exc:
            generation_error_detail = f"{type(exc).__name__}: {exc}"
            generation_error = "Local visual memory could not be completed. The video is saved and can be retried."
            MEMORY_STORE.mark_analysis_failed(video_hash, generation_error_detail)
            analysis_status.update(label="Visual memory generation failed", state="error", expanded=True)
    video_record = MEMORY_STORE.get_video(video_hash) or video_record
if generation_error:
    st.error(generation_error)
    st.info("Your saved video remains available. Retry memory generation above, or continue with an existing memory.")
    if generation_error_detail:
        with st.expander("Technical details"):
            st.code(generation_error_detail)

analyze_tab, explore_tab, ask_tab = st.tabs(["Analyze", "Explore", "Ask"])
seek_time = min(max(0.0, float(st.session_state.get("video_memory_seek_time", 0.0))), metadata["duration"])
memory_items = MEMORY_STORE.list_memories(video_hash, model_fingerprint=MEMORY_FINGERPRINT)
with analyze_tab:
    player, details = st.columns([1.45, 1])
    with player:
        st.video(video_path, start_time=max(0.0, seek_time))
    with details:
        st.markdown(f"**{metadata['duration']:.1f} s** · {metadata['width']} × {metadata['height']} · {metadata['fps']:.2f} fps")
        st.caption(f"Video SHA-256: `{video_hash[:16]}…` · source video and visual memory are saved locally")
        if seek_time:
            st.caption(f"Player starts near evidence at {seek_time:.2f} s")
        count_col, cache_col = st.columns(2)
        count_col.metric("Persistent memory items", len(memory_items))
        cache_col.metric("Cached model analyses", len(VIDEO_CACHE.list_analyses(video_hash)))
        demo_metadata = video_record.get("analysis_metadata", {})
        if demo_metadata.get("course_demo_pack_id"):
            st.info(
                "Course demo memory restored from a previously completed real-video run. "
                "Restoring this saved memory did not run models."
            )
            source_url = demo_metadata.get("course_demo_source_url")
            license_name = demo_metadata.get("course_demo_license_name")
            if source_url and license_name:
                st.caption(f"Course media source: {source_url} · {license_name}")
        status_label = {
            "complete": "Generated · review evidence", "partial": "Partial · retry available",
            "pending": "Waiting for analysis", "failed": "Needs retry"
        }.get(video_record["analysis_status"], video_record["analysis_status"].title())
        st.write(f"**Memory status:** {status_label}")
        st.caption("Memory generation, source clips, and evidence frames stay in the local RecallVision data directory.")

with explore_tab:
    st.subheader("Persistent visual memory")
    if not memory_items:
        st.info("No structured memory records are available for this video yet.")
    else:
        st.caption("Each card separates a model-generated claim from frame-local visual checks. UNKNOWN means the source evidence did not settle visibility.")
    entity_groups: dict[str, list[dict[str, Any]]] = {}
    other_memories: list[dict[str, Any]] = []
    for memory in memory_items:
        if memory["kind"] == "entity_observation":
            entity_id = memory.get("payload", {}).get("entity_id")
            entity_groups.setdefault(str(entity_id or memory["label"]).casefold(), []).append(memory)
        else:
            other_memories.append(memory)

    def _render_memory_observation(memory: dict[str, Any]) -> None:
        timestamp = float(memory["start_seconds"])
        state = f"{memory['verification_status'].upper()} · {memory['visibility_state'].upper()}"
        st.markdown(f"**{timestamp:.2f} s** · `{state}`")
        st.write("**Semantic claim** · " + memory["claim"])
        st.caption(
            f"Identity: {memory['identity_status']} · model-reported confidence: {memory['confidence']} "
            f"(not calibrated) · {memory.get('uncertainty_reason') or 'No uncertainty note saved.'}"
        )
        if memory.get("visual_evidence"):
            for frame_record in memory["visual_evidence"]:
                detections = frame_record.get("detections", [])
                if detections:
                    for item in detections:
                        label = item.get("raw_model_label") or memory["label"]
                        box = item.get("bbox_xyxy")
                        if item.get("target_presence_corroborated"):
                            crop = item.get("crop_review", {})
                            identity = crop.get("reference_match")
                            suffix = f" · reference match: {identity}" if identity else ""
                            st.caption(
                                f"Crop-reviewed target candidate · {label} · detector score {item.get('score')} "
                                f"(uncalibrated) · box {box}{suffix}"
                            )
                        elif item.get("crop_review"):
                            crop = item["crop_review"]
                            st.caption(
                                f"Detector proposal · {label} · crop review: {crop.get('visibility', 'unknown')} "
                                f"· reference match: {crop.get('reference_match', 'not checked')} · not counted as target support."
                            )
                        elif item.get("candidate_accepted"):
                            st.caption(
                                f"Detector proposal only · {label} · score {item.get('score')} (uncalibrated); "
                                "the target was not visually confirmed in this frame."
                            )
                        else:
                            st.caption(f"Rejected detector proposal · {item.get('rejection_reason', 'not accepted')}")
                else:
                    st.caption("Visual review: no acceptable candidate; this does not establish absence.")
        evidence_path = MEMORY_STORE.find_evidence_frame(video_hash, timestamp)
        evidence_col, seek_col = st.columns([4, 1])
        with evidence_col:
            if evidence_path:
                st.image(str(evidence_path), caption=f"Original source frame · {timestamp:.2f} s", use_container_width=True)
            else:
                st.caption(f"Original source frame at {timestamp:.2f} s can be decoded from the saved video.")
        with seek_col:
            if st.button(f"Play {timestamp:.2f} s", key=f"memory-seek-{memory['item_id']}-{timestamp}"):
                st.session_state["video_memory_seek_time"] = timestamp
                st.rerun()

    for entity_key, observations in sorted(entity_groups.items()):
        observations.sort(key=lambda item: float(item["start_seconds"]))
        label = observations[0]["label"]
        latest = observations[-1]["start_seconds"]
        with st.expander(f"Object · {label} · {len(observations)} sampled observations · latest {latest:.2f} s"):
            for memory in observations:
                _render_memory_observation(memory)
                if memory is not observations[-1]:
                    st.divider()

    for memory in other_memories:
        start_seconds = float(memory["start_seconds"])
        end_seconds = float(memory["end_seconds"])
        time_label = f"{start_seconds:.2f} s" if abs(end_seconds - start_seconds) < 0.05 else f"{start_seconds:.2f}–{end_seconds:.2f} s"
        with st.expander(f"{memory['kind'].replace('_', ' ').title()} · {memory['label']} · {time_label}"):
            st.write("**Semantic claim** · " + memory["claim"])
            st.markdown(f"**{memory['verification_status'].upper()}** · {memory['uncertainty_reason']}")
            st.caption(f"Source timestamps: {', '.join(f'{value:.2f} s' for value in memory['evidence_seconds'])}")
            for seconds in memory["evidence_seconds"]:
                evidence_path = MEMORY_STORE.find_evidence_frame(video_hash, float(seconds))
                if evidence_path:
                    st.image(str(evidence_path), caption=f"Original source frame · {seconds:.2f} s", use_container_width=True)
                if st.button(f"Play {seconds:.2f} s", key=f"memory-seek-{memory['item_id']}-{seconds}"):
                    st.session_state["video_memory_seek_time"] = float(seconds)
                    st.rerun()

    st.divider()
    st.subheader("Evidence-backed moments")
    records = VIDEO_CACHE.list_analyses(video_hash)
    visible_records = [record for record in records if record.get("route") in {"timelens", "visibility_recheck", "endpoint_visibility", "event_recheck", "visibility_summary", "event_recheck_summary"}]
    if not visible_records:
        st.info("No grounded moments have been found for this clip yet. Run a question in Ask; Explore will show only saved model output and source evidence.")
    for record in visible_records:
        cached_result = record.get("result", {})
        route = record.get("route", "analysis")
        question_text = record.get("query", "")
        record_key = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        with st.expander(f"{route.replace('_', ' ').title()} · {question_text[:90]}"):
            raw = str(cached_result.get("raw_response", ""))
            if route == "timelens":
                intervals = _parse_intervals(raw, metadata["duration"])
                if intervals:
                    for start, end in intervals:
                        st.write(f"Candidate interval: {start:.2f}–{end:.2f} s")
                        if st.button(f"Play from {start:.2f} s", key=f"explore-tl-{record_key}-{start}"):
                            st.session_state["video_memory_seek_time"] = start
                            st.rerun()
                else:
                    st.caption("TimeLens2 returned no interval. This is a model result, not proof that the event did not occur.")
            elif route == "visibility_recheck":
                observations = extract_visibility_observations(raw)
                expected_times = record.get("config", {}).get("times", [])
                summary = resolve_visibility_observations(observations, expected_times=expected_times)
                st.json(summary)
                if summary.get("last_seen_seconds") is not None:
                    if st.button(f"Play from last confirmed view · {summary['last_seen_seconds']:.2f} s", key=f"explore-vis-{record_key}"):
                        st.session_state["video_memory_seek_time"] = summary["last_seen_seconds"]
                        st.rerun()
            elif route in {"visibility_summary", "event_recheck_summary"}:
                st.json(cached_result)
            elif route == "endpoint_visibility":
                st.json(_parse_json_object(raw) or {"raw_response": raw})
            elif route == "event_recheck":
                st.json(_parse_json_object(raw) or {"raw_response": raw})
            else:
                st.json(cached_result)

with ask_tab:
    st.subheader("Ask about this video")
    question = st.text_area(
        "Question",
        placeholder="Where did I leave my backpack? · When was the cup moved? · What happened after the person entered?",
        height=100,
        key="video_memory_question",
    )
    with st.expander("Analysis detail"):
        preferred_fps = st.slider(
            "Full-video sampling rate", min_value=0.25, max_value=2.0, value=1.0, step=0.25,
            help="Higher rates can catch short events but take longer on CPU. Boundary rechecks use source frames around model-proposed times.",
        )
    submitted = st.button("Analyze question", type="primary", use_container_width=True)

    if submitted and question.strip():
        cache_context = MEMORY_STORE.answer_cache_context(
            video_hash, verification_config_fingerprint=VERIFICATION_CONFIG_FINGERPRINT
        )
        saved = MEMORY_STORE.get_saved_answer(video_hash, question, cache_context=cache_context)
        if saved is not None and not saved_answer_requires_visual_recheck(question, saved):
            st.session_state["video_memory_result"] = {
                **saved,
                "question": question,
                "raw_direct": None,
                "raw_temporal": None,
                "raw_recheck": None,
                "raw_endpoint": None,
                "grounding_question": None,
                "visibility_resolution": None,
                "event_resolution": None,
                "cache_used": True,
                "saved_answer_cache_hit": True,
                "video_hash": video_hash,
                "video_path": str(video_path),
            }
            submitted = False
        else:
            memory_matches = (
                MEMORY_STORE.list_memories(video_hash, model_fingerprint=MEMORY_FINGERPRINT)
                if is_object_inventory_question(question) or is_event_summary_question(question)
                else MEMORY_STORE.search_memories(
                    video_hash, question, model_fingerprint=MEMORY_FINGERPRINT
                )
            )
            memory_answer = answer_from_memory(question, memory_matches)
            if memory_answer["status"] in {"supported", "memory_report"}:
                memory_result = {
                    "question": question,
                    "answer": memory_answer["answer"],
                    "abstain": False,
                    "confidence": memory_answer["confidence"],
                    "uncertainty": memory_answer["uncertainty"],
                    "verification_status": memory_answer.get("verification_status", "unverified"),
                    "input_mode": "persistent visual memory",
                    "evidence_seconds": memory_answer["evidence_seconds"],
                    "intervals": [],
                    "raw_direct": None,
                    "raw_temporal": None,
                    "raw_recheck": None,
                    "raw_endpoint": None,
                    "grounding_question": None,
                    "visibility_resolution": None,
                    "event_resolution": None,
                    "cache_used": True,
                    "video_hash": video_hash,
                    "video_path": str(video_path),
                    "memory_match_count": len(memory_matches),
                }
                st.session_state["video_memory_result"] = memory_result
                MEMORY_STORE.save_answer(
                    video_hash, question,
                    {key: value for key, value in memory_result.items() if key not in {
                        "raw_direct", "raw_temporal", "raw_recheck", "raw_endpoint", "video_path", "technical_error"
                    }},
                    provenance={
                        "route": "persistent_memory",
                        "memory_item_id": memory_answer.get("memory_item_id"),
                        "memory_item_ids": memory_answer.get("memory_item_ids", []),
                    },
                    cache_context=cache_context,
                )
                submitted = False

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

        route = route_question(question)
        direct_result: dict[str, Any] | None = None
        temporal_result: dict[str, Any] | None = None
        recheck_result: dict[str, Any] | None = None
        endpoint_result: dict[str, Any] | None = None
        grounding_question: str | None = None
        evidence_times: list[float] = []
        intervals: list[list[float]] = []
        answer = ""
        abstain = True
        confidence = "unknown"
        uncertainty = "No source-frame evidence was confirmed."
        input_mode = "full video"
        event_resolution: dict[str, Any] | None = None
        visibility_resolution: dict[str, Any] | None = None
        verification_ran = False
        verification_error_details: str | None = None
        fps_qwen = _model_fps(metadata["duration"], "qwen", preferred_fps)
        fps_time = _model_fps(metadata["duration"], "timelens", preferred_fps)

        try:
            if route in {"last_seen", "qwen_final_frame"} and visibility_target(question):
                target = visibility_target(question) or ""
                grounding_question = _visibility_query(question) or question
                try:
                    with st.spinner("Checking saved sample frames with the local target detector…"):
                        review_result = _review_target_memory_frames(
                            store=MEMORY_STORE, video_hash=video_hash, video_path=video_path,
                            metadata=metadata, target=target, memories=memory_items,
                            focus_end=route == "qwen_final_frame",
                        )
                    recheck_result = review_result
                    verification_ran = bool(review_result["frame_reviews"])
                    input_mode = "target detector on sampled source frames"
                    reviewed = review_result["frame_reviews"]
                    evidence_times = [
                        float(frame["visual_evidence"][0]["timestamp_seconds"])
                        for frame in reviewed
                    ]
                    accepted = review_result["supported_candidates"]
                    if accepted:
                        latest = max(accepted, key=lambda item: item["review"]["visual_evidence"][0]["timestamp_seconds"])
                        candidate = latest["candidate"]
                        frame_record = latest["review"]["visual_evidence"][0]
                        x1, y1, x2, y2 = candidate["bbox_xyxy"]
                        center_x = (x1 + x2) / (2 * metadata["width"])
                        center_y = (y1 + y2) / (2 * metadata["height"])
                        horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
                        vertical = "upper" if center_y < 1 / 3 else "lower" if center_y > 2 / 3 else "middle"
                        region = (
                            f"{horizontal} side" if (y2 - y1) / metadata["height"] > 0.6
                            else f"{vertical}-{horizontal} area"
                        )
                        latest_time = float(frame_record["timestamp_seconds"])
                        later_unknowns = [
                            float(frame["visual_evidence"][0]["timestamp_seconds"])
                            for frame in reviewed
                            if float(frame["visual_evidence"][0]["timestamp_seconds"]) > latest_time
                            and frame["visibility_state"] != "present"
                        ]
                        answer = (
                            f"A source-frame crop review supports a {target} candidate in the {region} "
                            f"of the frame at about {latest_time:.2f} s. This is the latest sampled frame with a "
                            "supported candidate, not an exact disappearance time or a real-world location."
                        )
                        if latest["review"].get("identity_status") == "associated":
                            answer = (
                                f"A targeted crop comparison supports the same {target} in the {region} "
                                f"of the frame at about {latest_time:.2f} s. This is the latest sampled frame "
                                "with a supported candidate, not an exact disappearance time or a real-world location."
                            )
                        if later_unknowns:
                            answer += (
                                f" Later sample checks at {', '.join(f'{value:.2f} s' for value in later_unknowns[:3])} "
                                "were inconclusive; the detector's misses do not establish that the object was absent."
                            )
                            uncertainty = "Later visibility is UNKNOWN; this reports a frame-local candidate, not continuous visibility."
                        else:
                            uncertainty = "Sparse sampling cannot establish the exact last-seen time or continuity between frames."
                        abstain = False
                        confidence = "not calibrated"
                        evidence_times = sorted({latest_time, *later_unknowns[:2]})
                        visibility_resolution = {
                            "status": "sampled_candidate_with_unknown_later_visibility" if later_unknowns else "sampled_candidate",
                            "last_supported_sample_seconds": latest_time,
                            "later_unknown_seconds": later_unknowns,
                            "continuous_visibility": False,
                            "identity_status": (
                                "associated to the earlier reference crop for this sighting"
                                if latest["review"].get("identity_status") == "associated"
                                else "unverified across frames"
                            ),
                        }
                    else:
                        answer = ""
                        abstain = True
                        confidence = "not calibrated"
                        uncertainty = (
                            "No localized detector candidate was accepted in the checked source frames. "
                            "This does not show that the target was absent; the saved semantic claims remain unverified."
                        )
                        visibility_resolution = {
                            "status": "unknown",
                            "unknown_seconds": evidence_times,
                            "absence_established": False,
                        }
                except Exception as exc:
                    answer = ""
                    abstain = True
                    confidence = "unverified"
                    verification_error_details = f"{type(exc).__name__}: {exc}"
                    uncertainty = (
                        "The targeted local visual check could not be completed. The saved semantic claim "
                        "remains unverified, so this answer is withheld."
                    )
                    evidence_times = sorted({
                        float(value) for item in memory_items if item["kind"] == "entity_observation"
                        for value in item.get("evidence_seconds", [])
                    })[:8]
                    input_mode = "target detector unavailable"

            elif route == "qwen_final_frame":
                from PIL import Image

                requested_times = sorted({max(0.0, metadata["duration"] - offset) for offset in (2.0, 1.0, 1.0 / metadata["fps"])})
                final_frames: list[tuple[Any, float]] = []
                for requested in requested_times:
                    source = _source_frame(video_path, requested, video_hash, metadata)
                    if source is not None:
                        frame, actual = source
                        if not final_frames or actual != final_frames[-1][1]:
                            final_frames.append((Image.fromarray(frame), actual))
                direct_result = _cached_analysis(
                    video_hash, "qwen_final_frame", question, QWEN_FINGERPRINT,
                    {"times": [round(ts, 3) for _, ts in final_frames], "schema": CACHE_SCHEMA},
                    lambda: ask_frames(final_frames, question),
                )
                parsed = _parsed_response(direct_result) or {}
                answer = str(parsed.get("answer", "")).strip()
                location, object_state = str(parsed.get("location", "")).strip(), str(parsed.get("object_state", "")).strip()
                if not bool(parsed.get("abstain", True)) and location and object_state:
                    answer = f"{location.rstrip('.; ')}, {object_state.rstrip('.; ')}."
                    abstain = False
                    confidence = str(parsed.get("confidence", "unknown")).lower()
                uncertainty = str(parsed.get("uncertainty", "")).strip()
                evidence_times = [ts for _, ts in final_frames]
                input_mode = "final source frames"

            else:
                temporal_route = route == "when"
                if temporal_route:
                    with st.spinner("Locating the event in time…"):
                        temporal_result = _cached_analysis(
                            video_hash, "timelens", question, TIMELENS_FINGERPRINT,
                            {"fps": fps_time, "max_new_tokens": 192, "schema": CACHE_SCHEMA},
                            lambda: ask_video(video_path, question, route="timelens", fps=fps_time, max_new_tokens=192),
                        )
                    intervals = _parse_intervals(temporal_result.get("raw_response", ""), metadata["duration"])
                with st.spinner("Reading the clip with Qwen3.5…"):
                    direct_result = _cached_analysis(
                        video_hash, "qwen", question, QWEN_FINGERPRINT,
                        {"fps": fps_qwen, "max_new_tokens": 320, "schema": CACHE_SCHEMA},
                        lambda: ask_video(video_path, question, route="qwen", fps=fps_qwen, max_new_tokens=320),
                    )
                parsed = _parsed_response(direct_result)
                answer = str(parsed.get("answer", "")).strip() if parsed else str(direct_result.get("raw_response", ""))
                abstain = bool(parsed.get("abstain", False)) if parsed else True
                confidence = str(parsed.get("confidence", "unknown")).lower() if parsed else "unknown"
                uncertainty = str(parsed.get("uncertainty", "")).strip() if parsed else "The model response was not structured; treat it as unverified."
                evidence_times = _extract_evidence_times(parsed, metadata["duration"])

                motion_question = any(term in question.lower() for term in ("move", "moved", "moving", "pick up", "picked up", "put down", "拿起", "移动", "放下"))
                claim = classify_event_claim(answer, abstain=abstain) if motion_question else "uncertain"
                # A positive movement answer needs a targeted source-frame review even when
                # Qwen and TimeLens2 agree; agreement between two model outputs is not visual proof.
                time_conflict = motion_question and temporal_route
                if time_conflict:
                    probe_times = build_event_probe_times(
                        metadata["duration"], intervals, evidence_times, fps=metadata["fps"], max_frames=6
                    )
                    from PIL import Image

                    event_frames: list[tuple[Any, float]] = []
                    for requested in probe_times:
                        source = _source_frame(video_path, requested, video_hash, metadata)
                        if source is not None:
                            frame, actual = source
                            if not event_frames or actual != event_frames[-1][1]:
                                event_frames.append((Image.fromarray(frame), actual))
                    with st.spinner("Qwen and TimeLens2 disagree; checking target motion against the scene in source frames…"):
                        recheck_result = _cached_analysis(
                            video_hash, "event_recheck", question, QWEN_FINGERPRINT,
                            {"times": [round(ts, 3) for _, ts in event_frames], "schema": CACHE_SCHEMA},
                            lambda: ask_event_recheck_frames(event_frames, question),
                        )
                    recheck_json = _parse_json_object(recheck_result.get("raw_response", "")) or {}
                    event_resolution = resolve_event_disagreement(
                        qwen_claim=claim, has_temporal_candidate=bool(intervals), recheck=recheck_json
                    )
                    VIDEO_CACHE.put_analysis(
                        video_hash, "event_recheck_summary", question, QWEN_FINGERPRINT,
                        event_resolution, {"schema": CACHE_SCHEMA},
                    )
                    if event_resolution["status"] == "resolved_stationary":
                        camera_note = " The camera moved while the target stayed in the same place relative to the scene." if event_resolution["camera_motion"] else ""
                        answer = f"No target movement is supported in the checked frames; it appears stationary relative to nearby scene details.{camera_note}"
                        abstain = False
                        confidence = str(recheck_json.get("confidence", "medium")).lower()
                        uncertainty = "This correction is based on a small set of targeted frames."
                        evidence_times = event_resolution["evidence_seconds"]
                    elif event_resolution["status"] == "resolved_moved":
                        abstain = False
                        confidence = str(recheck_json.get("confidence", "medium")).lower()
                        evidence_times = event_resolution["evidence_seconds"]
                        uncertainty = "Targeted frame review supports relative object movement; actor attribution is not inferred."
                    else:
                        answer = ""
                        abstain = True
                        uncertainty = "The models disagreed and the targeted source-frame check could not distinguish object movement from camera motion."
                        evidence_times = [ts for _, ts in event_frames]

                if not evidence_times and intervals:
                    evidence_times = sorted({round(ts, 2) for start, end in intervals for ts in (start, (start + end) / 2, end)})[:8]

            st.session_state["video_memory_result"] = {
                "question": question,
                "answer": answer,
                "abstain": abstain,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "input_mode": input_mode,
                "evidence_seconds": sorted({round(float(ts), 3) for ts in evidence_times}),
                "intervals": intervals,
                "raw_direct": direct_result,
                "raw_temporal": temporal_result,
                "raw_recheck": recheck_result,
                "raw_endpoint": endpoint_result,
                "grounding_question": grounding_question,
                "visibility_resolution": visibility_resolution,
                "event_resolution": event_resolution,
                "technical_error": verification_error_details,
                "cache_used": any(result.get("cache_hit") for result in (direct_result, temporal_result, recheck_result) if result),
                "video_hash": video_hash,
                "video_path": str(video_path),
            }
            saved_result = st.session_state["video_memory_result"]
            MEMORY_STORE.save_answer(
                video_hash, question,
                {key: value for key, value in saved_result.items() if key not in {
                    "raw_direct", "raw_temporal", "raw_recheck", "raw_endpoint", "video_path", "technical_error"
                }},
                provenance={
                    "route": "video_qa",
                    "direct_model": (direct_result or {}).get("model"),
                    "temporal_model": (temporal_result or {}).get("model"),
                },
                cache_context=MEMORY_STORE.answer_cache_context(
                    video_hash, verification_config_fingerprint=VERIFICATION_CONFIG_FINGERPRINT
                ),
            )
            _persist_evidence_frames(MEMORY_STORE, video_hash, video_path, metadata, saved_result["evidence_seconds"])
            if verification_ran:
                st.rerun()
        except Exception as exc:
            st.error("The local analysis could not finish. Your saved video and existing memories are still available; try again.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
            st.info("If the problem repeats, check that the pinned local models are installed and enough memory is available.")

    result = st.session_state.get("video_memory_result")
    if result and result.get("video_hash") == video_hash:
        st.divider()
        st.subheader("Answer")
        st.caption(f"Question: {result['question']}")
        if result.get("technical_error"):
            with st.expander("Technical details"):
                st.code(result["technical_error"])
        if result["abstain"] or not result["answer"]:
            st.warning("The available source frames do not support a reliable answer. This can mean the target is occluded, outside the view, or too ambiguous to identify.")
        else:
            st.markdown(result["answer"])
        status_col, confidence_col, cache_col = st.columns(3)
        if result["abstain"]:
            answer_status = "Abstained"
        elif result.get("verification_status") == "unverified":
            answer_status = "Memory-reported · Unverified"
        elif result.get("input_mode") == "target detector on sampled source frames":
            answer_status = (
                "Reference-matched sample"
                if str((result.get("visibility_resolution") or {}).get("identity_status", "")).startswith("associated")
                else "Frame-local candidate"
            )
        else:
            answer_status = "Evidence-linked"
        status_col.metric("Answer status", answer_status)
        confidence_col.metric("Model confidence · uncalibrated", str(result["confidence"]).capitalize())
        if result.get("input_mode") == "persistent visual memory":
            st.caption("This answer reuses stored model-generated memory. Its verification state is shown above; confidence is not a calibrated probability.")
            reuse_label = "Saved answer" if result.get("saved_answer_cache_hit") else "Memory retrieval"
        else:
            reuse_label = "Saved result" if result.get("cache_used") else "Fresh inference"
        cache_col.metric("Answer path", reuse_label)
        if result["uncertainty"]:
            st.caption(f"Uncertainty: {result['uncertainty']}")
        if result["intervals"]:
            st.markdown("**Time candidates**")
            for start, end in result["intervals"]:
                st.write(f"{start:.2f}–{end:.2f} s")
        if result["evidence_seconds"]:
            st.markdown("**Source evidence · click to seek**")
            cols = st.columns(min(4, len(result["evidence_seconds"])))
            for idx, seconds in enumerate(result["evidence_seconds"]):
                source = _source_frame(video_path, seconds, video_hash, metadata)
                with cols[idx % len(cols)]:
                    if source is not None:
                        frame, actual_seconds = source
                        st.caption(f"Source frame · {actual_seconds:.2f} s")
                        st.image(frame, use_container_width=True)
                        if st.button(f"Play from {actual_seconds:.2f} s", key=f"answer-seek-{idx}-{actual_seconds}"):
                            st.session_state["video_memory_seek_time"] = actual_seconds
                            st.rerun()
                    else:
                        st.warning(f"Could not decode source frame at {seconds:.2f} s.")
        with st.expander("Model and latency details"):
            st.json({
                "input_mode": result["input_mode"],
                "direct_analysis": result["raw_direct"],
                "temporal_grounding": result["raw_temporal"],
                "targeted_recheck": result["raw_recheck"],
                "end_of_video_check": result["raw_endpoint"],
            })
