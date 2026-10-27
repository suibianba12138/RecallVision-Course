"""Conservative, frame-local interpretation of prompt-conditioned detections."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


def _as_record(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "as_record"):
        return candidate.as_record()
    if isinstance(candidate, dict):
        return dict(candidate)
    raise TypeError("detector candidates must be records or mappings")


def review_detector_frame(
    detections: Iterable[Any], *, width: int, height: int, frame_index: int,
    timestamp_seconds: float, frame_sha256: str, minimum_score: float = 0.45,
    maximum_box_area_fraction: float = 0.35,
) -> dict[str, Any]:
    """Record one sampled frame without treating misses as negative visibility.

    A single plausible box corroborates only a frame-local candidate presence.
    It deliberately leaves persistent entity identity unverified.
    """
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if frame_index < 0 or not math.isfinite(float(timestamp_seconds)) or timestamp_seconds < 0:
        raise ValueError("frame index and timestamp must be finite and nonnegative")
    if len(frame_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in frame_sha256):
        raise ValueError("frame_sha256 must be a 64-character hex digest")
    if not 0.0 <= minimum_score <= 1.0 or not 0.0 < maximum_box_area_fraction < 1.0:
        raise ValueError("detector review thresholds are invalid")

    candidates: list[dict[str, Any]] = []
    accepted: list[int] = []
    rejected_reasons: list[str] = []
    for raw_candidate in detections:
        candidate = _as_record(raw_candidate)
        raw_box = candidate.get("bbox_xyxy")
        score = candidate.get("score")
        label = str(candidate.get("raw_model_label", candidate.get("label", "")))
        entry: dict[str, Any] = {
            "bbox_xyxy": list(raw_box) if isinstance(raw_box, (list, tuple)) else None,
            "score": score,
            "raw_model_label": label,
            "candidate_accepted": False,
        }
        if candidate.get("candidate_id") is not None:
            entry["candidate_id"] = str(candidate["candidate_id"])
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            numeric_score = float(score)
            values_valid = all(math.isfinite(value) for value in (x1, y1, x2, y2, numeric_score))
        except (TypeError, ValueError):
            values_valid = False
            x1 = y1 = x2 = y2 = numeric_score = 0.0
        if not values_valid or x2 <= x1 or y2 <= y1:
            entry["rejection_reason"] = "invalid box or score"
            rejected_reasons.append("invalid box")
        elif numeric_score > 1.0 or numeric_score < 0.0:
            entry["rejection_reason"] = "box or score is outside the source frame bounds"
            rejected_reasons.append("out-of-frame candidate")
        else:
            # Some post-processors emit sub-pixel negative/overflow coordinates.
            # Clip only tiny numerical drift; a substantially out-of-frame box is invalid.
            drift = max(-x1, -y1, x2 - width, y2 - height, 0.0)
            if drift > 2.0:
                entry["rejection_reason"] = "box or score is outside the source frame bounds"
                rejected_reasons.append("out-of-frame candidate")
                candidates.append(entry)
                continue
            clipped = [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
                       max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))]
            if clipped != [x1, y1, x2, y2]:
                entry["bbox_xyxy_unclipped"] = entry["bbox_xyxy"]
                entry["bbox_xyxy"] = clipped
            x1, y1, x2, y2 = clipped
            area_fraction = ((x2 - x1) * (y2 - y1)) / float(width * height)
            if area_fraction <= 0:
                entry["rejection_reason"] = "invalid box or score"
                rejected_reasons.append("invalid box")
                candidates.append(entry)
                continue
            entry["area_fraction"] = round(area_fraction, 6)
            if area_fraction > maximum_box_area_fraction or (x2 - x1) / width > 0.92 or (y2 - y1) / height > 0.96:
                entry["rejection_reason"] = "implausibly broad detector box"
                rejected_reasons.append("implausibly broad box")
            elif numeric_score < minimum_score:
                entry["rejection_reason"] = "below the configured review threshold"
                rejected_reasons.append("below threshold")
            else:
                entry["candidate_accepted"] = True
                accepted.append(len(candidates))
        candidates.append(entry)

    if len(accepted) == 1:
        verification_status = "corroborated"
        visibility_state = "present"
        identity_status = "unverified"
        reason = (
            "A localized detector candidate corroborates presence only in this exact frame; "
            "cross-frame entity identity remains unverified."
        )
    elif len(accepted) > 1:
        verification_status = "conflicted"
        visibility_state = "unknown"
        identity_status = "ambiguous"
        reason = "Multiple localized detector candidates make this frame ambiguous; entity identity is unresolved."
    else:
        verification_status = "unknown"
        visibility_state = "unknown"
        identity_status = "unverified"
        if "implausibly broad box" in rejected_reasons:
            reason = "The detector returned an implausibly broad candidate box; presence is not verified."
        else:
            reason = "No acceptable target candidate was found; a detector miss cannot establish absence."

    return {
        "verification_status": verification_status,
        "visibility_state": visibility_state,
        "identity_status": identity_status,
        "uncertainty_reason": reason,
        "visual_evidence": [{
            "frame_index": int(frame_index),
            "timestamp_seconds": round(float(timestamp_seconds), 3),
            "frame_sha256": frame_sha256.lower(),
            "detections": candidates,
        }],
    }


def combine_detector_crop_review(
    detector_review: dict[str, Any], crop_review: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require target appearance agreement before accepting any detector box.

    The crop model only reviews detector-proposed regions. Its answer cannot
    establish absence or identity continuity; disagreement therefore stays
    UNKNOWN/CONFLICTED rather than becoming a negative observation.
    """
    result = {
        **detector_review,
        "visual_evidence": [dict(record) for record in detector_review.get("visual_evidence", [])],
    }
    if not result["visual_evidence"]:
        return result
    frame = result["visual_evidence"][0]
    if crop_review is None:
        result.update(
            verification_status="unknown", visibility_state="unknown", identity_status="unverified",
            uncertainty_reason="The target crop was not reviewed; detector output remains a candidate only.",
        )
        return result
    crop = {
        "visibility": str(crop_review.get("visibility", "uncertain")).strip().lower(),
        "location": str(crop_review.get("location", "")).strip(),
        "confidence": str(crop_review.get("confidence", "unknown")).strip().lower(),
        "model_fingerprint": str(crop_review.get("model_fingerprint", "")),
    }
    frame["crop_review"] = crop
    candidates = frame.get("detections", [])
    if (
        crop["visibility"] == "present" and len(candidates) == 1
        and candidates[0].get("rejection_reason") != "box or score is outside the source frame bounds"
    ):
        result.update(
            verification_status="corroborated", visibility_state="present", identity_status="unverified",
            uncertainty_reason=(
                "The targeted crop review supports target presence in this exact source frame; "
                "the detector box may be coarse and identity continuity across frames is unverified."
            ),
        )
        candidates[0]["target_presence_corroborated"] = True
        return result
    if candidates:
        result.update(
            verification_status="conflicted", visibility_state="unknown", identity_status="ambiguous",
            uncertainty_reason=(
                "The detector candidate and targeted crop review did not agree on the target identity; "
                "this frame remains UNKNOWN and is not treated as absence."
            ),
        )
    else:
        result.update(
            verification_status="unknown", visibility_state="unknown", identity_status="unverified",
            uncertainty_reason="No detector candidate or crop review established visibility; absence is unknown.",
        )
    return result


_TARGET_LABEL_ALIASES = {
    "backpack": {"backpack", "rucksack", "knapsack"},
    "bag": {"bag", "backpack", "rucksack", "knapsack"},
    "cup": {"cup", "mug"},
    "mug": {"cup", "mug"},
    "bicycle": {"bicycle", "bike"},
    "bike": {"bicycle", "bike"},
}


def filter_target_candidates(
    detections: Iterable[Any], target_description: str, *, max_candidates: int = 4,
    duplicate_iou_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Keep target-labeled boxes and suppress nested duplicate proposals in one frame.

    Suppression is frame-local geometry only. Distinct boxes are retained so the crop
    verifier can reject lookalikes or leave a genuinely ambiguous frame unresolved.
    """
    if max_candidates <= 0 or not 0.0 <= duplicate_iou_threshold <= 1.0:
        raise ValueError("candidate filtering limits are invalid")
    target_words = re.findall(r"[\w]+", target_description.casefold())
    if not target_words:
        return []
    category = target_words[-1]
    accepted_labels = _TARGET_LABEL_ALIASES.get(category, {category})
    selected: list[dict[str, Any]] = []
    for raw in detections:
        candidate = _as_record(raw)
        label = str(candidate.get("raw_model_label", candidate.get("label", ""))).casefold()
        label_words = set(re.findall(r"[\w]+", label))
        if not (label_words & accepted_labels):
            continue
        box = candidate.get("bbox_xyxy")
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
            score = float(candidate.get("score"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)) or x2 <= x1 or y2 <= y1:
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in selected:
        x1, y1, x2, y2 = (float(value) for value in candidate["bbox_xyxy"])
        area = (x2 - x1) * (y2 - y1)
        duplicate = False
        for previous in kept:
            px1, py1, px2, py2 = (float(value) for value in previous["bbox_xyxy"])
            intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                0.0, min(y2, py2) - max(y1, py1)
            )
            previous_area = (px2 - px1) * (py2 - py1)
            union = area + previous_area - intersection
            if union > 0 and intersection / union >= duplicate_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    return kept


def combine_detector_crop_reviews(
    detector_review: dict[str, Any], crop_reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve frame-local candidate crops by ID without merging distinct objects."""
    result = {
        **detector_review,
        "visual_evidence": [dict(record) for record in detector_review.get("visual_evidence", [])],
    }
    if not result["visual_evidence"]:
        return result
    frame = result["visual_evidence"][0]
    candidates = frame.get("detections", [])
    if not candidates:
        result.update(
            verification_status="unknown", visibility_state="unknown", identity_status="unverified",
            uncertainty_reason="No localized target-labeled candidate was found; absence remains unknown.",
        )
        return result

    present: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        review = crop_reviews.get(candidate_id)
        if review is None:
            unresolved.append(candidate)
            continue
        crop = {
            "visibility": str(review.get("visibility", "uncertain")).strip().lower(),
            "location": str(review.get("location", "")).strip(),
            "confidence": str(review.get("confidence", "unknown")).strip().lower(),
            "model_fingerprint": str(review.get("model_fingerprint", "")),
            "candidate_id": candidate_id,
        }
        reference_match = str(review.get("reference_match", "")).strip().casefold()
        if reference_match in {"same", "match", "same object"}:
            crop["reference_match"] = "same"
            crop["visibility"] = "present"
        elif reference_match in {"different", "not same", "different object"}:
            crop["reference_match"] = "different"
            crop["visibility"] = "absent"
        elif reference_match in {"uncertain", "unknown"}:
            crop["reference_match"] = "uncertain"
            crop["visibility"] = "uncertain"
        candidate["crop_review"] = crop
        if crop["visibility"] == "present" and candidate.get("rejection_reason") != "box or score is outside the source frame bounds":
            present.append(candidate)
        elif crop["visibility"] != "absent":
            unresolved.append(candidate)

    if len(present) == 1 and not unresolved:
        present[0]["target_presence_corroborated"] = True
        for candidate in candidates:
            candidate.setdefault("target_presence_corroborated", False)
        identity_status = (
            "associated" if present[0].get("crop_review", {}).get("reference_match") == "same"
            else "unverified"
        )
        identity_reason = (
            "A unique candidate matched the previously supported reference crop; association is based on this targeted visual comparison."
            if identity_status == "associated" else
            "Exactly one target candidate crop supports presence at this source frame; cross-frame identity remains unverified."
        )
        result.update(
            verification_status="corroborated", visibility_state="present", identity_status=identity_status,
            uncertainty_reason=identity_reason,
        )
    else:
        for candidate in candidates:
            candidate.setdefault("target_presence_corroborated", False)
        multiple_possible = len(present) > 1 or bool(unresolved)
        result.update(
            verification_status="conflicted" if len(present) or multiple_possible else "conflicted",
            visibility_state="unknown",
            identity_status="ambiguous" if len(candidates) > 1 or multiple_possible else "unverified",
            uncertainty_reason=(
                "Multiple or unresolved target candidates prevent a reliable frame-local match; visibility remains UNKNOWN."
                if multiple_possible else
                "The crop review rejected the detector candidate; this does not establish target absence."
            ),
        )
    return result
