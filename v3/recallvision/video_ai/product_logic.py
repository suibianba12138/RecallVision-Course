from __future__ import annotations

import math
import json
import re
from typing import Any


def route_question(question: str) -> str:
    text = re.sub(r"\s+", " ", question.lower()).strip()
    last_seen = (
        r"\bwhere did i leave\b", r"\blast seen\b", r"\blast visible\b",
        r"\bwhere\b.*\blast\b",
        r"\bwhere .*\b(last seen|last visible)\b", r"最后看到", r"最后可见",
        r"\bwhere .* before it (?:disappeared|left the frame|went out of view)\b",
        r"最后放在哪", r"最后在哪里",
    )
    if any(re.search(pattern, text) for pattern in last_seen):
        return "last_seen"

    endpoint = ("at the end", "at end", "end of the clip", "end of video", "final frame", "last frame", "最后", "片尾", "视频结束")
    location = ("where", "location", "position", "位置", "哪里", "哪儿", "在哪")
    if any(term in text for term in endpoint) and any(term in text for term in location):
        return "qwen_final_frame"

    temporal = (
        r"\bwhen\b", r"\bwhat time\b", r"\bwhich moment\b", r"\bhow long\b",
        r"\b(before|after)\b", r"\b(move|moved|moving|pick(?:ed)? up|put down|disappear|leave|left the frame)\b",
        "什么时候", "何时", "多久", "之前", "之后", "移动", "拿起", "放下", "消失",
    )
    if any(re.search(term, text) if term.startswith(r"\b") else term in text for term in temporal):
        return "when"
    return "qwen"


def visibility_target(question: str) -> str | None:
    patterns = (
        r"\bwhere\s+did\s+i\s+(?:leave|last\s+see)\s+(?:my|the|a|an)?\s*(.+?)(?:[?.!]|$)",
        r"\bwhere\s+(?:was|is)\s+(?:my|the|a|an)?\s*(.+?)\s+last\s+(?:seen|visible)\b",
        r"\bwhere\s+(?:was|is)\s+(?:my|the|a|an)?\s*(.+?)\s+(?:last|latest|most recently)\b",
        r"\bwhere\s+(?:was|is)\s+(?:my|the|a|an)?\s*(.+?)\s+(?:before it disappeared|before it left)\b",
        r"\bwhere\s+(?:was|is)\s+(?:my|the|a|an)?\s*(.+?)\s+at the end(?: of (?:the )?(?:video|clip))?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            target = re.sub(r"\s+", " ", match.group(1)).strip(" .?!")
            if target:
                return target
    return None


def build_visibility_probe_times(
    *, duration: float, fps: float, intervals: list[list[float]], extra_times: list[float] | None = None, max_frames: int = 8
) -> list[float]:
    if not math.isfinite(duration) or duration <= 0 or max_frames <= 0:
        return []
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 1.0
    last_request = max(0.0, duration - 1.0 / safe_fps)
    valid_intervals = [pair for pair in intervals if isinstance(pair, (list, tuple)) and len(pair) == 2]
    if valid_intervals:
        end = max(max(0.0, min(duration, float(pair[1]))) for pair in valid_intervals)
        points = [end + offset for offset in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    else:
        points = []
    points.extend(extra_times or [])
    points.append(last_request)
    clean: list[float] = []
    for point in sorted(points):
        bounded = round(max(0.0, min(last_request, point)), 2)
        if not clean or bounded - clean[-1] >= max(0.5, 0.5 / safe_fps):
            clean.append(bounded)
    if len(clean) > max_frames:
        # Preserve both sides of the candidate boundary and retain the latest sample.
        picks = [round(i * (len(clean) - 1) / (max_frames - 1)) for i in range(max_frames)] if max_frames > 1 else [len(clean) - 1]
        clean = [clean[index] for index in picks]
    return clean


def build_adaptive_visibility_probe_times(
    reviews: list[dict[str, Any]], *, duration: float, fps: float, max_extra_frames: int = 3,
    large_gap_seconds: float = 1.0,
) -> list[float]:
    """Refine the first sparse unknown interval after the latest verified sighting.

    One midpoint checks the transition from the last supported sighting to the first
    inconclusive sample. Two earlier-biased interior probes then inspect a later sparse
    gap, which is useful when the object is leaving view and the latest samples are
    already negative-looking or ambiguous. Detector misses remain unknown and are never
    treated as negative anchors.
    """
    if (not math.isfinite(duration) or duration <= 0 or max_extra_frames <= 0
            or not math.isfinite(large_gap_seconds) or large_gap_seconds <= 0):
        return []
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 1.0
    last_request = max(0.0, duration - 1.0 / safe_fps)
    present_times: list[float] = []
    sample_times: list[tuple[float, str]] = []
    for review in reviews:
        try:
            frame = review.get("visual_evidence", [])[0]
            timestamp = float(frame["timestamp_seconds"])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or not 0.0 <= timestamp <= last_request:
            continue
        status = str(review.get("verification_status", "unknown")).casefold()
        visibility = str(review.get("visibility_state", "unknown")).casefold()
        sample_times.append((timestamp, visibility))
        if status in {"corroborated", "human_verified"} and visibility == "present":
            present_times.append(timestamp)
    if not present_times:
        return []

    latest_present = max(present_times)
    future_unknowns = sorted({
        timestamp for timestamp, visibility in sample_times
        if timestamp > latest_present + 0.5 / safe_fps and visibility in {"unknown", "occluded"}
    })
    if not future_unknowns:
        return []

    planned: list[float] = []
    first_unknown = future_unknowns[0]
    first_gap = first_unknown - latest_present
    if first_gap >= large_gap_seconds:
        planned.append((latest_present + first_unknown) / 2.0)
    if len(future_unknowns) >= 2:
        left, right = future_unknowns[0], future_unknowns[1]
        if right - left >= large_gap_seconds:
            span = right - left
            planned.extend((left + span / 5.0, left + 2.0 * span / 5.0))

    existing = [timestamp for timestamp, _ in sample_times]
    clean: list[float] = []
    minimum_separation = max(0.25, 0.5 / safe_fps)
    for candidate in planned:
        bounded = round(max(0.0, min(last_request, candidate)), 3)
        if any(abs(bounded - value) < minimum_separation for value in existing + clean):
            continue
        clean.append(bounded)
        if len(clean) >= max_extra_frames:
            break
    return sorted(clean)


def build_event_probe_times(
    duration: float, intervals: list[list[float]], evidence_times: list[float], *, fps: float = 1.0, max_frames: int = 6
) -> list[float]:
    if not math.isfinite(duration) or duration <= 0 or max_frames <= 0:
        return []
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 1.0
    last_request = max(0.0, duration - 1.0 / safe_fps)
    candidates: list[float] = []
    for pair in intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            start, end = sorted((float(pair[0]), float(pair[1])))
        except (TypeError, ValueError):
            continue
        start, end = max(0.0, start), min(last_request, end)
        span = max(0.0, end - start)
        candidates.extend(start + span * fraction for fraction in (0.0, 0.25, 0.5, 0.75, 1.0))
    candidates.extend(evidence_times)
    if not candidates:
        candidates = [0.0, duration / 2.0, last_request]
    elif len(candidates) == 1:
        candidates.extend([0.0, duration / 2.0, last_request])
    clean = sorted({round(max(0.0, min(last_request, float(value))), 2) for value in candidates if math.isfinite(float(value))})
    if len(clean) > max_frames:
        indices = [round(i * (len(clean) - 1) / (max_frames - 1)) for i in range(max_frames)] if max_frames > 1 else [len(clean) - 1]
        clean = [clean[index] for index in indices]
    return clean


def classify_event_claim(answer: str, *, abstain: bool = False) -> str:
    if abstain or not answer.strip():
        return "uncertain"
    text = re.sub(r"\s+", " ", answer.lower()).strip()
    negative = (
        r"\b(did not|didn't|does not|doesn't|never)\s+(?:appear to\s+)?(?:move|shift|change)\b",
        r"\b(no|not any)\s+(?:object\s+)?(?:movement|motion|displacement)\b",
        r"\b(stationary|remained in place|stayed in place|same relative position|did not change position)\b",
        "没有移动", "未发生移动", "保持静止", "位置没有变化",
    )
    if any(re.search(pattern, text) if pattern.startswith(r"\b") else pattern in text for pattern in negative):
        return "stationary"
    positive = (
        r"\b(moved|moves|moving|movement|displaced|shifted|pick(?:s|ed)? up|put down|carried away)\b",
        "移动了", "被拿起", "被放下", "发生位移",
    )
    return "moved" if any(re.search(pattern, text) if pattern.startswith(r"\b") else pattern in text for pattern in positive) else "uncertain"


def resolve_visibility_observations(
    observations: list[dict[str, Any]], *, expected_times: list[float] | None = None
) -> dict[str, Any]:
    normalized: dict[float, dict[str, Any]] = {}
    conflicts: set[float] = set()
    for item in observations:
        try:
            seconds = float(item.get("seconds"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(seconds) or seconds < 0:
            continue
        visibility = str(item.get("visibility", "uncertain")).strip().lower()
        if visibility in {"yes", "visible", "present", "found"}:
            visibility = "present"
        elif visibility in {"no", "not visible", "not_visible", "absent", "missing"}:
            visibility = "absent"
        else:
            visibility = "uncertain"
        key = round(seconds, 3)
        previous = normalized.get(key, {}).get("visibility")
        if previous in {"present", "absent"} and visibility in {"present", "absent"} and previous != visibility:
            conflicts.add(key)
        normalized[key] = {**item, "seconds": key, "visibility": visibility}

    ordered = sorted(normalized.values(), key=lambda item: item["seconds"])
    present = [item for item in ordered if item["visibility"] == "present" and str(item.get("confidence", "medium")).lower() != "low"]
    observed_times = {item["seconds"] for item in ordered}
    missing = sorted({round(float(value), 3) for value in (expected_times or []) if math.isfinite(float(value)) and round(float(value), 3) not in observed_times})
    if not present:
        return {
            "status": "unconfirmed",
            "last_seen_seconds": None,
            "last_seen_location": None,
            "last_seen_confidence": "unknown",
            "first_later_absent_seconds": None,
            "final_sample_visibility": ordered[-1]["visibility"] if ordered else "uncertain",
            "real_world_location": "unknown",
            "missing_seconds": missing,
            "conflict_seconds": sorted(conflicts),
        }
    last = present[-1]
    later_absent = next((item["seconds"] for item in ordered if item["seconds"] > last["seconds"] and item["visibility"] == "absent"), None)
    missing_after_last_present = any(seconds > last["seconds"] for seconds in missing)
    status = "incomplete" if missing_after_last_present else ("conflicted" if conflicts else "confirmed_last_seen")
    return {
        "status": status,
        "last_seen_seconds": last["seconds"],
        "last_seen_location": str(last.get("location", "")).strip() or None,
        "last_seen_confidence": str(last.get("confidence", "medium")).lower(),
        "first_later_absent_seconds": later_absent,
        "final_sample_visibility": ordered[-1]["visibility"] if ordered else "uncertain",
        "real_world_location": "unknown",
        "missing_seconds": missing,
        "conflict_seconds": sorted(conflicts),
    }


def extract_visibility_observations(raw: str) -> list[dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    value: Any = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for left, right in (("[", "]"), ("{", "}")):
            start, end = text.find(left), text.rfind(right)
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start:end + 1])
                    break
                except json.JSONDecodeError:
                    continue
    if isinstance(value, dict):
        if "observations" in value:
            value = value.get("observations", [])
        elif "seconds" in value and "visibility" in value:
            value = [value]
        else:
            keyed: list[dict[str, Any]] = []
            for timestamp, observation in value.items():
                if isinstance(observation, dict) and "visibility" in observation:
                    try:
                        float(timestamp)
                    except (TypeError, ValueError):
                        continue
                    keyed.append({**observation, "seconds": observation.get("seconds", timestamp)})
            value = keyed
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def resolve_event_disagreement(
    *, qwen_claim: str, has_temporal_candidate: bool, recheck: dict[str, Any] | None
) -> dict[str, Any]:
    del qwen_claim, has_temporal_candidate  # Empty grounding is a disagreement signal, not ground truth.
    recheck = recheck or {}
    status = str(recheck.get("event_status", "uncertain")).strip().lower().replace("-", "_").replace(" ", "_")
    confidence = str(recheck.get("confidence", "low")).strip().lower()
    raw_times = recheck.get("evidence_seconds", [])
    if isinstance(raw_times, str):
        try:
            raw_times = json.loads(raw_times)
        except json.JSONDecodeError:
            raw_times = []
    times: set[float] = set()
    if isinstance(raw_times, list):
        for value in raw_times:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0:
                times.add(round(number, 3))
    abstain_value = recheck.get("abstain", False)
    abstain = abstain_value if isinstance(abstain_value, bool) else str(abstain_value).strip().lower() in {"true", "yes", "1"}
    enough_evidence = len(times) >= 2 and confidence in {"medium", "high"} and not abstain
    camera_value = recheck.get("camera_motion")
    if camera_value is True or (isinstance(camera_value, str) and camera_value.strip().lower() in {"true", "yes"}):
        camera_motion: bool | None = True
    elif camera_value is False or (isinstance(camera_value, str) and camera_value.strip().lower() in {"false", "no"}):
        camera_motion = False
    else:
        camera_motion = None
    if enough_evidence and status in {"stationary", "no_movement", "no_move"}:
        resolved = "resolved_stationary"
    elif enough_evidence and status in {"moved", "movement", "object_moved"}:
        resolved = "resolved_moved"
    else:
        resolved = "unresolved"
    return {
        "status": resolved,
        "answer_supported": resolved != "unresolved",
        "camera_motion": camera_motion,
        "evidence_seconds": sorted(times),
        "event_status": status if resolved != "unresolved" else "uncertain",
    }
