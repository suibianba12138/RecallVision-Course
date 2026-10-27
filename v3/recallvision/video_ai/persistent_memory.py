from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


MEMORY_SCHEMA_VERSION = 2
MEMORY_GENERATION_VERSION = "structured-video-memory-v1"
VERIFICATION_STATUSES = {"unverified", "corroborated", "conflicted", "unknown", "human_verified"}
VISIBILITY_STATES = {"present", "absent", "occluded", "unknown"}
IDENTITY_STATUSES = {"unverified", "associated", "ambiguous", "human_verified"}


def build_memory_prompt() -> str:
    return """Build persistent visual memory from this ordered sample of timestamped source frames, using only visible evidence. Inspect every supplied timestamp, especially changes from beginning to end. This is a sparse sample, not a full frame-by-frame video inspection; do not infer what happens between supplied frames.
Return one minified JSON object on a single line, with no prose. It must have one array field named "items". Each record has fields: kind, entity_id, label, claim, time_seconds, start_seconds, end_seconds, evidence_seconds, confidence. Use kind values scene, entity_observation, event, or relation. For an entity_observation, include entity_id, label, claim, and its time_seconds; for the other kinds use label, claim, start_seconds, and end_seconds. Scene and relation records are optional if no clear evidence exists.
Use numeric times, an array of numeric evidence_seconds, and confidence as high, medium, or low. For each entity_observation, report only one exact supplied timestamp: time_seconds must be a scalar and evidence_seconds must contain only that same timestamp. Never spread one entity sighting across multiple frames or infer continuous visibility between sampled timestamps. Use only timestamps printed before the images; never use frame numbers, defaults, or made-up times. Include 3 to 4 distinctive entities when present, with at most 2 observations per entity. Prioritize bags, containers, devices, handled or moved objects, and other items a user may ask about. Avoid listing minor stationery unless it changes or matters to an event. Include at most 4 events, 2 scenes, and 2 relations, with event records first. For movement, give a before and after time and evidence from at least two supplied times. Use one entity_id across time only when continuity supports it. Do not merge similar objects because they share a category or location. Do not infer disappearance from temporary occlusion or leaving camera view. Do not infer who caused movement from proximity alone or infer personal identity. Low confidence is allowed; omit claims without direct evidence. Empty arrays are valid."""


def default_memory_root() -> Path:
    configured = os.environ.get("RECALLVISION_MEMORY_DIR")
    if configured:
        return Path(configured).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser() / "recallvision-v3"
    return Path.home() / ".local" / "share" / "recallvision-v3"


def _make_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _finite_seconds(value: Any, duration: float) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0 or seconds > duration:
        return None
    return round(seconds, 3)


def _evidence_times(record: dict[str, Any], duration: float) -> list[float]:
    values = record.get("evidence_seconds", [])
    if not isinstance(values, list):
        return []
    result = sorted({
        seconds
        for value in values
        if (seconds := _finite_seconds(value, duration)) is not None
    })
    return result[:8]


def _confidence(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return "medium" if math.isfinite(numeric) and numeric >= 0.7 else "low"
    confidence = str(value or "low").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else "low"


def _item(
    *, kind: str, label: Any, claim: Any, start: Any, end: Any,
    evidence: Any, confidence: Any, model_fingerprint: str, payload: dict[str, Any],
    duration: float,
) -> dict[str, Any] | None:
    label_text = re.sub(r"\s+", " ", str(label or "")).strip()
    claim_text = re.sub(r"\s+", " ", str(claim or "")).strip().rstrip(".")
    if not label_text or not claim_text:
        return None
    start_seconds = _finite_seconds(start, duration)
    end_seconds = _finite_seconds(end, duration)
    if start_seconds is None or end_seconds is None:
        return None
    if end_seconds < start_seconds:
        start_seconds, end_seconds = end_seconds, start_seconds
    valid_evidence = sorted({
        seconds
        for value in evidence if (seconds := _finite_seconds(value, duration)) is not None
    })
    if not valid_evidence:
        return None
    return {
        "kind": kind,
        "label": label_text,
        "claim": claim_text,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "evidence_seconds": valid_evidence[:8],
        "confidence": _confidence(confidence),
        "model_fingerprint": model_fingerprint,
        "payload": payload,
        "verification_status": "unverified",
        "visibility_state": "unknown",
        "identity_status": "unverified",
        "uncertainty_reason": "Model-generated semantic claim; no independent visual verification has been saved.",
        "verification_model_fingerprint": None,
        "verification_config_fingerprint": None,
        "visual_evidence": [],
    }


def normalize_memory_document(
    document: dict[str, Any], *, duration: float, model_fingerprint: str,
    sampled_timestamps: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Normalize model-generated scene/entity/event records and discard unsupported times."""
    if not math.isfinite(float(duration)) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    result: list[dict[str, Any]] = []
    sampled = None if sampled_timestamps is None else sorted({
        seconds for value in sampled_timestamps
        if (seconds := _finite_seconds(value, duration)) is not None
    })

    def align_evidence(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None or sampled is None:
            return item
        aligned = sorted({
            nearest
            for value in item["evidence_seconds"]
            for nearest in [min(sampled, key=lambda source_time: abs(source_time - value))]
            if abs(nearest - value) <= 0.15
        })
        if not aligned:
            return None
        item["evidence_seconds"] = aligned
        if item["kind"] == "entity_observation":
            observed = float(item["start_seconds"])
            timestamp = min(aligned, key=lambda source_time: abs(source_time - observed))
            item["start_seconds"] = timestamp
            item["end_seconds"] = timestamp
        return item

    flat_items = document.get("items")
    if isinstance(flat_items, list):
        for record in flat_items:
            if not isinstance(record, dict):
                continue
            kind = str(record.get("kind", "")).strip().lower()
            if kind == "relation" and record.get("entity_id") and record.get("time_seconds") is not None:
                kind = "entity_observation"
            if kind not in {"scene", "entity_observation", "event", "relation"}:
                continue
            evidence = _evidence_times(record, duration)

            # Some vision models return one object claim with several frame
            # timestamps (or mislabel it as a relation). Keep each sighting
            # tied to one exact sampled frame instead of implying continuous
            # visibility across the whole interval.
            raw_times = record.get("time_seconds")
            if kind == "entity_observation" and raw_times is not None:
                values = raw_times if isinstance(raw_times, (list, tuple)) else [raw_times]
                observation_times = sorted({
                    seconds for value in values
                    if (seconds := _finite_seconds(value, duration)) is not None
                })
                payload = {key: value for key, value in record.items() if key not in {
                    "kind", "label", "claim", "start_seconds", "end_seconds", "time_seconds",
                    "evidence_seconds", "confidence",
                }}
                for timestamp in observation_times:
                    item = align_evidence(_item(
                        kind="entity_observation", label=record.get("label"),
                        claim=record.get("claim"), start=timestamp, end=timestamp,
                        evidence=[timestamp], confidence=record.get("confidence"),
                        model_fingerprint=model_fingerprint, payload=payload, duration=duration,
                    ))
                    if item:
                        result.append(item)
                continue

            def time_value(value: Any, default: float | None, *, latest: bool = False) -> float | None:
                values = value if isinstance(value, (list, tuple)) else [value]
                valid = [seconds for item in values if (seconds := _finite_seconds(item, duration)) is not None]
                if valid:
                    return max(valid) if latest else min(valid)
                return default

            start = time_value(record.get("start_seconds", record.get("time_seconds")), min(evidence) if evidence else None)
            end = time_value(record.get("end_seconds", record.get("time_seconds")), max(evidence) if evidence else None, latest=True)
            if kind == "entity_observation" and record.get("time_seconds") is not None:
                start = end = time_value(record.get("time_seconds"), start, latest=True)
            item = align_evidence(_item(
                kind=kind, label=record.get("label"), claim=record.get("claim"),
                start=start, end=end, evidence=evidence, confidence=record.get("confidence"),
                model_fingerprint=model_fingerprint,
                payload={key: value for key, value in record.items() if key not in {
                    "kind", "label", "claim", "start_seconds", "end_seconds", "time_seconds",
                    "evidence_seconds", "confidence",
                }},
                duration=duration,
            ))
            if item:
                result.append(item)
        return result

    scenes = document.get("scenes", [])
    if isinstance(scenes, list):
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            evidence = _evidence_times(scene, duration)
            start = scene.get("start_seconds", min(evidence) if evidence else None)
            end = scene.get("end_seconds", max(evidence) if evidence else None)
            label = scene.get("label", "scene")
            item = align_evidence(_item(
                kind="scene", label=label, claim=scene.get("summary"), start=start, end=end,
                evidence=evidence, confidence=scene.get("confidence"),
                model_fingerprint=model_fingerprint, payload=scene, duration=duration,
            ))
            if item:
                result.append(item)

    entities = document.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            label = entity.get("description") or entity.get("label") or entity.get("category")
            observations = entity.get("observations", [])
            if not isinstance(observations, list) and isinstance(observations, dict):
                observations = [observations]
            if not isinstance(observations, list):
                observations = []
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                timestamp = observation.get("time_seconds", observation.get("timestamp_seconds"))
                evidence = _evidence_times(observation, duration)
                if not evidence and timestamp is not None:
                    valid_timestamp = _finite_seconds(timestamp, duration)
                    if valid_timestamp is not None:
                        evidence = [valid_timestamp]
                valid_timestamp = _finite_seconds(timestamp, duration)
                if valid_timestamp is None and evidence:
                    valid_timestamp = evidence[-1]
                location = str(observation.get("location") or "").strip()
                state = str(observation.get("state") or "").strip()
                claim = " ".join(part for part in (
                    str(label or "").strip(),
                    ("is " + state) if state else "",
                    location if location and location.lower() not in str(label).lower() else "",
                ) if part)
                payload = {**entity, "observation": observation}
                item = align_evidence(_item(
                    kind="entity_observation", label=label, claim=claim,
                    start=valid_timestamp, end=valid_timestamp, evidence=evidence,
                    confidence=observation.get("confidence", entity.get("confidence")),
                    model_fingerprint=model_fingerprint, payload=payload, duration=duration,
                ))
                if item:
                    result.append(item)

    events = document.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            evidence = _evidence_times(event, duration)
            label = event.get("event_type") or event.get("label") or "event"
            start = event.get("start_seconds", min(evidence) if evidence else None)
            end = event.get("end_seconds", max(evidence) if evidence else None)
            item = align_evidence(_item(
                kind="event", label=label, claim=event.get("summary"), start=start, end=end,
                evidence=evidence, confidence=event.get("confidence"),
                model_fingerprint=model_fingerprint, payload=event, duration=duration,
            ))
            if item:
                result.append(item)

    relations = document.get("relations", [])
    if isinstance(relations, list):
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            evidence = _evidence_times(relation, duration)
            start = relation.get("start_seconds", min(evidence) if evidence else None)
            end = relation.get("end_seconds", max(evidence) if evidence else None)
            label = relation.get("relation") or relation.get("label") or "temporal relation"
            item = align_evidence(_item(
                kind="relation", label=label, claim=relation.get("summary"), start=start, end=end,
                evidence=evidence, confidence=relation.get("confidence"),
                model_fingerprint=model_fingerprint, payload=relation, duration=duration,
            ))
            if item:
                result.append(item)
    return result


def parse_memory_response(raw_response: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw_response.strip(), flags=re.IGNORECASE)
    is_array = text.lstrip().startswith("[")
    left, right = (text.find("["), text.rfind("]")) if is_array else (text.find("{"), text.rfind("}"))
    if left < 0 or right <= left:
        raise ValueError("Memory model output did not contain a JSON object or array")
    try:
        value = json.loads(text[left:right + 1])
    except json.JSONDecodeError as exc:
        marker = re.search(r'"items"\s*:\s*\[', text)
        recovered: list[dict[str, Any]] = []
        if marker or is_array:
            decoder = json.JSONDecoder()
            cursor = marker.end() if marker else text.find("[") + 1
            while cursor < len(text):
                while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
                    cursor += 1
                if cursor >= len(text) or text[cursor] in "]}":
                    break
                try:
                    item, next_cursor = decoder.raw_decode(text, cursor)
                except json.JSONDecodeError:
                    break
                if not isinstance(item, dict):
                    break
                recovered.append(item)
                cursor = next_cursor
        if recovered:
            return {"items": recovered, "_partial": True}
        raise ValueError(f"Memory model output was not valid JSON: {exc.msg}") from exc
    if isinstance(value, list):
        value = {"items": value}
    if not isinstance(value, dict):
        raise ValueError("Memory model output must be a JSON object")
    if "items" in value:
        if not isinstance(value["items"], list):
            raise ValueError("Memory items must be a JSON array")
        return value
    expected = {"scenes", "entities", "events", "relations"}
    if not expected.issubset(value):
        raise ValueError("Memory model JSON is missing scenes, entities, events, or relations")
    if any(not isinstance(value[key], list) for key in expected):
        raise ValueError("Each memory collection must be a JSON array")
    return value


_STOP_WORDS = {
    "a", "an", "and", "are", "at", "did", "do", "does", "for", "from", "give",
    "happen", "happened", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "show", "tell", "that", "the", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with", "leave", "last",
    "visible", "move", "moved", "moving", "movement", "pick", "picked", "up", "put", "down",
    "before", "after", "end", "happen", "happens", "视频", "什么时候",
    "什么", "哪里", "哪个", "谁", "在", "了", "吗", "的", "我", "我的", "最后",
}


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    normalized: set[str] = set()
    for token in raw:
        if token in _STOP_WORDS or len(token) < 2:
            continue
        token = {
            "moved": "move", "moving": "move", "movement": "move", "moves": "move",
            "picked": "pick", "picking": "pick", "leaves": "leave", "left": "leave",
            "objects": "object", "items": "item",
        }.get(token, token)
        normalized.add(token)
    return normalized


def _query_key(question: str, cache_context: dict[str, Any] | None = None) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    context = json.dumps(cache_context, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if cache_context else ""
    return hashlib.sha256((normalized + "\0" + context).encode("utf-8")).hexdigest()


def _requires_precise_visibility(question: str) -> bool:
    last_seen = bool(re.search(
        r"\bwhere did (?:i|we) leave\b|\blast (?:seen|visible|observed)\b|"
        r"\bwhere\b.*\b(?:last|latest|most recently)\b|"
        r"\bwhere\b.*\bbefore it (?:disappeared|left the frame|went out of view)\b|最後看到|最后看到|最后可见|最后可視",
        question, re.I,
    ))
    endpoint = bool(re.search(
        r"\b(at the end|end of (?:the )?(?:clip|video)|final frame|last frame)\b|片尾|视频结束",
        question, re.I,
    ))
    return last_seen or endpoint


def saved_answer_requires_visual_recheck(question: str, answer: dict[str, Any]) -> bool:
    """Do not reuse a memory-only answer for queries that ask for precise visibility timing."""
    return answer.get("input_mode") == "persistent visual memory" and _requires_precise_visibility(question)


def is_object_inventory_question(question: str) -> bool:
    return bool(re.search(
        r"\bwhat\s+(?:objects?|items?)\b.*\b(?:appeared|shown|present|in the video|in this clip)\b|"
        r"\bwhat\s+does\s+(?:the\s+)?video\s+show\b|有哪些物体|视频里有什么物体|影片里有哪些物体",
        question, re.I,
    ))


def is_event_summary_question(question: str) -> bool:
    return bool(re.search(
        r"\bwhat happened(?: to| in| during)?\b(?!\s+(?:after|before))|"
        r"\bwhat occurred(?: in| during)?\b(?!\s+(?:after|before))|发生了什么|发生什么事",
        question, re.I,
    ))


def answer_from_memory(question: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a conservative answer only when one evidence-backed memory clearly matches."""
    if not matches:
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
    if is_object_inventory_question(question):
        observations = [item for item in matches if item.get("kind") == "entity_observation"]
        grouped: dict[str, dict[str, Any]] = {}
        for item in observations:
            payload = item.get("payload", {})
            entity_id = payload.get("entity_id") if isinstance(payload, dict) else None
            key = str(entity_id or " ".join(_tokens(str(item.get("claim", item.get("label", ""))))))
            if not key:
                continue
            entry = grouped.setdefault(key, {
                "claim": str(item.get("claim", "")).strip().rstrip("."),
                "label": str(item.get("label", "")).strip(),
                "times": set(),
                "item_ids": [],
            })
            entry["times"].update(float(value) for value in item.get("evidence_seconds", []))
            if item.get("item_id") is not None:
                entry["item_ids"].append(int(item["item_id"]
                ))
        if not grouped:
            return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
        descriptions: list[str] = []
        evidence: set[float] = set()
        item_ids: list[int] = []
        for entry in grouped.values():
            claim = entry["claim"] or entry["label"] or "unlabeled object candidate"
            times = sorted(entry["times"])
            evidence.update(times)
            item_ids.extend(entry["item_ids"])
            time_text = ", ".join(f"{value:.2f} s" for value in times[:4])
            descriptions.append(f"{claim} (sampled at {time_text})" if time_text else claim)
        return {
            "status": "memory_report", "answer": "The stored memory lists these unverified object candidates: "
            + "; ".join(descriptions) + ".", "abstain": False,
            "verification_status": "unverified", "confidence": "not calibrated",
            "uncertainty": "This is a model-generated object inventory, not an independently confirmed list. "
            "Some objects may be missing, mislabeled, or represented by only a sampled observation.",
            "evidence_seconds": sorted(evidence)[:24], "memory_item_ids": item_ids,
        }
    if is_event_summary_question(question):
        question_terms = _tokens(question)
        events = [item for item in matches if item.get("kind") in {"event", "relation"}]
        relevant = []
        for item in events:
            item_terms = _tokens(" ".join((str(item.get("label", "")), str(item.get("claim", "")))))
            if not question_terms or not (question_terms & item_terms):
                # A broad question about the video may have no object term; specific
                # target questions must overlap the event's stored description.
                if question_terms and question_terms != {"video"}:
                    continue
            relevant.append(item)
        if not relevant:
            return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
        answer_parts: list[str] = []
        evidence: set[float] = set()
        item_ids: list[int] = []
        for item in relevant[:8]:
            claim = str(item.get("claim", "")).strip().rstrip(".")
            times = sorted(float(value) for value in item.get("evidence_seconds", []))
            evidence.update(times)
            if item.get("item_id") is not None:
                item_ids.append(int(item["item_id"]))
            if claim:
                timestamp_text = ", ".join(f"{value:.2f} s" for value in times[:6])
                answer_parts.append(f"{claim} (sampled at {timestamp_text})" if timestamp_text else claim)
        if not answer_parts:
            return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
        statuses = {str(item.get("verification_status", "unverified")) for item in relevant}
        verification_status = statuses.pop() if len(statuses) == 1 else "conflicted"
        return {
            "status": "memory_report", "answer": "The stored event memory reports these candidates: "
            + "; ".join(answer_parts) + ".", "abstain": False,
            "verification_status": verification_status,
            "confidence": "not calibrated",
            "uncertainty": f"Event verification status is {verification_status.upper()}; descriptions are model-generated claims. "
            "The listed times are sampled evidence frames, not exact event boundaries. Check the source frames before treating the event as confirmed.",
            "evidence_seconds": sorted(evidence)[:24], "memory_item_ids": item_ids,
        }
    best = matches[0]
    try:
        score = float(best.get("retrieval_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score < 0.5 or best.get("confidence") not in {"high", "medium"} or not best.get("evidence_seconds"):
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}

    question_terms = _tokens(question)
    memory_terms = _tokens(" ".join((
        str(best.get("label", "")), str(best.get("claim", "")),
        json.dumps(best.get("payload", {}), ensure_ascii=False),
    )))
    if not question_terms or len(question_terms & memory_terms) / len(question_terms) < 0.67:
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}

    temporal = bool(re.search(r"\b(when|what time|before|after)\b|什么时候|之前|之后", question, re.I))
    location = bool(re.search(r"\b(where|location|last|leave)\b|哪里|在哪|最后", question, re.I))
    best_entity = best.get("payload", {}).get("entity_id")
    for candidate in matches[1:]:
        try:
            candidate_score = float(candidate.get("retrieval_score", 0.0))
        except (TypeError, ValueError):
            continue
        candidate_entity = candidate.get("payload", {}).get("entity_id")
        if candidate_score < score - 0.08:
            continue
        if best_entity and candidate_entity and best_entity != candidate_entity:
            return {
                "status": "ambiguous", "answer": "", "abstain": True,
                "uncertainty": "More than one stored entity matches the description.",
                "evidence_seconds": sorted(set(best.get("evidence_seconds", []) + candidate.get("evidence_seconds", []))),
            }

    if location:
        if (
            best.get("verification_status") not in {"corroborated", "human_verified"}
            or best.get("visibility_state") != "present"
        ):
            return {
                "status": "insufficient", "answer": "", "abstain": True,
                "uncertainty": "Unverified location claim. " + str(best.get("uncertainty_reason")
                or "No independent visual evidence confirms this location."),
                "evidence_seconds": list(best.get("evidence_seconds", [])),
                "memory_item_id": best.get("item_id"),
            }
        best_label = str(best.get("label", "")).casefold()
        same_entity = [candidate for candidate in matches if (
            candidate.get("kind") == "entity_observation"
            and (
                (best_entity and candidate.get("payload", {}).get("entity_id") == best_entity)
                or (not best_entity and str(candidate.get("label", "")).casefold() == best_label)
            )
        )]
        if same_entity:
            best = min(same_entity, key=lambda item: min(
                float(value) for value in item.get("evidence_seconds", [item["start_seconds"]])
            ))

    claim = str(best["claim"]).strip().rstrip(".")
    if not claim:
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
    if location and _requires_precise_visibility(question):
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
    if temporal and best["kind"] not in {"event", "relation"}:
        return {"status": "insufficient", "answer": "", "abstain": True, "evidence_seconds": []}
    if temporal and best["kind"] in {"event", "relation"}:
        if best.get("verification_status") not in {"corroborated", "human_verified"}:
            return {
                "status": "insufficient", "answer": "", "abstain": True,
                "uncertainty": best.get("uncertainty_reason")
                or "The event is a model-generated claim without independent visual verification.",
                "evidence_seconds": list(best.get("evidence_seconds", [])),
                "memory_item_id": best.get("item_id"),
            }
        start, end = float(best["start_seconds"]), float(best["end_seconds"])
        interval = f"{start:.2f} s" if abs(end - start) < 0.05 else f"{start:.2f}–{end:.2f} s"
        answer = f"The stored model memory reports: {claim} (around {interval})."
    elif location:
        when = min(float(value) for value in best["evidence_seconds"])
        capitalized = claim[0].upper() + claim[1:]
        answer = (
            f"The stored model memory reports: {capitalized} "
            f"(linked to a sampled source frame at about {when:.2f} s). "
            "This memory does not establish the exact last-seen time or its real-world location afterward."
        )
    else:
        answer = "The stored model memory reports: " + claim[0].lower() + claim[1:] + "."
    return {
        "status": "supported", "answer": answer, "abstain": False,
        "confidence": best["confidence"],
        "verification_status": best.get("verification_status", "unverified"),
        "visibility_state": best.get("visibility_state", "unknown"),
        "identity_status": best.get("identity_status", "unverified"),
        "uncertainty": (
            "This is a sampled model-generated sighting; the exact last-seen time and later real-world location are unknown."
            if location else
            "Answer derived from stored model-generated memory; verify the linked source frame."
        ),
        "evidence_seconds": [when] if location else list(best["evidence_seconds"]),
        "memory_item_id": best.get("item_id"), "retrieval_score": score,
    }


class PersistentMemoryStore:
    """Local SQLite store for durable, source-linked V3 visual memories."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_memory_root()
        self.database_path = self.root / "visual_memory.sqlite3"
        self.videos_root = self.root / "videos"
        self.evidence_root = self.root / "evidence"
        _make_private_directory(self.root)
        _make_private_directory(self.videos_root)
        _make_private_directory(self.evidence_root)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        for path in (self.database_path, Path(f"{self.database_path}-wal"), Path(f"{self.database_path}-shm")):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                pass
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO schema_info(singleton, version) VALUES (1, 1);
                CREATE TABLE IF NOT EXISTS videos (
                    video_hash TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                analysis_metadata_json TEXT NOT NULL DEFAULT '{}',
                    analysis_status TEXT NOT NULL DEFAULT 'pending',
                    analysis_error TEXT,
                    analysis_raw TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_items (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_hash TEXT NOT NULL REFERENCES videos(video_hash) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    evidence_seconds_json TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    verification_status TEXT NOT NULL DEFAULT 'unverified',
                    visibility_state TEXT NOT NULL DEFAULT 'unknown',
                    identity_status TEXT NOT NULL DEFAULT 'unverified',
                    uncertainty_reason TEXT NOT NULL DEFAULT 'No independent visual verification has been saved.',
                    verification_model_fingerprint TEXT,
                    verification_config_fingerprint TEXT,
                    visual_evidence_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS memory_items_video_time
                    ON memory_items(video_hash, start_seconds);
                CREATE TABLE IF NOT EXISTS evidence_frames (
                    video_hash TEXT NOT NULL REFERENCES videos(video_hash) ON DELETE CASCADE,
                    frame_index INTEGER NOT NULL,
                    time_seconds REAL NOT NULL,
                    path TEXT NOT NULL,
                    PRIMARY KEY(video_hash, frame_index)
                );
                CREATE TABLE IF NOT EXISTS saved_answers (
                    video_hash TEXT NOT NULL REFERENCES videos(video_hash) ON DELETE CASCADE,
                    query_key TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(video_hash, query_key)
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(videos)").fetchall()}
            if "analysis_metadata_json" not in columns:
                db.execute("ALTER TABLE videos ADD COLUMN analysis_metadata_json TEXT NOT NULL DEFAULT '{}'")
            version = int(db.execute("SELECT version FROM schema_info WHERE singleton=1").fetchone()[0])
            if version == 1:
                item_columns = {row[1] for row in db.execute("PRAGMA table_info(memory_items)").fetchall()}
                additions = {
                    "verification_status": "TEXT NOT NULL DEFAULT 'unverified'",
                    "visibility_state": "TEXT NOT NULL DEFAULT 'unknown'",
                    "identity_status": "TEXT NOT NULL DEFAULT 'unverified'",
                    "uncertainty_reason": "TEXT NOT NULL DEFAULT 'Legacy model claim; no independent visual verification was saved.'",
                    "verification_model_fingerprint": "TEXT",
                    "verification_config_fingerprint": "TEXT",
                    "visual_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
                }
                for name, declaration in additions.items():
                    if name not in item_columns:
                        db.execute(f"ALTER TABLE memory_items ADD COLUMN {name} {declaration}")
                legacy_rows = db.execute(
                    "SELECT * FROM memory_items WHERE kind='entity_observation' ORDER BY item_id"
                ).fetchall()
                for row in legacy_rows:
                    try:
                        evidence_times = sorted(set(float(value) for value in json.loads(row["evidence_seconds_json"])))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        evidence_times = []
                    if not evidence_times:
                        evidence_times = [float(row["start_seconds"])]
                    payload = json.loads(row["payload_json"] or "{}")
                    for index, timestamp in enumerate(evidence_times):
                        values = (
                            timestamp, timestamp, json.dumps([timestamp]),
                            "unverified", "unknown", "unverified",
                            "Legacy model-generated claim; this exact sampled frame has not been independently verified.",
                            None, None, "[]",
                        )
                        if index == 0:
                            db.execute(
                                """UPDATE memory_items SET start_seconds=?, end_seconds=?, evidence_seconds_json=?,
                                   verification_status=?, visibility_state=?, identity_status=?, uncertainty_reason=?,
                                   verification_model_fingerprint=?, verification_config_fingerprint=?, visual_evidence_json=?
                                   WHERE item_id=?""",
                                (*values, row["item_id"]),
                            )
                        else:
                            db.execute(
                                """INSERT INTO memory_items(video_hash,kind,label,claim,start_seconds,end_seconds,
                                   evidence_seconds_json,confidence,model_fingerprint,payload_json,verification_status,
                                   visibility_state,identity_status,uncertainty_reason,verification_model_fingerprint,
                                   verification_config_fingerprint,visual_evidence_json)
                                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (row["video_hash"], row["kind"], row["label"], row["claim"],
                                 timestamp, timestamp, json.dumps([timestamp]), row["confidence"],
                                 row["model_fingerprint"], json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                 *values[3:]),
                            )
                db.execute(
                    "UPDATE memory_items SET uncertainty_reason='Legacy model-generated claim; no independent visual verification was saved.' WHERE uncertainty_reason='No independent visual verification has been saved.'"
                )
                db.execute("UPDATE videos SET schema_version=?", (MEMORY_SCHEMA_VERSION,))
                db.execute("UPDATE schema_info SET version=? WHERE singleton=1", (MEMORY_SCHEMA_VERSION,))
                version = MEMORY_SCHEMA_VERSION
            if version != MEMORY_SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported visual memory database schema: {version}")

    @staticmethod
    def hash_file(source_path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(source_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def register_video(
        self, source_path: str | Path, *, source_name: str, metadata: dict[str, Any],
        model_fingerprint: str,
    ) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        video_hash = self.hash_file(source)
        suffix = Path(source_name or source.name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = source.suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,8}", source.suffix.lower()) else ".mp4"
        target = self.videos_root / f"{video_hash}{suffix}"
        if source != target.resolve():
            if not target.is_file():
                fd, temp_name = tempfile.mkstemp(prefix=f".{video_hash}.", suffix=".tmp", dir=self.videos_root)
                os.close(fd)
                try:
                    shutil.copyfile(source, temp_name)
                    if self.hash_file(temp_name) != video_hash:
                        raise IOError("Video changed while it was being copied into persistent memory")
                    os.replace(temp_name, target)
                finally:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
        now = time.time()
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._db() as db:
            existing = db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone()
            if existing is None:
                db.execute(
                    """INSERT INTO videos(video_hash, source_name, source_path, metadata_json,
                       model_fingerprint, schema_version, analysis_status, created_at, updated_at)
                       VALUES(?,?,?,?,?,?, 'pending', ?,?)""",
                    (video_hash, Path(source_name).name, str(target), metadata_json,
                     model_fingerprint, MEMORY_SCHEMA_VERSION, now, now),
                )
            else:
                stale = (
                    existing["model_fingerprint"] != model_fingerprint
                    or existing["metadata_json"] != metadata_json
                    or existing["schema_version"] != MEMORY_SCHEMA_VERSION
                )
                db.execute(
                    """UPDATE videos SET source_name=?, source_path=?, metadata_json=?,
                       model_fingerprint=?, schema_version=?, analysis_status=?, analysis_error=?, updated_at=?
                       WHERE video_hash=?""",
                    (Path(source_name).name, str(target), metadata_json, model_fingerprint, MEMORY_SCHEMA_VERSION,
                     "pending" if stale else existing["analysis_status"],
                     None if stale else existing["analysis_error"], now, video_hash),
                )
                if stale:
                    db.execute("DELETE FROM saved_answers WHERE video_hash=?", (video_hash,))
            return self._video_dict(db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone())

    @staticmethod
    def _video_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        value["analysis_metadata"] = json.loads(value.pop("analysis_metadata_json", "{}"))
        return value

    @staticmethod
    def _memory_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["evidence_seconds"] = json.loads(value.pop("evidence_seconds_json"))
        value["payload"] = json.loads(value.pop("payload_json"))
        value["visual_evidence"] = json.loads(value.pop("visual_evidence_json", "[]"))
        return value

    def get_video(self, video_hash: str) -> dict[str, Any] | None:
        with self._db() as db:
            return self._video_dict(db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone())

    def ensure_analysis_version(self, video_hash: str, model_fingerprint: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone()
            if row is None:
                return None
            stale = (
                row["model_fingerprint"] != model_fingerprint
                or row["schema_version"] != MEMORY_SCHEMA_VERSION
            )
            if stale:
                db.execute(
                    """UPDATE videos SET model_fingerprint=?, schema_version=?, analysis_status='pending',
                       analysis_error=NULL, updated_at=? WHERE video_hash=?""",
                    (model_fingerprint, MEMORY_SCHEMA_VERSION, time.time(), video_hash),
                )
                db.execute("DELETE FROM saved_answers WHERE video_hash=?", (video_hash,))
                row = db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone()
            return self._video_dict(row)

    def list_videos(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM videos ORDER BY updated_at DESC").fetchall()
        return [self._video_dict(row) for row in rows if row is not None]

    def replace_memories(
        self, video_hash: str, memories: Iterable[dict[str, Any]], *, raw_response: str,
        analysis_metadata: dict[str, Any] | None = None,
        complete: bool = True,
    ) -> None:
        items = list(memories)
        with self._db() as db:
            video = db.execute("SELECT * FROM videos WHERE video_hash=?", (video_hash,)).fetchone()
            if video is None:
                raise KeyError(f"Unknown video: {video_hash}")
            duration = float(json.loads(video["metadata_json"]).get("duration", 0.0))
            if duration <= 0:
                raise ValueError("Registered video metadata must contain a positive duration")
            for item in items:
                if item.get("kind") not in {"scene", "entity_observation", "event", "relation"}:
                    raise ValueError(f"Unsupported memory kind: {item.get('kind')}")
                if not item.get("label") or not item.get("claim"):
                    raise ValueError("Every memory needs a label and claim")
                if not isinstance(item.get("evidence_seconds"), list) or not item["evidence_seconds"]:
                    raise ValueError("Every memory needs at least one source evidence timestamp")
                times = [
                    _finite_seconds(value, duration)
                    for value in [item.get("start_seconds"), item.get("end_seconds"), *item["evidence_seconds"]]
                ]
                if any(value is None for value in times):
                    raise ValueError("Memory timestamps must fall inside the source video")

            if complete:
                db.execute("DELETE FROM memory_items WHERE video_hash=?", (video_hash,))
                db.execute("DELETE FROM saved_answers WHERE video_hash=?", (video_hash,))
                existing_keys: set[tuple[Any, ...]] = set()
            else:
                existing_keys = {
                    (
                        row["kind"], row["label"], row["claim"], row["start_seconds"], row["end_seconds"],
                        row["evidence_seconds_json"], row["confidence"], row["model_fingerprint"], row["payload_json"],
                    )
                    for row in db.execute(
                        "SELECT kind,label,claim,start_seconds,end_seconds,evidence_seconds_json,confidence,model_fingerprint,payload_json "
                        "FROM memory_items WHERE video_hash=?", (video_hash,)
                    ).fetchall()
                }
            for item in items:
                model = item.get("model_fingerprint", video["model_fingerprint"])
                evidence_json = json.dumps(item["evidence_seconds"])
                confidence = _confidence(item.get("confidence"))
                payload_json = json.dumps(item.get("payload", {}), ensure_ascii=False, sort_keys=True)
                memory_key = (
                    item["kind"], item["label"], item["claim"], float(item["start_seconds"]),
                    float(item["end_seconds"]), evidence_json, confidence, model, payload_json,
                )
                if memory_key in existing_keys:
                    continue
                db.execute(
                    """INSERT INTO memory_items(video_hash, kind, label, claim, start_seconds,
                       end_seconds, evidence_seconds_json, confidence, model_fingerprint, payload_json,
                       verification_status, visibility_state, identity_status, uncertainty_reason,
                       verification_model_fingerprint, verification_config_fingerprint, visual_evidence_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (video_hash, item["kind"], item["label"], item["claim"],
                     float(item["start_seconds"]), float(item["end_seconds"]),
                     evidence_json, confidence, model, payload_json,
                     item.get("verification_status", "unverified"),
                     item.get("visibility_state", "unknown"),
                     item.get("identity_status", "unverified"),
                     item.get("uncertainty_reason") or "Model-generated semantic claim; no independent visual verification has been saved.",
                     item.get("verification_model_fingerprint"),
                     item.get("verification_config_fingerprint"),
                     json.dumps(item.get("visual_evidence", []), ensure_ascii=False, sort_keys=True)),
                )
                existing_keys.add(memory_key)
            db.execute(
                """UPDATE videos SET analysis_status=?, analysis_error=NULL,
                   analysis_raw=?, analysis_metadata_json=?, updated_at=? WHERE video_hash=?""",
                ("complete" if complete else "partial", raw_response,
                 json.dumps(analysis_metadata or {}, ensure_ascii=False, sort_keys=True),
                 time.time(), video_hash),
            )

    def mark_analysis_failed(self, video_hash: str, error: str, *, raw_response: str | None = None) -> None:
        with self._db() as db:
            db.execute(
                """UPDATE videos SET analysis_status='failed', analysis_error=?, analysis_raw=COALESCE(?, analysis_raw),
                   updated_at=? WHERE video_hash=?""",
                (error[:2000], raw_response, time.time(), video_hash),
            )

    def list_memories(
        self, video_hash: str, *, kinds: set[str] | None = None,
        model_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["video_hash=?"]
        parameters: list[Any] = [video_hash]
        if model_fingerprint is not None:
            conditions.append("model_fingerprint=?")
            parameters.append(model_fingerprint)
        with self._db() as db:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                conditions.append(f"kind IN ({placeholders})")
                parameters.extend(sorted(kinds))
            rows = db.execute(
                f"SELECT * FROM memory_items WHERE {' AND '.join(conditions)} ORDER BY start_seconds",
                parameters,
            ).fetchall()
        return [self._memory_dict(row) for row in rows]

    def answer_cache_context(self, video_hash: str, *, verification_config_fingerprint: str) -> dict[str, Any]:
        """Return the current model, schema, verifier, and evidence revision for QA cache keys."""
        video = self.get_video(video_hash)
        if video is None:
            raise KeyError(f"Unknown video: {video_hash}")
        evidence_revision = hashlib.sha256(json.dumps([
            {
                "item_id": item["item_id"],
                "model": item["model_fingerprint"],
                "verification": item["verification_status"],
                "visibility": item["visibility_state"],
                "identity": item["identity_status"],
                "verifier": item["verification_model_fingerprint"],
                "config": item["verification_config_fingerprint"],
                "visual_evidence": item["visual_evidence"],
            }
            for item in self.list_memories(video_hash)
        ], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {
            "memory_schema": MEMORY_SCHEMA_VERSION,
            "memory_model": video["model_fingerprint"],
            "verification_config": verification_config_fingerprint,
            "evidence_revision": evidence_revision,
        }

    def list_evidence_frames(self, video_hash: str) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM evidence_frames WHERE video_hash=? ORDER BY time_seconds", (video_hash,)
            ).fetchall()
        return [dict(row) for row in rows if Path(row["path"]).is_file()]

    def clear_evidence_frames(self, video_hash: str) -> None:
        """Remove saved source-frame references and their local image files."""
        with self._db() as db:
            rows = db.execute(
                "SELECT path FROM evidence_frames WHERE video_hash=?", (video_hash,)
            ).fetchall()
            db.execute("DELETE FROM evidence_frames WHERE video_hash=?", (video_hash,))
        for row in rows:
            try:
                Path(row["path"]).unlink()
            except FileNotFoundError:
                pass

    def search_memories(
        self, video_hash: str, question: str, *, limit: int = 8,
        model_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        query_terms = _tokens(question)
        if not query_terms:
            return []
        is_temporal = bool(re.search(r"\b(when|before|after|move|pick|happen)\b|什么时候|之前|之后|移动|拿起", question, re.I))
        is_location = bool(re.search(r"\b(where|location|last|leave)\b|哪里|在哪|最后", question, re.I))
        ranked: list[dict[str, Any]] = []
        for item in self.list_memories(video_hash, model_fingerprint=model_fingerprint):
            payload = json.dumps(item.get("payload", {}), ensure_ascii=False)
            item_terms = _tokens(" ".join((item["label"], item["claim"], payload)))
            overlap = query_terms & item_terms
            if not overlap:
                continue
            score = len(overlap) / len(query_terms)
            if is_temporal and item["kind"] in {"event", "relation"}:
                score += 0.12
            if is_location and item["kind"] == "entity_observation":
                score += 0.12
            score += {"high": 0.05, "medium": 0.02, "low": 0.0}.get(item["confidence"], 0.0)
            ranked.append({**item, "retrieval_score": round(min(1.0, score), 4)})
        ranked.sort(key=lambda item: (-item["retrieval_score"], -item["start_seconds"], item["label"].casefold()))
        return ranked[:max(0, limit)]

    def save_evidence_frame(self, video_hash: str, *, frame_index: int, time_seconds: float, jpeg_bytes: bytes) -> Path:
        if not jpeg_bytes:
            raise ValueError("Evidence frame is empty")
        video = self.get_video(video_hash)
        if video is None:
            raise KeyError(f"Unknown video: {video_hash}")
        duration = float(video["metadata"].get("duration", 0.0))
        seconds = _finite_seconds(time_seconds, duration)
        if seconds is None or frame_index < 0:
            raise ValueError("Evidence frame timestamp/index is invalid")
        directory = self.evidence_root / video_hash
        _make_private_directory(directory)
        target = directory / f"{int(frame_index):010d}.jpg"
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(jpeg_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        with self._db() as db:
            db.execute(
                """INSERT OR REPLACE INTO evidence_frames(video_hash, frame_index, time_seconds, path)
                   VALUES(?,?,?,?)""",
                (video_hash, int(frame_index), seconds, str(target)),
            )
        return target

    def find_evidence_frame(self, video_hash: str, time_seconds: float, *, tolerance: float = 0.75) -> Path | None:
        with self._db() as db:
            row = db.execute(
                """SELECT path, ABS(time_seconds - ?) AS delta FROM evidence_frames
                   WHERE video_hash=? ORDER BY delta LIMIT 1""",
                (float(time_seconds), video_hash),
            ).fetchone()
        if row is None or row["delta"] > tolerance:
            return None
        path = Path(row["path"])
        return path if path.is_file() else None

    def get_saved_answer(
        self, video_hash: str, question: str, *, cache_context: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute(
                "SELECT answer_json FROM saved_answers WHERE video_hash=? AND query_key=?",
                (video_hash, _query_key(question, cache_context)),
            ).fetchone()
        return json.loads(row["answer_json"]) if row else None

    def save_answer(
        self, video_hash: str, question: str, answer: dict[str, Any], *, provenance: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> None:
        with self._db() as db:
            db.execute(
                """INSERT OR REPLACE INTO saved_answers(video_hash, query_key, query_text,
                   answer_json, provenance_json, updated_at) VALUES(?,?,?,?,?,?)""",
                (video_hash, _query_key(question, cache_context), question.strip(),
                 json.dumps(answer, ensure_ascii=False, sort_keys=True),
                 json.dumps(provenance, ensure_ascii=False, sort_keys=True), time.time()),
            )

    def update_visual_verification(
        self, item_id: int, result: dict[str, Any], *,
        verifier_fingerprint: str, config_fingerprint: str,
    ) -> None:
        verification_status = str(result.get("verification_status", "unknown"))
        visibility_state = str(result.get("visibility_state", "unknown"))
        identity_status = str(result.get("identity_status", "unverified"))
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"Unsupported verification status: {verification_status}")
        if visibility_state not in VISIBILITY_STATES:
            raise ValueError(f"Unsupported visibility state: {visibility_state}")
        if identity_status not in IDENTITY_STATUSES:
            raise ValueError(f"Unsupported identity status: {identity_status}")
        evidence = result.get("visual_evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("visual_evidence must be a list of exact-frame records")
        reason = str(result.get("uncertainty_reason") or "Visual review did not establish a more specific state.")
        with self._db() as db:
            row = db.execute("SELECT video_hash FROM memory_items WHERE item_id=?", (int(item_id),)).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory item: {item_id}")
            db.execute(
                """UPDATE memory_items SET verification_status=?, visibility_state=?, identity_status=?,
                   uncertainty_reason=?, verification_model_fingerprint=?, verification_config_fingerprint=?,
                   visual_evidence_json=? WHERE item_id=?""",
                (verification_status, visibility_state, identity_status, reason,
                 str(verifier_fingerprint), str(config_fingerprint),
                 json.dumps(evidence, ensure_ascii=False, sort_keys=True), int(item_id)),
            )
            db.execute("DELETE FROM saved_answers WHERE video_hash=?", (row["video_hash"],))

    def save_visual_observation(
        self, video_hash: str, *, label: str, target_description: str,
        timestamp_seconds: float, result: dict[str, Any], model_fingerprint: str,
        verifier_fingerprint: str, config_fingerprint: str,
    ) -> int:
        """Persist one verifier result as an exact-frame observation, idempotently."""
        verification_status = str(result.get("verification_status", "unknown"))
        visibility_state = str(result.get("visibility_state", "unknown"))
        identity_status = str(result.get("identity_status", "unverified"))
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"Unsupported verification status: {verification_status}")
        if visibility_state not in VISIBILITY_STATES:
            raise ValueError(f"Unsupported visibility state: {visibility_state}")
        if identity_status not in IDENTITY_STATUSES:
            raise ValueError(f"Unsupported identity state: {identity_status}")
        evidence = result.get("visual_evidence", [])
        if not isinstance(evidence, list) or len(evidence) != 1:
            raise ValueError("A visual observation requires exactly one source-frame evidence record")
        try:
            evidence_seconds = float(evidence[0]["timestamp_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The source-frame evidence must contain a numeric timestamp") from exc
        if not math.isfinite(evidence_seconds) or abs(evidence_seconds - timestamp_seconds) > 0.05:
            raise ValueError("Observation timestamp must match its exact-frame evidence")
        video = self.get_video(video_hash)
        if video is None:
            raise KeyError(f"Unknown video: {video_hash}")
        duration = float(video["metadata"].get("duration", 0.0))
        seconds = _finite_seconds(timestamp_seconds, duration)
        if seconds is None:
            raise ValueError("Observation timestamp falls outside the video")

        payload = json.dumps(
            {"target_description": target_description, "source": "targeted_visual_recheck"},
            ensure_ascii=False, sort_keys=True,
        )
        claim = f"Frame-local visual recheck for the requested {target_description.strip()} candidate."
        reason = str(result.get("uncertainty_reason") or "Visual review did not establish a more specific state.")
        evidence_json = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        timestamps_json = json.dumps([round(seconds, 3)])
        with self._db() as db:
            row = db.execute(
                """SELECT item_id FROM memory_items WHERE video_hash=? AND kind='visual_observation'
                   AND label=? AND start_seconds=? AND payload_json=? AND verification_config_fingerprint=?""",
                (video_hash, label, seconds, payload, config_fingerprint),
            ).fetchone()
            if row is not None:
                item_id = int(row["item_id"])
                db.execute(
                    """UPDATE memory_items SET claim=?, end_seconds=?, evidence_seconds_json=?,
                       confidence='unknown', model_fingerprint=?, verification_status=?, visibility_state=?,
                       identity_status=?, uncertainty_reason=?, verification_model_fingerprint=?,
                       verification_config_fingerprint=?, visual_evidence_json=? WHERE item_id=?""",
                    (claim, seconds, timestamps_json, model_fingerprint, verification_status,
                     visibility_state, identity_status, reason, verifier_fingerprint, config_fingerprint,
                     evidence_json, item_id),
                )
                return item_id
            cursor = db.execute(
                """INSERT INTO memory_items(video_hash,kind,label,claim,start_seconds,end_seconds,
                   evidence_seconds_json,confidence,model_fingerprint,payload_json,verification_status,
                   visibility_state,identity_status,uncertainty_reason,verification_model_fingerprint,
                   verification_config_fingerprint,visual_evidence_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (video_hash, "visual_observation", label, claim, seconds, seconds, timestamps_json,
                 "unknown", model_fingerprint, payload, verification_status, visibility_state,
                 identity_status, reason, verifier_fingerprint, config_fingerprint, evidence_json),
            )
            return int(cursor.lastrowid)
