"""Deterministic WHO_MOVED_OBJECT retrieval from authorized F3 memory only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.query.schema import Query, QueryType


class WhoMovedAnswerStatus(str, Enum):
    """Explicit fail-closed states for P3C retrieval."""

    SUCCESS = "SUCCESS"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    NO_AUTHORIZED_ATTRIBUTION = "NO_AUTHORIZED_ATTRIBUTION"
    AMBIGUOUS_MULTIPLE_EVENT_REFERENCE = "AMBIGUOUS_MULTIPLE_EVENT_REFERENCE"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    UNRESOLVED_PERSON_ID = "UNRESOLVED_PERSON_ID"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


@dataclass(frozen=True)
class AuthorizedAttributedEvent:
    """One F3 derived event paired with its F3 authorization record."""

    event: EventRecord
    authorization: Mapping[str, Any]
    attributed_event_path: str | None = None
    authorization_report_path: str | None = None
    source_manifest_path: str | None = None


@dataclass(frozen=True)
class WhoMovedQueryResult:
    """Structured P3C result with event and authorization provenance."""

    question_id: str
    answer: str
    status: WhoMovedAnswerStatus
    target_entity_id: str | None = None
    target_class: str | None = None
    reference_event_id: str | None = None
    resolved_event_id: str | None = None
    person_ids: tuple[str, ...] = ()
    authorization_status: str | None = None
    source_event_fingerprint: str | None = None
    source_artifact_sha256: str | None = None
    time_interval: tuple[float, float] | None = None
    frame_interval: tuple[int | None, int | None] | None = None
    attributed_event: EventRecord | None = None
    authorization: Mapping[str, Any] = field(default_factory=dict)
    attributed_event_path: str | None = None
    authorization_report_path: str | None = None
    source_manifest_path: str | None = None
    unresolved_person_ids: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        """Serialize a stable, provenance-complete result record."""

        event = self.attributed_event
        return {
            "question_id": self.question_id,
            "query_type": QueryType.WHO_MOVED_OBJECT.value,
            "answer": self.answer,
            "status": self.status.value,
            "target_entity_id": self.target_entity_id,
            "target_class": self.target_class,
            "reference_event_id": self.reference_event_id,
            "resolved_event_id": self.resolved_event_id,
            "person_ids": list(self.person_ids),
            "unresolved_person_ids": list(self.unresolved_person_ids),
            "authorization_status": self.authorization_status,
            "source_event_fingerprint": self.source_event_fingerprint,
            "source_artifact_sha256": self.source_artifact_sha256,
            "time_interval": (
                None
                if self.time_interval is None
                else {"start_time": self.time_interval[0], "end_time": self.time_interval[1]}
            ),
            "frame_interval": (
                None
                if self.frame_interval is None
                else {"start_frame": self.frame_interval[0], "end_frame": self.frame_interval[1]}
            ),
            "attributed_event": event.to_record() if event is not None else None,
            "source_attributed_event": {
                "path": self.attributed_event_path,
                "event_id": event.event_id if event is not None else None,
            },
            "f3_authorization_provenance": {
                "authorization_report_path": self.authorization_report_path,
                "source_manifest_path": self.source_manifest_path,
                "authorization_status": self.authorization_status,
                "source_event_id": self.authorization.get("source_event_id"),
                "source_event_fingerprint": self.authorization.get("source_event_hash"),
                "source_artifact_sha256": self.authorization.get("source_event_file_sha256"),
                "derived_event_sha256": self.authorization.get("derived_event_hash"),
            },
            "evidence": [_evidence_to_record(item) for item in event.evidence] if event else [],
        }


def answer_who_moved_object(
    query: Query,
    attributed_events: Sequence[AuthorizedAttributedEvent],
    persistent_entities: Sequence[Mapping[str, Any]],
) -> WhoMovedQueryResult:
    """Answer WHO_MOVED_OBJECT using only authorized attributed event memory.

    The function deliberately accepts no observations, video, F1 package, F2
    sidecar, pose result, or VLM provider.  Person IDs are copied from the
    authorized event and resolved only against persistent entity memory.
    """

    if query.query_type is not QueryType.WHO_MOVED_OBJECT:
        return _result(query, f"UNSUPPORTED_QUERY: {query.query_type.value}", WhoMovedAnswerStatus.UNSUPPORTED_QUERY)

    targets = _target_candidates(query, persistent_entities)
    target_label = query.target_class or ", ".join(query.target_entity_ids) or "target"
    if not targets:
        return _result(
            query,
            f"TARGET_NOT_FOUND: no persistent object matches {target_label}.",
            WhoMovedAnswerStatus.TARGET_NOT_FOUND,
            target_class=query.target_class,
        )
    if len(targets) != 1:
        return _result(
            query,
            f"AMBIGUOUS_TARGET: multiple persistent objects match {target_label}.",
            WhoMovedAnswerStatus.AMBIGUOUS_TARGET,
            target_class=query.target_class,
        )

    target_id, target_class = targets[0]
    eligible = [item for item in attributed_events if _is_authorized_attributed_event(item)]

    if query.reference_event_id is not None:
        references = [item for item in eligible if item.event.event_id == query.reference_event_id]
        if len(references) != 1:
            return _result(
                query,
                f"INVALID_REFERENCE: authorized attributed event {query.reference_event_id!r} was not uniquely resolved.",
                WhoMovedAnswerStatus.INVALID_REFERENCE,
                target_entity_id=target_id,
                target_class=target_class,
            )
        selected = references[0]
        invalid_reason = _invalid_event_reference(query, target_id, selected.event)
        if invalid_reason is not None:
            return _result(
                query,
                f"INVALID_REFERENCE: {invalid_reason}.",
                WhoMovedAnswerStatus.INVALID_REFERENCE,
                target_entity_id=target_id,
                target_class=target_class,
            )
    else:
        matching = [
            item
            for item in eligible
            if _event_matches_target_and_video(query, target_id, item.event)
        ]
        if not matching:
            return _result(
                query,
                f"NO_AUTHORIZED_ATTRIBUTION: no authorized attributed MOVE_OBJECT event was found for the {target_class}.",
                WhoMovedAnswerStatus.NO_AUTHORIZED_ATTRIBUTION,
                target_entity_id=target_id,
                target_class=target_class,
            )
        if len(matching) > 1:
            return _result(
                query,
                f"AMBIGUOUS_MULTIPLE_EVENT_REFERENCE: multiple authorized attributed MOVE_OBJECT events match the {target_class}; provide reference_event_id.",
                WhoMovedAnswerStatus.AMBIGUOUS_MULTIPLE_EVENT_REFERENCE,
                target_entity_id=target_id,
                target_class=target_class,
            )
        selected = matching[0]

    person_ids = tuple(selected.event.person_ids)
    entities = _person_entities(persistent_entities)
    unresolved = tuple(person_id for person_id in person_ids if person_id not in entities)
    if unresolved:
        return _event_result(
            query,
            selected,
            target_id,
            target_class,
            f"UNRESOLVED_PERSON_ID: persistent person entity memory does not resolve {', '.join(unresolved)}.",
            WhoMovedAnswerStatus.UNRESOLVED_PERSON_ID,
            unresolved_person_ids=unresolved,
        )

    if len(person_ids) == 1:
        answer = f"The {target_class} was moved by the tracked person {person_ids[0]}."
    else:
        answer = f"The {target_class} was moved by the tracked persons {', '.join(person_ids)}."
    return _event_result(
        query,
        selected,
        target_id,
        target_class,
        answer,
        WhoMovedAnswerStatus.SUCCESS,
    )


def load_authorized_event_memory(root: str | Path) -> list[AuthorizedAttributedEvent]:
    """Load only F3 attributed-event artifacts and their authorization reports."""

    source_root = Path(root)
    if not source_root.is_dir():
        raise ValueError(f"F3 attributed-event root does not exist: {source_root}")
    loaded: list[AuthorizedAttributedEvent] = []
    for event_path in sorted(source_root.rglob("attributed_move_object_event.json")):
        report_path = event_path.with_name("authorization_report.json")
        if not report_path.is_file():
            raise ValueError(f"F3 attributed event has no matching authorization report: {event_path}")
        event_record = _load_json_object(event_path, "attributed event")
        authorization = _load_json_object(report_path, "F3 authorization report")
        event = _event_from_record(event_record)
        derived_hash = authorization.get("derived_event_hash")
        if not isinstance(derived_hash, str) or _sha256_file(event_path) != derived_hash:
            raise ValueError(f"F3 derived event hash does not match: {event_path}")
        if authorization.get("source_event_id") != event.event_id:
            raise ValueError(f"F3 authorization source_event_id does not match: {event_path}")
        manifest_path = event_path.with_name("source_manifest.json")
        loaded.append(
            AuthorizedAttributedEvent(
                event=event,
                authorization=authorization,
                attributed_event_path=str(event_path),
                authorization_report_path=str(report_path),
                source_manifest_path=str(manifest_path) if manifest_path.is_file() else None,
            )
        )
    return loaded


def load_persistent_entities(path: str | Path) -> list[dict[str, Any]]:
    """Load the P2A persistent entity memory without loading observations."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("entities"), list):
        value = value["entities"]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("persistent entity memory must be a JSON list of objects")
    return value


def _is_authorized_attributed_event(item: AuthorizedAttributedEvent) -> bool:
    authorization = item.authorization
    return (
        authorization.get("authorization_status") == "AUTHORIZED"
        and authorization.get("source_event_id") == item.event.event_id
        and item.event.event_type is EventType.MOVE_OBJECT
        and bool(item.event.person_ids)
        and all(isinstance(person_id, str) and person_id for person_id in item.event.person_ids)
    )


def _target_candidates(
    query: Query, persistent_entities: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    objects: list[tuple[str, str]] = []
    for entity in persistent_entities:
        persistent_id = entity.get("persistent_id")
        class_name = entity.get("class_name")
        if not isinstance(persistent_id, str) or not persistent_id:
            continue
        if not isinstance(class_name, str) or not class_name:
            continue
        entity_type = entity.get("entity_type")
        if entity_type == "person" or class_name.casefold() == "person" or ":person:" in persistent_id:
            continue
        objects.append((persistent_id, class_name))
    if query.target_entity_ids:
        objects = [item for item in objects if item[0] in query.target_entity_ids]
    if query.target_class is not None:
        objects = [item for item in objects if item[1].casefold() == query.target_class.casefold()]
    return sorted(objects)


def _person_entities(persistent_entities: Sequence[Mapping[str, Any]]) -> set[str]:
    resolved: set[str] = set()
    for entity in persistent_entities:
        persistent_id = entity.get("persistent_id")
        class_name = entity.get("class_name")
        entity_type = entity.get("entity_type")
        if not isinstance(persistent_id, str) or not persistent_id:
            continue
        if entity_type == "person" or class_name == "person" or ":person:" in persistent_id:
            resolved.add(persistent_id)
    return resolved


def _event_matches_target_and_video(query: Query, target_id: str, event: EventRecord) -> bool:
    return (
        event.event_type is EventType.MOVE_OBJECT
        and target_id in event.object_ids
        and _event_video_id(event) == query.video_id
        and all(evidence.video_id == query.video_id for evidence in event.evidence)
    )


def _invalid_event_reference(query: Query, target_id: str, event: EventRecord) -> str | None:
    if event.event_type is not EventType.MOVE_OBJECT:
        return f"event {event.event_id!r} is not a MOVE_OBJECT event"
    if not _event_matches_target_and_video(query, target_id, event):
        if _event_video_id(event) != query.video_id:
            return f"event {event.event_id!r} does not belong to query video {query.video_id!r}"
        if target_id not in event.object_ids:
            return f"event {event.event_id!r} does not reference target {target_id!r}"
        return f"event {event.event_id!r} contains evidence from another video"
    return None


def _event_result(
    query: Query,
    selected: AuthorizedAttributedEvent,
    target_id: str,
    target_class: str,
    answer: str,
    status: WhoMovedAnswerStatus,
    *,
    unresolved_person_ids: tuple[str, ...] = (),
) -> WhoMovedQueryResult:
    event = selected.event
    return WhoMovedQueryResult(
        question_id=query.question_id,
        answer=answer,
        status=status,
        target_entity_id=target_id,
        target_class=target_class,
        reference_event_id=query.reference_event_id or event.event_id,
        resolved_event_id=event.event_id if status in {WhoMovedAnswerStatus.SUCCESS, WhoMovedAnswerStatus.UNRESOLVED_PERSON_ID} else None,
        person_ids=tuple(event.person_ids) if status is WhoMovedAnswerStatus.SUCCESS else (),
        authorization_status=selected.authorization.get("authorization_status"),
        source_event_fingerprint=_optional_text(selected.authorization.get("source_event_hash")),
        source_artifact_sha256=_optional_text(selected.authorization.get("source_event_file_sha256")),
        time_interval=(event.start_time, event.end_time),
        frame_interval=_frame_interval(event),
        attributed_event=event,
        authorization=dict(selected.authorization),
        attributed_event_path=selected.attributed_event_path,
        authorization_report_path=selected.authorization_report_path,
        source_manifest_path=selected.source_manifest_path,
        unresolved_person_ids=unresolved_person_ids,
    )


def _result(query: Query, answer: str, status: WhoMovedAnswerStatus, **kwargs: Any) -> WhoMovedQueryResult:
    return WhoMovedQueryResult(
        question_id=query.question_id,
        answer=answer,
        status=status,
        reference_event_id=query.reference_event_id,
        **kwargs,
    )


def _frame_interval(event: EventRecord) -> tuple[int | None, int | None]:
    if not event.evidence:
        return (None, None)
    first = event.evidence[0].frame_indices[0] if event.evidence[0].frame_indices else None
    last = event.evidence[-1].frame_indices[-1] if event.evidence[-1].frame_indices else None
    return (first, last)


def _event_video_id(event: EventRecord) -> str | None:
    if event.evidence:
        return event.evidence[0].video_id
    if event.source_segment_ids:
        return event.source_segment_ids[0].split("#", 1)[0]
    return None


def _event_from_record(record: Mapping[str, Any]) -> EventRecord:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("attributed event evidence must be a JSON list")
    try:
        return EventRecord(
            event_id=_required_text(record, "event_id"),
            event_type=EventType(record["event_type"]),
            start_time=_required_number(record, "start_time"),
            end_time=_required_number(record, "end_time"),
            person_ids=_required_string_list(record, "person_ids"),
            object_ids=_required_string_list(record, "object_ids"),
            zone_before=record.get("zone_before"),
            zone_after=record.get("zone_after"),
            evidence=[_evidence_from_record(item) for item in evidence],
            source_segment_ids=tuple(_required_string_list(record, "source_segment_ids")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed attributed event: {error}") from error


def _evidence_from_record(record: Any) -> EvidenceRef:
    if not isinstance(record, dict):
        raise ValueError("event evidence item must be an object")
    frames = record.get("frame_indices")
    if not isinstance(frames, list) or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames):
        raise ValueError("event evidence frame_indices must be a list of integers")
    timestamp = record.get("timestamp")
    if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))):
        raise ValueError("event evidence timestamp must be numeric or null")
    return EvidenceRef(
        video_id=_required_text(record, "video_id"),
        timestamp=None if timestamp is None else float(timestamp),
        start_time=_optional_number(record.get("start_time")),
        end_time=_optional_number(record.get("end_time")),
        frame_indices=frames,
        segment_ref=record.get("segment_ref"),
    )


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_string_list(record: Mapping[str, Any], key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return value


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("optional evidence time must be numeric or null")
    return float(value)


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _evidence_to_record(evidence: EvidenceRef) -> dict[str, Any]:
    return {
        "video_id": evidence.video_id,
        "timestamp": evidence.timestamp,
        "start_time": evidence.start_time,
        "end_time": evidence.end_time,
        "frame_indices": list(evidence.frame_indices),
        "segment_ref": evidence.segment_ref,
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
