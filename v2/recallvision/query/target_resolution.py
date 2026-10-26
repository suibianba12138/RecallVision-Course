"""Generic query-target normalization and persistent-entity resolution.

The resolver only consumes information present in the query and detector/memory
candidate records. It has no access to annotations, GT identities, evidence
frames, or answer locations.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from recallvision.query.schema import Query, TargetReference


class TargetResolutionStatus(str, Enum):
    """Explicit target-binding outcomes used by all query handlers."""

    RESOLVED = "RESOLVED"
    NO_COMPATIBLE_ENTITY = "NO_COMPATIBLE_ENTITY"
    AMBIGUOUS_MULTIPLE_ENTITIES = "AMBIGUOUS_MULTIPLE_ENTITIES"
    DETECTOR_VOCABULARY_UNSUPPORTED = "DETECTOR_VOCABULARY_UNSUPPORTED"
    INSUFFICIENT_ENTITY_EVIDENCE = "INSUFFICIENT_ENTITY_EVIDENCE"


@dataclass(frozen=True)
class TargetCandidate:
    """A detector/memory candidate that may be selected for a query target."""

    persistent_id: str
    class_name: str
    video_id: str
    entity_type: str = "object"

    def __post_init__(self) -> None:
        if not self.persistent_id:
            raise ValueError("persistent_id is required")
        if not self.class_name:
            raise ValueError("class_name is required")
        if not self.video_id:
            raise ValueError("video_id is required")


@dataclass(frozen=True)
class TargetResolution:
    """Deterministic resolution result with auditable candidate provenance."""

    reference: TargetReference
    status: TargetResolutionStatus
    candidate_entity_ids: tuple[str, ...] = ()
    compatible_system_classes: tuple[str, ...] = ()
    resolved_entity_id: str | None = None


# These are the standard Ultralytics COCO spellings used by the frozen P1B
# backend. The set is only used to distinguish unsupported vocabulary from a
# supported class with no observed entity; it does not create object bindings.
COCO_DETECTOR_CLASSES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    }
)


# Only globally defensible ontology spellings are included. D0-specific
# neighboring classes such as duffel bag -> suitcase are intentionally absent.
DETECTOR_ONTOLOGY_ALIASES: dict[str, tuple[str, ...]] = {
    "television": ("tv",),
    "tv": ("television",),
    "sofa": ("couch",),
    "couch": ("sofa",),
    "motorbike": ("motorcycle",),
    "motorcycle": ("motorbike",),
    "aeroplane": ("airplane",),
    "airplane": ("aeroplane",),
}

_IRREGULAR_PLURALS = {
    "children": "child",
    "men": "man",
    "women": "woman",
    "people": "person",
    "mice": "mouse",
}
_PLURAL_EXCEPTIONS = {"bus", "gas", "glass", "class"}


def build_target_reference(raw_phrase: str) -> TargetReference:
    """Normalize query wording without adding scene-specific object knowledge."""

    if not isinstance(raw_phrase, str):
        raise TypeError("raw_phrase must be a string")
    phrase = _normalize_phrase(raw_phrase)
    if not phrase:
        raise ValueError("target phrase must be non-empty")
    canonical = _singularize_phrase(phrase)
    return TargetReference(
        raw_phrase=raw_phrase.strip(),
        canonical_label=canonical,
        semantic_aliases=DETECTOR_ONTOLOGY_ALIASES.get(canonical, ()),
    )


def resolve_query_target(
    query: Query,
    candidates: Sequence[TargetCandidate],
) -> TargetResolution:
    """Resolve a query target against candidates from one memory source."""

    reference = query.target_reference
    if reference is None:
        if query.target_class:
            reference = build_target_reference(query.target_class)
        elif query.target_entity_ids:
            reference = TargetReference(raw_phrase="", canonical_label="")
        else:
            return TargetResolution(
                reference=TargetReference(raw_phrase="", canonical_label=""),
                status=TargetResolutionStatus.INSUFFICIENT_ENTITY_EVIDENCE,
            )
    return resolve_target_reference(
        reference,
        candidates,
        video_id=query.video_id,
        target_entity_ids=query.target_entity_ids,
    )


def resolve_target_reference(
    reference: TargetReference,
    candidates: Sequence[TargetCandidate],
    *,
    video_id: str,
    target_entity_ids: Sequence[str] = (),
    detector_ontology: Collection[str] = COCO_DETECTOR_CLASSES,
) -> TargetResolution:
    """Resolve one target reference with deterministic fail-closed semantics."""

    allowed_ids = set(target_entity_ids)
    labels = _reference_labels(reference)
    compatible: dict[str, TargetCandidate] = {}
    for candidate in candidates:
        if candidate.video_id != video_id or candidate.entity_type == "person":
            continue
        if allowed_ids and candidate.persistent_id not in allowed_ids:
            continue
        normalized_class = _canonicalize_label(candidate.class_name)
        if labels and normalized_class not in labels:
            continue
        compatible.setdefault(candidate.persistent_id, candidate)

    ordered = tuple(sorted(compatible.values(), key=lambda item: item.persistent_id))
    ids = tuple(item.persistent_id for item in ordered)
    classes = tuple(sorted({item.class_name for item in ordered}))
    if len(ordered) == 1:
        return TargetResolution(
            reference=reference,
            status=TargetResolutionStatus.RESOLVED,
            candidate_entity_ids=ids,
            compatible_system_classes=classes,
            resolved_entity_id=ordered[0].persistent_id,
        )
    if len(ordered) > 1:
        return TargetResolution(
            reference=reference,
            status=TargetResolutionStatus.AMBIGUOUS_MULTIPLE_ENTITIES,
            candidate_entity_ids=ids,
            compatible_system_classes=classes,
        )

    ontology = {_normalize_phrase(item) for item in detector_ontology}
    if labels and not labels.intersection(ontology) and not allowed_ids:
        status = TargetResolutionStatus.DETECTOR_VOCABULARY_UNSUPPORTED
    else:
        status = TargetResolutionStatus.NO_COMPATIBLE_ENTITY
    return TargetResolution(reference=reference, status=status)


def candidates_from_persistent_entities(
    entities: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
) -> tuple[TargetCandidate, ...]:
    """Adapt P2A-R1 entity records into generic object candidates."""

    candidates: list[TargetCandidate] = []
    for entity in entities:
        persistent_id = entity.get("persistent_id")
        class_name = entity.get("class_name")
        entity_type = entity.get("entity_type", "object")
        if not isinstance(persistent_id, str) or not persistent_id:
            continue
        if not isinstance(class_name, str) or not class_name:
            continue
        if entity_type == "person" or class_name.casefold() == "person" or ":person:" in persistent_id:
            continue
        candidates.append(
            TargetCandidate(
                persistent_id=persistent_id,
                class_name=class_name,
                video_id=video_id,
                entity_type="object",
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.persistent_id))


def candidates_from_timelines(timelines: Sequence[Any]) -> tuple[TargetCandidate, ...]:
    """Adapt P2B timelines into generic target candidates."""

    return tuple(
        sorted(
            (
                TargetCandidate(
                    persistent_id=timeline.persistent_id,
                    class_name=timeline.class_name,
                    video_id=timeline.video_id,
                )
                for timeline in timelines
            ),
            key=lambda item: (item.video_id, item.persistent_id),
        )
    )


def candidates_from_observations(observations: Sequence[Any]) -> tuple[TargetCandidate, ...]:
    """Adapt raw/consolidated observations, deduplicating persistent IDs."""

    candidates: dict[tuple[str, str], TargetCandidate] = {}
    for observation in observations:
        if observation.entity_type == "person" or not observation.persistent_id:
            continue
        candidate = TargetCandidate(
            persistent_id=observation.persistent_id,
            class_name=observation.class_name,
            video_id=observation.video_id,
        )
        candidates[(candidate.video_id, candidate.persistent_id)] = candidate
    return tuple(sorted(candidates.values(), key=lambda item: (item.video_id, item.persistent_id)))


def target_reference_matches_class(reference: TargetReference, class_name: str) -> bool:
    """Return whether a detector class is compatible with a query reference."""

    return _canonicalize_label(class_name) in _reference_labels(reference)


def _reference_labels(reference: TargetReference) -> frozenset[str]:
    labels = {
        _canonicalize_label(reference.canonical_label),
        *(_canonicalize_label(item) for item in reference.semantic_aliases),
    }
    labels.discard("")
    return frozenset(labels)


def _canonicalize_label(value: str) -> str:
    normalized = _normalize_phrase(value)
    return _singularize_phrase(normalized) if normalized else ""


def _normalize_phrase(value: str) -> str:
    value = re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE)
    return " ".join(value.split())


def _singularize_phrase(phrase: str) -> str:
    words = phrase.split()
    words[-1] = _singularize_token(words[-1])
    return " ".join(words)


def _singularize_token(token: str) -> str:
    if token in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[token]
    if token in _PLURAL_EXCEPTIONS or len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
