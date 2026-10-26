"""Contract tests for VLM-F2 external interpretation ingestion."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import EvidenceRef
from recallvision.vlm_f1.hashing import package_sha256
from recallvision.vlm_f1.schema import OUTPUT_SCHEMA_VERSION, PACKAGE_VERSION, write_contract_files
from recallvision.vlm_f2.errors import VlmF2ValidationError
from recallvision.vlm_f2.ingest import ingest_interpretation, write_sidecar
from recallvision.vlm_f2.schema import (
    ATTRIBUTION_SCOPE_EVENT_LOCAL,
    ATTRIBUTION_SCOPE_VIDEO_LOCAL,
    ID_LEVEL_1_UNCERTAIN,
    ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
    IdentityValidationAssertion,
)


VIDEO_ID = "p1b_pixabay_bottle"
EVENT_ID = "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"
OBJECT_ID = "obj:bottle:0001"
PERSON_ID = "obj:person:0001"
FRAMES = [78, 85, 91, 92, 96, 99, 100, 107, 113]


def _build_package(tmp_path: Path) -> tuple[Path, str]:
    package = tmp_path / "f1-package"
    package.mkdir()
    write_contract_files(package)
    (package / "frames").mkdir()
    (package / "frames/frame_0092.jpg").write_bytes(b"frame-92")
    manifest = {
        "package_version": PACKAGE_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "video_id": VIDEO_ID,
        "move_object_event_id": EVENT_ID,
        "target_object_id": OBJECT_ID,
        "candidate_labels": ["P1", "P2"],
        "candidate_mapping": {"P1": PERSON_ID, "P2": "obj:person:0002"},
        "selected_frame_indices": FRAMES,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return package, package_sha256(package)


def _source(package_hash: str, **changes: object) -> dict[str, object]:
    interpretation: dict[str, object] = {
        "status": "SUPPORTED_AGENT",
        "candidate_label": "P1",
        "interaction_supported": True,
        "causal_support": True,
        "evidence_frames": [92, 99],
        "brief_reason": "The supplied sequence supports candidate-caused displacement.",
    }
    interpretation.update(changes.pop("interpretation", {}))  # type: ignore[arg-type]
    result: dict[str, object] = {
        "package_version": PACKAGE_VERSION,
        "package_sha256": package_hash,
        "video_id": VIDEO_ID,
        "move_object_event_id": EVENT_ID,
        "target_object_id": OBJECT_ID,
        "interpretation_source": "EXTERNAL_CAPABILITY_EVALUATION_FIXTURE",
        "interpretation": interpretation,
    }
    result.update(changes)
    return result


def _assertion(
    *,
    level: str = ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    scope: str = ATTRIBUTION_SCOPE_EVENT_LOCAL,
    **changes: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "video_id": VIDEO_ID,
        "move_object_event_id": EVENT_ID,
        "persistent_person_id": PERSON_ID,
        "identity_level": level,
        "attribution_scope": scope,
        "ready_for_attribution": True,
        "validation_source": "DOCUMENTED_EXTERNAL_HUMAN_REVIEW",
    }
    result.update(changes)
    return result


def test_supported_agent_maps_closed_candidate_and_passes_event_identity(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)

    result = ingest_interpretation(
        package,
        _source(package_hash),
        identity_assertion=_assertion(),
    )

    validated = result["validated_agent_interpretation"]
    report = result["validation_report"]
    assert validated["source_status"] == "SUPPORTED_AGENT"
    assert validated["validated_status"] == "VALIDATED_SUPPORTED_AGENT"
    assert validated["candidate_label"] == "P1"
    assert validated["mapped_persistent_person_id"] == PERSON_ID
    assert validated["identity_gate_status"] == "PASS_EVENT_LOCAL"
    assert validated["attribution_ready"] is True
    assert report["package_binding_status"] == "PASS"
    assert report["evidence_validation_status"] == "PASS"


def test_unknown_candidate_is_rejected_even_if_reason_mentions_person_id(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)

    with pytest.raises(VlmF2ValidationError, match="closed candidate"):
        ingest_interpretation(
            package,
            _source(
                package_hash,
                interpretation={
                    "candidate_label": "obj:person:0001",
                    "brief_reason": "P1 and obj:person:0001 are visually identical.",
                },
            ),
            identity_assertion=_assertion(),
        )


def test_supported_agent_requires_candidate_causal_support_and_evidence(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    for changes, message in (
        ({"candidate_label": None}, "non-null"),
        ({"causal_support": False}, "causal_support"),
        ({"evidence_frames": []}, "non-empty"),
    ):
        with pytest.raises(VlmF2ValidationError, match=message):
            ingest_interpretation(
                package,
                _source(package_hash, interpretation=changes),
                identity_assertion=_assertion(),
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("package_sha256", "0" * 64, "package SHA256"),
        ("package_version", "VLM-F1-PACKAGE-0.0", "package_version"),
        ("video_id", "other-video", "video_id"),
        ("move_object_event_id", "other-event", "move_object_event_id"),
        ("target_object_id", "obj:bottle:9999", "target_object_id"),
    ],
)
def test_package_binding_is_exact(tmp_path: Path, field: str, value: str, message: str) -> None:
    package, package_hash = _build_package(tmp_path)
    source = _source(package_hash, **{field: value})

    with pytest.raises(VlmF2ValidationError, match=message):
        ingest_interpretation(package, source, identity_assertion=_assertion())


def test_evidence_frames_must_be_subset_of_f1_selected_frames(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)

    with pytest.raises(VlmF2ValidationError, match="absent from the package"):
        ingest_interpretation(
            package,
            _source(package_hash, interpretation={"evidence_frames": [92, 999]}),
            identity_assertion=_assertion(),
        )


@pytest.mark.parametrize(
    "assertion_changes",
    [
        {"persistent_person_id": "obj:person:0002"},
        {"move_object_event_id": "move-object:other:obj:bottle:0001:1->2"},
        {"identity_level": ID_LEVEL_1_UNCERTAIN},
        {"ready_for_attribution": False},
    ],
)
def test_identity_gate_blocks_attribution_without_rewriting_visual_interpretation(
    tmp_path: Path, assertion_changes: dict[str, object]
) -> None:
    package, package_hash = _build_package(tmp_path)

    result = ingest_interpretation(
        package,
        _source(package_hash),
        identity_assertion=_assertion(**assertion_changes),
    )

    validated = result["validated_agent_interpretation"]
    assert validated["source_status"] == "SUPPORTED_AGENT"
    assert validated["candidate_label"] == "P1"
    assert validated["mapped_persistent_person_id"] == PERSON_ID
    assert validated["validated_status"] == "SUPPORTED_AGENT_IDENTITY_BLOCKED"
    assert validated["attribution_ready"] is False
    assert validated["identity_gate_status"].startswith("FAILED_")


def test_missing_identity_preserves_visual_result_and_blocks_attribution(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)

    result = ingest_interpretation(package, _source(package_hash))
    validated = result["validated_agent_interpretation"]

    assert validated["mapped_persistent_person_id"] == PERSON_ID
    assert validated["identity_gate_status"] == "FAILED_MISSING"
    assert validated["attribution_ready"] is False


def test_direct_identity_assertion_instances_are_revalidated(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    invalid_assertion = IdentityValidationAssertion(
        video_id=VIDEO_ID,
        move_object_event_id=EVENT_ID,
        persistent_person_id=PERSON_ID,
        identity_level=ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
        attribution_scope=ATTRIBUTION_SCOPE_EVENT_LOCAL,
        ready_for_attribution=1,  # type: ignore[arg-type]
        validation_source="",
    )

    with pytest.raises(VlmF2ValidationError, match="validation_source"):
        ingest_interpretation(
            package,
            _source(package_hash),
            identity_assertion=invalid_assertion,
        )


def test_video_local_level_three_identity_passes_for_same_video(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)

    result = ingest_interpretation(
        package,
        _source(package_hash),
        identity_assertion=_assertion(
            level=ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
            scope=ATTRIBUTION_SCOPE_VIDEO_LOCAL,
        ),
    )

    validated = result["validated_agent_interpretation"]
    assert validated["identity_gate_status"] == "PASS_VIDEO_LOCAL"
    assert validated["attribution_ready"] is True


@pytest.mark.parametrize("status", ["AMBIGUOUS_AGENT", "NO_SUPPORTED_AGENT", "INSUFFICIENT_VISUAL_EVIDENCE"])
def test_abstentions_do_not_map_candidates_or_require_identity(tmp_path: Path, status: str) -> None:
    package, package_hash = _build_package(tmp_path)
    result = ingest_interpretation(
        package,
        _source(
            package_hash,
            interpretation={
                "status": status,
                "candidate_label": None,
                "interaction_supported": status == "NO_SUPPORTED_AGENT",
                "causal_support": False,
                "evidence_frames": [],
            },
        ),
    )

    validated = result["validated_agent_interpretation"]
    assert validated["mapped_persistent_person_id"] is None
    assert validated["identity_gate_status"] == "NOT_REQUIRED"
    assert validated["attribution_ready"] is False


def test_negative_control_preserves_interaction_without_causal_support(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    result = ingest_interpretation(
        package,
        _source(
            package_hash,
            interpretation={
                "status": "NO_SUPPORTED_AGENT",
                "candidate_label": None,
                "interaction_supported": True,
                "causal_support": False,
                "evidence_frames": [92],
            },
        ),
    )

    validated = result["validated_agent_interpretation"]
    assert validated["interaction_supported"] is True
    assert validated["causal_support"] is False
    assert validated["attribution_ready"] is False


def test_brief_reason_is_explanatory_and_never_authorizes_attribution(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    result = ingest_interpretation(
        package,
        _source(
            package_hash,
            interpretation={
                "status": "NO_SUPPORTED_AGENT",
                "candidate_label": None,
                "interaction_supported": False,
                "causal_support": False,
                "evidence_frames": [],
                "brief_reason": "P1 moved the bottle; obj:person:0001 is the identity.",
            },
        ),
        identity_assertion=_assertion(),
    )

    validated = result["validated_agent_interpretation"]
    assert validated["mapped_persistent_person_id"] is None
    assert validated["attribution_ready"] is False


def test_event_record_is_not_mutated_by_ingestion(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    event = EventRecord(
        event_id=EVENT_ID,
        event_type=EventType.MOVE_OBJECT,
        start_time=3.0,
        end_time=3.3,
        person_ids=[],
        object_ids=[OBJECT_ID],
        evidence=[EvidenceRef(video_id=VIDEO_ID, timestamp=3.0, frame_indices=[92])],
    )
    before = copy.deepcopy(event.to_record())

    ingest_interpretation(package, _source(package_hash), identity_assertion=_assertion())

    assert event.to_record() == before
    assert event.person_ids == []


def test_sidecar_files_and_hash_are_deterministic(tmp_path: Path) -> None:
    package, package_hash = _build_package(tmp_path)
    result = ingest_interpretation(
        package,
        _source(package_hash),
        identity_assertion=_assertion(),
    )
    first = write_sidecar(result, tmp_path / "out-a")
    second = write_sidecar(result, tmp_path / "out-b")

    assert first["sidecar_sha256"] == second["sidecar_sha256"]
    assert (
        (tmp_path / "out-a/validated_agent_interpretation.json").read_bytes()
        == (tmp_path / "out-b/validated_agent_interpretation.json").read_bytes()
    )
