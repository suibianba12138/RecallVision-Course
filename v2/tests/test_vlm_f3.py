"""Contract tests for VLM-F3 attribution authorization."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from recallvision.vlm_f1.hashing import package_sha256, sha256_file
from recallvision.vlm_f1.schema import OUTPUT_SCHEMA_VERSION, PACKAGE_VERSION, write_contract_files
from recallvision.vlm_f2.schema import (
    ATTRIBUTION_SCOPE_EVENT_LOCAL,
    ATTRIBUTION_SCOPE_VIDEO_LOCAL,
    ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
    ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
    VALIDATED_INTERPRETATION_SCHEMA_VERSION,
    VALIDATION_REPORT_SCHEMA_VERSION,
)
from recallvision.vlm_f3.authorization import authorize_attribution, write_artifacts
from recallvision.cli.vlm_f3 import run as run_f3_cli
from recallvision.vlm_f3.errors import VlmF3InputError


VIDEO_ID = "p1b_pixabay_bottle"
EVENT_ID = "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"
OBJECT_ID = "obj:bottle:0001"
PERSON_ID = "obj:person:0001"
EVENT_FRAMES = [92, 99]


def _event(*, person_ids: list[str] | None = None, event_id: str = EVENT_ID) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "MOVE_OBJECT",
        "start_time": 3.069733333333333,
        "end_time": 3.3033,
        "person_ids": [] if person_ids is None else person_ids,
        "object_ids": [OBJECT_ID],
        "zone_before": "left_region",
        "zone_after": "center_region",
        "evidence": [
            {
                "video_id": VIDEO_ID,
                "timestamp": 3.069733333333333,
                "start_time": None,
                "end_time": None,
                "frame_indices": [92],
                "segment_ref": f"{VIDEO_ID}#frame=92",
            },
            {
                "video_id": VIDEO_ID,
                "timestamp": 3.3033,
                "start_time": None,
                "end_time": None,
                "frame_indices": [99],
                "segment_ref": f"{VIDEO_ID}#frame=99",
            },
        ],
        "source_segment_ids": [
            f"{VIDEO_ID}#obj:bottle:0001#segment=1",
            f"{VIDEO_ID}#obj:bottle:0001#segment=2",
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_inputs(
    tmp_path: Path,
    *,
    event: dict[str, object] | None = None,
    source_status: str = "SUPPORTED_AGENT",
    validated_status: str = "VALIDATED_SUPPORTED_AGENT",
    identity_gate_status: str = "PASS_EVENT_LOCAL",
    attribution_ready: bool = True,
    mapped_person_id: str | None = PERSON_ID,
    identity_event_id: str = EVENT_ID,
    interaction_supported: bool = True,
    causal_support: bool = True,
    report_overrides: dict[str, object] | None = None,
    sidecar_overrides: dict[str, object] | None = None,
) -> dict[str, Path | str | dict[str, object]]:
    root = tmp_path / "inputs"
    root.mkdir(parents=True)
    event_record = event or _event()
    source_events = root / "move_object_events.json"
    _write_json(source_events, [event_record])

    package = root / "f1-package"
    package.mkdir()
    write_contract_files(package)
    manifest = {
        "package_version": PACKAGE_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "video_id": VIDEO_ID,
        "move_object_event_id": EVENT_ID,
        "target_object_id": OBJECT_ID,
        "candidate_labels": ["P1", "P2"],
        "candidate_mapping": {"P1": PERSON_ID, "P2": "obj:person:0002"},
        "selected_frame_indices": [78, 85, 91, 92, 96, 99, 100, 107, 113],
        "source_artifacts": {
            "events": {
                "path": "move_object_events.json",
                "sha256": sha256_file(source_events),
            },
            "event_evidence": {"path": "move_object_evidence.json", "sha256": "event-evidence"},
        },
    }
    _write_json(package / "manifest.json", manifest)
    package_hash = package_sha256(package)

    sidecar = {
        "validated_interpretation_schema_version": VALIDATED_INTERPRETATION_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "package_sha256": package_hash,
        "video_id": VIDEO_ID,
        "move_object_event_id": EVENT_ID,
        "target_object_id": OBJECT_ID,
        "interpretation_source": "EXTERNAL_CAPABILITY_EVALUATION_FIXTURE",
        "source_status": source_status,
        "validated_status": validated_status,
        "candidate_label": "P1" if mapped_person_id is not None else None,
        "mapped_persistent_person_id": mapped_person_id,
        "interaction_supported": interaction_supported,
        "causal_support": causal_support,
        "evidence_frames": EVENT_FRAMES,
        "brief_reason": "The supplied sequence supports candidate-caused displacement.",
        "identity_gate_status": identity_gate_status,
        "attribution_ready": attribution_ready,
        "identity_assertion": {
            "video_id": VIDEO_ID,
            "move_object_event_id": identity_event_id,
            "persistent_person_id": mapped_person_id or PERSON_ID,
            "identity_level": ID_LEVEL_2_EVENT_LOCAL_VALIDATED,
            "attribution_scope": ATTRIBUTION_SCOPE_EVENT_LOCAL,
            "ready_for_attribution": attribution_ready,
            "validation_source": "DOCUMENTED_EXTERNAL_HUMAN_REVIEW",
        },
    }
    if sidecar_overrides:
        sidecar.update(sidecar_overrides)
    sidecar_path = root / "validated_agent_interpretation.json"
    _write_json(sidecar_path, sidecar)

    report = {
        "validation_report_schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "package_binding_status": "PASS",
        "source_schema_status": "PASS",
        "candidate_validation_status": "PASS" if mapped_person_id is not None else "NOT_REQUIRED",
        "evidence_validation_status": "PASS",
        "identity_gate_status": identity_gate_status,
        "attribution_ready": attribution_ready,
        "vlm_inference": False,
        "provider_dependency": False,
        "event_mutation": False,
        "person_ids_populated": False,
        "who_moved_object_implemented": False,
    }
    if report_overrides:
        report.update(report_overrides)
    report_path = root / "validation_report.json"
    _write_json(report_path, report)
    return {
        "source_events": source_events,
        "f1_package": package,
        "f2_sidecar": sidecar_path,
        "f2_report": report_path,
        "package_hash": package_hash,
        "event": event_record,
    }


def _authorize(inputs: dict[str, Path | str | dict[str, object]], **kwargs: object) -> dict[str, object]:
    return authorize_attribution(
        source_events_path=inputs["source_events"],  # type: ignore[arg-type]
        event_id=EVENT_ID,
        f1_package=inputs["f1_package"],  # type: ignore[arg-type]
        f2_sidecar_path=inputs["f2_sidecar"],  # type: ignore[arg-type]
        f2_report_path=inputs["f2_report"],  # type: ignore[arg-type]
        **kwargs,
    )


def test_valid_authorization_writes_mapped_person_id(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)

    result = _authorize(inputs)
    written = write_artifacts(result, tmp_path / "out")

    assert result["authorization_status"] == "AUTHORIZED"
    assert result["derived_event"]["person_ids"] == [PERSON_ID]
    assert written["derived_event_path"] is not None
    assert json.loads(Path(written["derived_event_path"]).read_text())["person_ids"] == [PERSON_ID]


def test_source_event_is_immutable_and_derived_event_has_non_attribution_parity(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    before_bytes = Path(inputs["source_events"]).read_bytes()  # type: ignore[arg-type]
    source_before = copy.deepcopy(inputs["event"])

    result = _authorize(inputs)
    derived = result["derived_event"]
    source_after = json.loads(Path(inputs["source_events"]).read_text())[0]  # type: ignore[arg-type]

    assert Path(inputs["source_events"]).read_bytes() == before_bytes  # type: ignore[arg-type]
    assert source_after == source_before
    assert {key: value for key, value in derived.items() if key != "person_ids"} == {
        key: value for key, value in source_before.items() if key != "person_ids"
    }


@pytest.mark.parametrize(
    "change",
    [
        {"sidecar_overrides": {"move_object_event_id": "other-event"}},
        {"sidecar_overrides": {"video_id": "other-video"}},
        {"sidecar_overrides": {"target_object_id": "obj:bottle:9999"}},
    ],
)
def test_source_identity_binding_mismatch_is_blocked(tmp_path: Path, change: dict[str, object]) -> None:
    inputs = _build_inputs(tmp_path, **change)

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_SOURCE_MISMATCH"
    assert result["derived_event"] is None


def test_source_event_file_hash_mismatch_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    source_path = Path(inputs["source_events"])  # type: ignore[arg-type]
    source_path.write_text(source_path.read_text(encoding="utf-8").replace("left_region", "other_region"), encoding="utf-8")

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_SOURCE_MISMATCH"
    assert "hash" in result["authorization_reason"].lower()


def test_attribution_ready_false_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(
        tmp_path,
        attribution_ready=False,
        validated_status="VALIDATED_SUPPORTED_AGENT",
        identity_gate_status="PASS_EVENT_LOCAL",
        report_overrides={"identity_gate_status": "PASS_EVENT_LOCAL", "attribution_ready": False},
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_NOT_ATTRIBUTION_READY"
    assert result["derived_event"] is None


@pytest.mark.parametrize("status", ["AMBIGUOUS_AGENT", "NO_SUPPORTED_AGENT", "INSUFFICIENT_VISUAL_EVIDENCE"])
def test_abstention_status_cannot_authorize(tmp_path: Path, status: str) -> None:
    inputs = _build_inputs(
        tmp_path,
        source_status=status,
        validated_status=status,
        identity_gate_status="NOT_REQUIRED",
        attribution_ready=False,
        mapped_person_id=None,
        interaction_supported=status == "NO_SUPPORTED_AGENT",
        causal_support=False,
        report_overrides={
            "candidate_validation_status": "NOT_REQUIRED",
            "evidence_validation_status": "NOT_REQUIRED",
            "identity_gate_status": "NOT_REQUIRED",
            "attribution_ready": False,
        },
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_NOT_ATTRIBUTION_READY"
    assert result["derived_event"] is None


def test_identity_gate_failure_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(
        tmp_path,
        validated_status="VALIDATED_SUPPORTED_AGENT",
        identity_gate_status="FAILED_PERSON_MISMATCH",
        attribution_ready=True,
        report_overrides={"identity_gate_status": "FAILED_PERSON_MISMATCH", "attribution_ready": True},
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_IDENTITY_GATE"
    assert result["derived_event"] is None


def test_event_local_identity_for_different_event_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, identity_event_id="move-object:other:obj:bottle:0001:1->2")

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_IDENTITY_GATE"
    assert result["derived_event"] is None


def test_video_local_level_three_identity_can_authorize_same_video_person(tmp_path: Path) -> None:
    inputs = _build_inputs(
        tmp_path,
        identity_gate_status="PASS_VIDEO_LOCAL",
        sidecar_overrides={
            "identity_gate_status": "PASS_VIDEO_LOCAL",
            "identity_assertion": {
                "video_id": VIDEO_ID,
                "move_object_event_id": "move-object:other:obj:bottle:0001:1->2",
                "persistent_person_id": PERSON_ID,
                "identity_level": ID_LEVEL_3_VIDEO_LOCAL_VALIDATED,
                "attribution_scope": ATTRIBUTION_SCOPE_VIDEO_LOCAL,
                "ready_for_attribution": True,
                "validation_source": "DOCUMENTED_EXTERNAL_HUMAN_REVIEW",
            },
        },
        report_overrides={"identity_gate_status": "PASS_VIDEO_LOCAL"},
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "AUTHORIZED"
    assert result["derived_event"]["person_ids"] == [PERSON_ID]


def test_mapped_person_missing_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(
        tmp_path,
        mapped_person_id=None,
        validated_status="SUPPORTED_AGENT",
        attribution_ready=True,
        identity_gate_status="PASS_EVENT_LOCAL",
        report_overrides={"candidate_validation_status": "PASS"},
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


def test_already_attributed_source_is_blocked_without_overwrite(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, event=_event(person_ids=["obj:person:0099"]))

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_ALREADY_ATTRIBUTED"
    assert result["derived_event"] is None
    assert json.loads(Path(inputs["source_events"]).read_text())[0]["person_ids"] == ["obj:person:0099"]  # type: ignore[arg-type]


def test_corrupted_f2_summary_boolean_is_not_authorized(tmp_path: Path) -> None:
    inputs = _build_inputs(
        tmp_path,
        validated_status="SUPPORTED_AGENT_IDENTITY_BLOCKED",
        identity_gate_status="FAILED_MISSING",
        attribution_ready=True,
        report_overrides={"identity_gate_status": "FAILED_MISSING", "attribution_ready": True},
    )

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


@pytest.mark.parametrize(
    "flag",
    [
        "vlm_inference",
        "provider_dependency",
        "event_mutation",
        "person_ids_populated",
        "who_moved_object_implemented",
    ],
)
def test_f2_safety_flags_must_remain_false(tmp_path: Path, flag: str) -> None:
    inputs = _build_inputs(tmp_path, report_overrides={flag: True})

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


def test_malformed_candidate_label_is_blocked_without_type_error(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, sidecar_overrides={"candidate_label": []})

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


def test_cli_output_stays_inside_root_when_sidecar_video_id_is_unsafe(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, sidecar_overrides={"video_id": "../../outside"})
    output_root = tmp_path / "f3-output"

    result = run_f3_cli(
        source_events_path=inputs["source_events"],  # type: ignore[arg-type]
        event_id=EVENT_ID,
        f1_package=inputs["f1_package"],  # type: ignore[arg-type]
        f2_sidecar_path=inputs["f2_sidecar"],  # type: ignore[arg-type]
        f2_report_path=inputs["f2_report"],  # type: ignore[arg-type]
        output_root=output_root,
    )

    output_path = Path(result["output_dir"]).resolve()
    assert result["authorization_status"] == "BLOCKED_SOURCE_MISMATCH"
    assert output_path.is_relative_to(output_root.resolve())


def test_blocked_rerun_removes_stale_derived_event(tmp_path: Path) -> None:
    valid_inputs = _build_inputs(tmp_path / "valid")
    output_dir = tmp_path / "same-output"
    valid_result = _authorize(valid_inputs)
    valid_written = write_artifacts(valid_result, output_dir)
    assert Path(valid_written["derived_event_path"]).exists()

    blocked_inputs = _build_inputs(
        tmp_path / "blocked",
        attribution_ready=False,
        validated_status="VALIDATED_SUPPORTED_AGENT",
        identity_gate_status="PASS_EVENT_LOCAL",
        report_overrides={"identity_gate_status": "PASS_EVENT_LOCAL", "attribution_ready": False},
    )
    blocked_result = _authorize(blocked_inputs)
    blocked_written = write_artifacts(blocked_result, output_dir)

    assert blocked_written["derived_event_path"] is None
    assert not (output_dir / "attributed_move_object_event.json").exists()


@pytest.mark.parametrize("field", ["interaction_supported", "brief_reason", "interpretation_source"])
def test_missing_required_f2_sidecar_field_is_blocked(tmp_path: Path, field: str) -> None:
    inputs = _build_inputs(tmp_path)
    sidecar_path = Path(inputs["f2_sidecar"])  # type: ignore[arg-type]
    sidecar = json.loads(sidecar_path.read_text())
    del sidecar[field]
    _write_json(sidecar_path, sidecar)

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


def test_missing_identity_validation_source_is_blocked(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    sidecar_path = Path(inputs["f2_sidecar"])  # type: ignore[arg-type]
    sidecar = json.loads(sidecar_path.read_text())
    del sidecar["identity_assertion"]["validation_source"]
    _write_json(sidecar_path, sidecar)

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_INVALID_INTERPRETATION"
    assert result["derived_event"] is None


@pytest.mark.parametrize(
    "manifest_change",
    [
        {"candidate_labels": [""], "candidate_mapping": {"": PERSON_ID}},
        {"selected_frame_indices": [92.0]},
    ],
)
def test_malformed_f1_manifest_fails_closed_before_authorization(
    tmp_path: Path, manifest_change: dict[str, object]
) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = Path(inputs["f1_package"]) / "manifest.json"  # type: ignore[arg-type]
    manifest = json.loads(manifest_path.read_text())
    manifest.update(manifest_change)
    _write_json(manifest_path, manifest)

    with pytest.raises(VlmF3InputError):
        _authorize(inputs)


def test_f1_event_artifact_path_must_bind_to_supplied_source_file(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    manifest_path = Path(inputs["f1_package"]) / "manifest.json"  # type: ignore[arg-type]
    manifest = json.loads(manifest_path.read_text())
    manifest["source_artifacts"]["events"]["path"] = "unrelated_events.json"
    _write_json(manifest_path, manifest)

    result = _authorize(inputs)

    assert result["authorization_status"] == "BLOCKED_SOURCE_MISMATCH"
    assert result["derived_event"] is None


def test_deterministic_derived_event_report_and_hashes(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    result = _authorize(inputs)
    first = write_artifacts(result, tmp_path / "out-a")
    second = write_artifacts(result, tmp_path / "out-b")

    for key in ("derived_event_path", "authorization_report_path", "source_manifest_path"):
        assert Path(first[key]).read_bytes() == Path(second[key]).read_bytes()
    assert first["derived_event_sha256"] == second["derived_event_sha256"]
    assert first["authorization_report_sha256"] == second["authorization_report_sha256"]
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]


def test_source_file_is_unchanged_by_artifact_writing(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    source_path = Path(inputs["source_events"])  # type: ignore[arg-type]
    before_hash = sha256_file(source_path)

    result = _authorize(inputs)
    write_artifacts(result, tmp_path / "out")

    assert sha256_file(source_path) == before_hash
