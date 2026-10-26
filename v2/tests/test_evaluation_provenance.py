"""Tests-first prospective provenance and frozen-method guard."""

from __future__ import annotations

import json

import pytest

from recallvision.evaluation.provenance import (
    EvaluationProvenanceError,
    build_provenance_record,
    validate_provenance_record,
)


METHOD_HEAD = "bc9ca6370962df108bd44d4a50cbdf2132234e8a"
PROTOCOL_HEAD = "e3d66acea82033e7d1875b87cf2504e928a43d84"


def test_provenance_records_video_and_stage_hashes(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"small deterministic test video")

    record = build_provenance_record(
        video_path=video,
        video_id="eval:video:001",
        split="EVAL",
        method_head=METHOD_HEAD,
        evaluation_protocol_head=PROTOCOL_HEAD,
        tooling_head="a" * 40,
        configuration_hashes={"thresholds": "b" * 64},
        artifact_hashes={
            "observations": "c" * 64,
            "persistent_memory": "d" * 64,
            "state_timeline": "e" * 64,
            "events": "f" * 64,
            "f1": None,
            "f2": None,
            "f3": None,
            "query_output": "1" * 64,
        },
        annotation_schema_version="evaluation_annotation_v1",
    )

    validate_provenance_record(record)
    assert record["method_head"] == METHOD_HEAD
    assert record["evaluation_protocol_head"] == PROTOCOL_HEAD
    assert record["tooling_head"] != record["method_head"]
    assert len(record["video_sha256"]) == 64


def test_wrong_method_head_is_rejected_fail_closed(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    with pytest.raises(EvaluationProvenanceError, match="frozen method HEAD"):
        build_provenance_record(
            video_path=video,
            video_id="eval:video:001",
            split="EVAL",
            method_head="0" * 40,
            evaluation_protocol_head=PROTOCOL_HEAD,
            tooling_head="a" * 40,
            configuration_hashes={"thresholds": "b" * 64},
            artifact_hashes={
                "observations": "c" * 64,
                "persistent_memory": "d" * 64,
                "state_timeline": "e" * 64,
                "events": "f" * 64,
                "f1": None,
                "f2": None,
                "f3": None,
                "query_output": "1" * 64,
            },
            annotation_schema_version="evaluation_annotation_v1",
        )


def test_missing_required_stage_hash_is_rejected() -> None:
    record = {
        "provenance_schema_version": "eval_d0_provenance_v1",
        "video_id": "eval:video:001",
        "split": "EVAL",
        "video_sha256": "a" * 64,
        "method_head": METHOD_HEAD,
        "evaluation_protocol_head": PROTOCOL_HEAD,
        "tooling_head": "b" * 40,
        "configuration_hashes": {"thresholds": "c" * 64},
        "artifact_hashes": {"observations": "d" * 64},
        "annotation_schema_version": "evaluation_annotation_v1",
        "captured_at_utc": "2026-10-04T00:00:00Z",
    }

    with pytest.raises(EvaluationProvenanceError, match="artifact_hashes"):
        validate_provenance_record(record)
