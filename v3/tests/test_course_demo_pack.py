from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from recallvision.video_ai.persistent_memory import PersistentMemoryStore
from scripts.import_course_demo import restore_demo_video


def _make_video(path: Path) -> str:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((24, 32, 3), 25 + index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(video_hash: str) -> dict:
    return {
        "format": "recallvision-course-memory-pack",
        "version": 1,
        "pack_id": "rc1-course-demo-v1",
        "verification_config_fingerprint": "test-verification-v1",
        "videos": [{
            "sha256": video_hash,
            "source_name": "tiny-demo.avi",
            "source_url": "https://example.invalid/demo",
            "license_name": "test fixture",
            "model_fingerprint": "qwen-test:memory-v1",
            "metadata": {
                "duration": 0.8, "fps": 5.0, "frame_count": 4,
                "width": 32, "height": 24,
            },
            "analysis_metadata": {
                "generation_seconds": 3.5,
                "source_timestamps": [0.0, 0.4],
            },
            "memories": [{
                "kind": "entity_observation", "label": "Mug",
                "claim": "A blue mug is visible beside the notebook.",
                "start_seconds": 0.4, "end_seconds": 0.4,
                "evidence_seconds": [0.4], "confidence": "medium",
                "model_fingerprint": "qwen-test:memory-v1", "payload": {},
                "verification_status": "unverified", "visibility_state": "unknown",
                "identity_status": "unverified", "uncertainty_reason": "Model claim only.",
                "visual_evidence": [],
            }, {
                "kind": "visual_observation", "label": "mug",
                "claim": "Frame-local visual recheck supports the requested mug candidate.",
                "start_seconds": 0.4, "end_seconds": 0.4,
                "evidence_seconds": [0.4], "confidence": "medium",
                "model_fingerprint": "qwen-test:memory-v1",
                "payload": {"target_description": "blue mug"},
                "verification_status": "corroborated", "visibility_state": "present",
                "identity_status": "associated", "uncertainty_reason": "Frame-local support only.",
                "verification_model_fingerprint": "crop-review-test-v1",
                "verification_config_fingerprint": "crop-config-test-v1",
                "visual_evidence": [{
                    "timestamp_seconds": 0.4,
                    "detections": [{"raw_model_label": "mug", "score": 0.9,
                                    "bbox_xyxy": [1, 1, 20, 20],
                                    "target_presence_corroborated": True}],
                }],
            }],
            "evidence_frames": [{"frame_index": 2, "time_seconds": 0.4}],
            "saved_answers": [{
                "question": "What objects appeared in the video?",
                "answer": {
                    "question": "What objects appeared in the video?",
                    "answer": "The stored memory lists an unverified mug candidate.",
                    "abstain": False, "confidence": "not calibrated",
                    "verification_status": "unverified", "visibility_state": "unknown",
                    "identity_status": "unverified", "uncertainty": "Model claim only.",
                    "evidence_seconds": [0.4], "video_hash": video_hash,
                    "input_mode": "persistent visual memory", "intervals": [],
                },
                "provenance": {"input_mode": "persistent visual memory"},
            }],
        }],
    }


def test_restore_course_demo_memory_without_running_models(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny-demo.avi"
    digest = _make_video(video_path)
    store = PersistentMemoryStore(tmp_path / "memory")

    result = restore_demo_video(video_path, _pack(digest), store=store)

    video = store.get_video(digest)
    assert result["memory_count"] == 2
    assert video["analysis_status"] == "complete"
    assert video["analysis_metadata"]["course_demo_pack_id"] == "rc1-course-demo-v1"
    assert video["analysis_metadata"]["restore_performed_inference"] is False
    memories = store.list_memories(digest)
    assert len(memories) == 2
    visual = next(item for item in memories if item["kind"] == "visual_observation")
    assert visual["verification_status"] == "corroborated"
    assert visual["visibility_state"] == "present"
    evidence = store.find_evidence_frame(digest, 0.4, tolerance=0.01)
    assert evidence is not None and evidence.is_file()
    cache_context = store.answer_cache_context(
        digest, verification_config_fingerprint="test-verification-v1"
    )
    answer = store.get_saved_answer(
        digest, "What objects appeared in the video?", cache_context=cache_context
    )
    assert answer["answer"].startswith("The stored memory")

    second = restore_demo_video(video_path, _pack(digest), store=store)
    assert second["already_restored"] is True
    assert len(store.list_memories(digest)) == 2


def test_restore_rejects_wrong_source_bytes_before_mutating_store(tmp_path: Path) -> None:
    video_path = tmp_path / "wrong.avi"
    actual_hash = _make_video(video_path)
    store = PersistentMemoryStore(tmp_path / "memory")

    with pytest.raises(ValueError, match="does not match the course demo source hash"):
        restore_demo_video(video_path, _pack("0" * 64), store=store)

    assert actual_hash != "0" * 64
    assert store.list_videos() == []


def test_restored_course_demo_is_explicit_in_the_analyze_page(tmp_path: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest
    from recallvision.video_ai.persistent_memory import MEMORY_GENERATION_VERSION

    video_path = tmp_path / "tiny-demo.avi"
    digest = _make_video(video_path)
    pack = _pack(digest)
    current_memory_fingerprint = (
        "qwen3.5-2b:aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1:"
        + MEMORY_GENERATION_VERSION
    )
    pack["videos"][0]["model_fingerprint"] = current_memory_fingerprint
    for memory in pack["videos"][0]["memories"]:
        memory["model_fingerprint"] = current_memory_fingerprint
    store = PersistentMemoryStore(tmp_path / "app-store")
    restore_demo_video(video_path, pack, store=store)
    monkeypatch.setenv("RECALLVISION_MEMORY_DIR", str(tmp_path / "app-store"))
    monkeypatch.setenv("RECALLVISION_VIDEO_CACHE_DIR", str(tmp_path / "app-cache"))

    app = AppTest.from_file("recallvision/apps/video_memory.py").run(timeout=30)
    app.selectbox(key="saved_video_selector").set_value(digest).run(timeout=30)

    assert not app.exception
    assert any("Course demo memory restored" in item.value for item in app.info)


def test_course_demo_cli_help_runs_from_checkout_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "import_course_demo.py"), "--help"],
        cwd=project_root, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Locally downloaded Mixkit source video" in result.stdout


def test_course_demo_evidence_frame_indices_match_their_timestamps() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pack = json.loads((project_root / "demo" / "course_memory_pack_v1.json").read_text())

    for video in pack["videos"]:
        fps = float(video["metadata"]["fps"])
        for reference in video["evidence_frames"]:
            expected_index = round(float(reference["time_seconds"]) * fps)
            assert reference["frame_index"] == expected_index, (
                video["source_name"], reference, expected_index
            )


def test_interrupted_demo_restore_is_marked_failed_and_can_be_retried(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "tiny-demo.avi"
    digest = _make_video(video_path)
    pack = _pack(digest)
    store = PersistentMemoryStore(tmp_path / "memory")
    original_save_evidence_frame = store.save_evidence_frame
    fail_once = True

    def fail_first_evidence_write(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("simulated evidence storage failure")
        return original_save_evidence_frame(*args, **kwargs)

    monkeypatch.setattr(store, "save_evidence_frame", fail_first_evidence_write)
    with pytest.raises(OSError, match="simulated evidence storage failure"):
        restore_demo_video(video_path, pack, store=store)

    assert store.get_video(digest)["analysis_status"] == "failed"
    retry = restore_demo_video(video_path, pack, store=store)
    assert retry["already_restored"] is False
    assert store.get_video(digest)["analysis_status"] == "complete"
    assert len(store.list_memories(digest)) == 2
    assert len(store.list_evidence_frames(digest)) == 1


def test_malformed_saved_answer_is_rejected_before_registering_video(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny-demo.avi"
    digest = _make_video(video_path)
    pack = _pack(digest)
    pack["videos"][0]["saved_answers"][0]["answer"] = "not an answer object"
    store = PersistentMemoryStore(tmp_path / "memory")

    with pytest.raises(ValueError, match="saved course demo answer is malformed"):
        restore_demo_video(video_path, pack, store=store)

    assert store.list_videos() == []


def test_restoring_same_pack_removes_stale_misindexed_evidence_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny-demo.avi"
    digest = _make_video(video_path)
    pack = _pack(digest)
    store = PersistentMemoryStore(tmp_path / "memory")
    restore_demo_video(video_path, pack, store=store)
    store.save_evidence_frame(
        digest, frame_index=1, time_seconds=0.4, jpeg_bytes=b"stale frame at wrong source index"
    )

    result = restore_demo_video(video_path, pack, store=store)

    assert result["already_restored"] is False
    frames = store.list_evidence_frames(digest)
    assert [(row["frame_index"], row["time_seconds"]) for row in frames] == [(2, 0.4)]
