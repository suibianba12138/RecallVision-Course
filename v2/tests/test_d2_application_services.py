from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recallvision.apps.run_manager import (
    MAX_VIDEO_BYTES,
    VideoImportError,
    build_pipeline_argv,
    config_fingerprints,
    discover_run_manifests,
    is_compatible_run,
    store_uploaded_video,
)


def test_uploaded_video_uses_private_uuid_path_and_never_uses_supplied_path(tmp_path: Path) -> None:
    path, record = store_uploaded_video(
        tmp_path / "uploads", "../../unsafe\\folder\\clip.mp4", b"video-bytes",
    )
    assert path.is_file()
    assert path.name == "source.mp4"
    assert path.parent.parent == (tmp_path / "uploads").resolve()
    assert "/unsafe/" not in str(path)
    assert record["original_name"].endswith("clip.mp4")
    assert record["sha256"] == hashlib.sha256(b"video-bytes").hexdigest()


def test_upload_rejects_unsupported_extension_empty_and_oversized_payload(tmp_path: Path) -> None:
    with pytest.raises(VideoImportError) as extension:
        store_uploaded_video(tmp_path, "clip.txt", b"data")
    assert extension.value.status == "UNSUPPORTED_VIDEO_FORMAT"
    with pytest.raises(VideoImportError) as empty:
        store_uploaded_video(tmp_path, "clip.mp4", b"")
    assert empty.value.status == "EMPTY_UPLOAD"
    with pytest.raises(VideoImportError) as size:
        store_uploaded_video(tmp_path, "clip.mp4", b"12345", max_bytes=4)
    assert size.value.status == "UPLOAD_TOO_LARGE"
    assert MAX_VIDEO_BYTES > 0


def test_pipeline_command_is_fixed_argv_and_uses_explicit_local_model(tmp_path: Path) -> None:
    argv = build_pipeline_argv(
        backend_python=Path("/env/bin/python"), project_root=tmp_path,
        video_path=Path("/data/input.mp4"), output_root=Path("/out/runs"),
        model_path=Path("/models/yolo11n.pt"),
    )
    assert argv[:3] == ["/env/bin/python", "-m", "recallvision.cli.pipeline"]
    assert argv[argv.index("--model") + 1] == "/models/yolo11n.pt"
    assert argv[argv.index("--video") + 1] == "/data/input.mp4"
    assert "--target-entity-id" not in argv
    assert all(isinstance(part, str) for part in argv)


def test_run_reuse_requires_matching_video_model_configs_and_pinned_outputs(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    video.write_bytes(b"video")
    model.write_bytes(b"model")
    artifact = tmp_path / "obs.jsonl"
    artifact.write_text("{}\n")
    configs = {"p2a": {"path": "p2a.yaml", "sha256": "a" * 64}}
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({
        "run_id": "saved-run",
        "run_status": "COMPLETED",
        "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
        "model": {
            "resolved_path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "tracker": "bytetrack.yaml", "confidence_threshold": 0.25,
            "imgsz": 640, "device": "cpu",
        },
        "configs": configs,
        "stages": [{"name": name, "status": "COMPLETED", "outputs": [
            {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        ]} for name in ("P1B", "P2A-R1", "P2B", "P2C", "P2D")],
    }))
    expected = {
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "configs": configs, "tracker": "bytetrack.yaml", "confidence_threshold": 0.25,
        "imgsz": 640, "device": "cpu",
    }
    assert is_compatible_run(manifest_path, video, model, expected) == (True, "MATCH")
    artifact.write_text("changed\n")
    assert is_compatible_run(manifest_path, video, model, expected)[0] is False


def test_run_discovery_returns_newest_manifest_and_config_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "one" / "run_manifest.json"
    second = tmp_path / "two" / "run_manifest.json"
    first.parent.mkdir(); second.parent.mkdir()
    first.write_text(json.dumps({"run_id": "one", "run_status": "FAILED"}))
    second.write_text(json.dumps({"run_id": "two", "run_status": "COMPLETED"}))
    manifests = discover_run_manifests(tmp_path)
    assert set(manifests) == {first, second}
    fingerprints = config_fingerprints(Path(__file__).resolve().parents[1])
    assert {"p2a", "p2b", "p2c", "p2d", "motion"}.issubset(fingerprints)


def test_video_probe_uses_backend_python_module_and_returns_metadata(monkeypatch, tmp_path: Path) -> None:
    from recallvision.apps import run_manager

    video = tmp_path / "valid.mp4"
    video.write_bytes(b"video")
    python = tmp_path / "backend-python"
    python.write_text("")

    class Result:
        returncode = 0
        stdout = json.dumps({
            "status": "OK",
            "metadata": {"fps": 25.0, "width": 640, "height": 360, "frame_count": 50, "duration": 2.0},
        })
        stderr = ""

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(run_manager.subprocess, "run", fake_run)
    metadata = run_manager.probe_video(video, backend_python=python, project_root=tmp_path)

    assert metadata["duration"] == 2.0
    assert seen["argv"] == [
        str(python.resolve()), "-m", "recallvision.cli.video_metadata", "--video", str(video.resolve()),
    ]
    assert seen["kwargs"].get("shell", False) is False
    assert seen["kwargs"]["cwd"] == tmp_path.resolve()


def test_video_metadata_limit_is_explicit() -> None:
    from recallvision.apps.run_manager import validate_video_metadata

    metadata = {"fps": 25.0, "width": 640, "height": 360, "frame_count": 4000, "duration": 160.0}
    with pytest.raises(VideoImportError) as error:
        validate_video_metadata(metadata)
    assert error.value.status == "VIDEO_TOO_LONG"


def test_run_reuse_rejects_mutated_actual_video_even_when_expected_signature_is_old(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    video.write_bytes(b"video")
    model.write_bytes(b"model")
    artifact = tmp_path / "obs.jsonl"
    artifact.write_text("{}\n")
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest_path.parent.mkdir()
    configs = {"p2a": {"path": "p2a.yaml", "sha256": "b" * 64}}
    manifest_path.write_text(json.dumps({
        "run_id": "saved-run", "run_status": "COMPLETED",
        "video": {"path": str(video), "sha256": video_hash},
        "model": {"resolved_path": str(model), "sha256": model_hash, "tracker": "bytetrack.yaml",
                  "confidence_threshold": 0.25, "imgsz": 640, "device": "cpu"},
        "configs": configs,
        "stages": [{"name": name, "status": "COMPLETED", "outputs": [
            {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        ]} for name in ("P1B", "P2A-R1", "P2B", "P2C", "P2D")],
    }))
    expected = {"video_sha256": video_hash, "model_sha256": model_hash, "configs": configs,
                "tracker": "bytetrack.yaml", "confidence_threshold": 0.25, "imgsz": 640, "device": "cpu"}
    video.write_bytes(b"changed-video")

    compatible, reason = is_compatible_run(manifest_path, video, model, expected)
    assert compatible is False
    assert reason == "UPLOADED_VIDEO_HASH_MISMATCH"
