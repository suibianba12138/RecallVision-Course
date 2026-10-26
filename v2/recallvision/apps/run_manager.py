"""Safe local video import and controlled execution of the existing Pipeline CLI."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

MAX_VIDEO_BYTES = 250 * 1024 * 1024
MAX_VIDEO_SECONDS = 120.0
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpeg", ".mpg"}
PIPELINE_STAGES = ("P1B", "P2A-R1", "P2B", "P2C", "P2D")
CONFIG_FILES = {
    "p2a": "configs/p2a_r1_entity_consolidation.yaml",
    "p2b": "configs/p2b_state_timeline.yaml",
    "p2c": "configs/p2c_zone_transition_events.yaml",
    "p2d": "configs/p2d_move_object_events.yaml",
    "motion": "configs/p3b1_robust_motion.yaml",
}


class VideoImportError(ValueError):
    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.status = status
        self.details = details or {}
        super().__init__(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_uploaded_video(
    upload_root: str | Path,
    filename: str,
    data: bytes,
    *,
    max_bytes: int = MAX_VIDEO_BYTES,
) -> tuple[Path, dict[str, Any]]:
    """Write an upload into a fresh UUID directory, ignoring its supplied path."""
    if not isinstance(data, bytes) or not data:
        raise VideoImportError("EMPTY_UPLOAD", "The uploaded video is empty.")
    if len(data) > max_bytes:
        raise VideoImportError(
            "UPLOAD_TOO_LARGE", f"Video exceeds the {max_bytes // (1024 * 1024)} MiB upload limit.",
            {"size_bytes": len(data), "max_bytes": max_bytes},
        )
    display_name = str(filename or "video").replace("\\", "/").rsplit("/", 1)[-1].strip()[:160] or "video"
    suffix = Path(display_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise VideoImportError(
            "UNSUPPORTED_VIDEO_FORMAT", "Choose a supported video file: MP4, MOV, MKV, AVI, M4V, WEBM, MPEG, or MPG.",
            {"extension": suffix},
        )
    root = Path(upload_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    job_dir = root / str(uuid.uuid4())
    job_dir.mkdir(exist_ok=False)
    destination = job_dir / f"source{suffix}"
    try:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Keep a failed upload directory inspectable; never delete caller data.
        raise
    return destination, {
        "original_name": display_name,
        "stored_path": str(destination),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "import_id": job_dir.name,
    }


def probe_video(
    video_path: str | Path,
    *,
    backend_python: str | Path,
    project_root: str | Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Read metadata in the inference environment using a fixed module command."""
    video = Path(video_path).resolve()
    python = Path(backend_python).expanduser().resolve()
    if not python.is_file():
        raise VideoImportError("BACKEND_PYTHON_MISSING", f"Inference Python does not exist: {python}")
    if not video.is_file():
        raise VideoImportError("VIDEO_MISSING", "The selected video file is no longer available.")
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(Path(project_root).resolve()) + (os.pathsep + prior if prior else "")
    try:
        result = subprocess.run(
            [str(python), "-m", "recallvision.cli.video_metadata", "--video", str(video)],
            cwd=Path(project_root).resolve(), env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise VideoImportError("VIDEO_PROBE_TIMEOUT", "Video metadata inspection timed out.") from error
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise VideoImportError(
            "VIDEO_PROBE_FAILED", "The inference environment could not read this video.",
            {"returncode": result.returncode, "stderr": result.stderr[-1200:]},
        ) from error
    if result.returncode or not isinstance(payload, dict) or payload.get("status") != "OK":
        raise VideoImportError(
            str(payload.get("status", "VIDEO_PROBE_FAILED")) if isinstance(payload, dict) else "VIDEO_PROBE_FAILED",
            str(payload.get("error", "The selected file is not a readable video.")) if isinstance(payload, dict) else "The selected file is not a readable video.",
            payload.get("details", {}) if isinstance(payload, dict) else {"stderr": result.stderr[-1200:]},
        )
    return payload["metadata"]


def validate_video_metadata(metadata: dict[str, Any], *, max_seconds: float = MAX_VIDEO_SECONDS) -> None:
    duration = metadata.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise VideoImportError("INVALID_VIDEO_DURATION", "Video duration could not be read.", {"metadata": metadata})
    if duration > max_seconds:
        raise VideoImportError(
            "VIDEO_TOO_LONG", f"This application currently accepts videos up to {max_seconds:g} seconds.",
            {"duration_seconds": duration, "max_seconds": max_seconds},
        )
    if not all(isinstance(metadata.get(key), (int, float)) and metadata[key] > 0 for key in ("fps", "width", "height")):
        raise VideoImportError("INVALID_VIDEO_METADATA", "Frame rate or video dimensions are invalid.", {"metadata": metadata})


def config_fingerprints(project_root: str | Path) -> dict[str, dict[str, str | None]]:
    root = Path(project_root).expanduser().resolve()
    result = {}
    for name, relative in CONFIG_FILES.items():
        path = root / relative
        result[name] = {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
    return result


def build_pipeline_argv(
    *,
    backend_python: str | Path,
    project_root: str | Path,
    video_path: str | Path,
    output_root: str | Path,
    model_path: str | Path,
) -> list[str]:
    """Build a fixed argv list; user strings are arguments, never shell text."""
    return [
        str(Path(backend_python).expanduser().resolve()), "-m", "recallvision.cli.pipeline",
        "--video", str(Path(video_path).resolve()),
        "--output-root", str(Path(output_root).resolve()),
        "--model", str(Path(model_path).expanduser().resolve()),
        "--tracker", "bytetrack.yaml", "--conf", "0.25", "--imgsz", "640", "--device", "cpu",
    ]


def discover_run_manifests(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    paths = list(base.rglob("run_manifest.json"))
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    video = manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    return {
        "video_sha256": video.get("sha256"),
        "model_sha256": model.get("sha256"),
        "configs": manifest.get("configs"),
        "tracker": model.get("tracker"),
        "confidence_threshold": model.get("confidence_threshold"),
        "imgsz": model.get("imgsz"),
        "device": model.get("device"),
    }


def is_compatible_run(
    manifest_path: str | Path,
    video_path: str | Path,
    model_path: str | Path,
    expected: dict[str, Any],
) -> tuple[bool, str]:
    """Require same video/model/configuration and intact stage output hashes."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "MANIFEST_UNREADABLE"
    if manifest.get("run_status") != "COMPLETED":
        return False, "RUN_NOT_COMPLETED"
    if not Path(video_path).is_file() or not Path(model_path).is_file():
        return False, "INPUT_OR_MODEL_MISSING"
    signature = manifest_signature(manifest)
    expected_signature = dict(expected)
    actual_video_sha = sha256_file(video_path)
    actual_model_sha = sha256_file(model_path)
    if expected_signature.get("video_sha256", actual_video_sha) != actual_video_sha:
        return False, "UPLOADED_VIDEO_HASH_MISMATCH"
    if expected_signature.get("model_sha256", actual_model_sha) != actual_model_sha:
        return False, "MODEL_FILE_HASH_MISMATCH"
    expected_signature.setdefault("video_sha256", actual_video_sha)
    expected_signature.setdefault("model_sha256", actual_model_sha)
    for key, value in expected_signature.items():
        if signature.get(key) != value:
            return False, f"{key.upper()}_MISMATCH"
    video_record = manifest.get("video", {})
    source_path = Path(str(video_record.get("path", ""))).expanduser()
    if not source_path.is_absolute():
        source_path = (path.parent / source_path).resolve()
    if not source_path.is_file() or sha256_file(source_path) != signature["video_sha256"]:
        return False, "SAVED_SOURCE_VIDEO_MISSING_OR_CHANGED"
    stages = {item.get("name"): item for item in manifest.get("stages", []) if isinstance(item, dict)}
    if any(stages.get(name, {}).get("status") != "COMPLETED" for name in PIPELINE_STAGES):
        return False, "STAGE_INCOMPLETE"
    for name in PIPELINE_STAGES:
        outputs = stages[name].get("outputs", [])
        if not outputs:
            return False, f"{name}_OUTPUTS_MISSING"
        for record in outputs:
            artifact = Path(str(record.get("path", ""))).expanduser()
            if not artifact.is_absolute():
                artifact = (path.parent / artifact).resolve()
            if not artifact.is_file() or not isinstance(record.get("sha256"), str) or sha256_file(artifact) != record["sha256"]:
                return False, f"{name}_OUTPUT_HASH_MISMATCH"
    return True, "MATCH"


def run_pipeline_process(
    *,
    backend_python: str | Path,
    project_root: str | Path,
    video_path: str | Path,
    output_root: str | Path,
    model_path: str | Path,
    log_root: str | Path,
    timeout_seconds: int = 1800,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one offline job and report real manifest stage state while it runs."""
    python = Path(backend_python).expanduser().resolve()
    video = Path(video_path).resolve()
    model = Path(model_path).expanduser().resolve()
    if not python.is_file():
        raise VideoImportError("BACKEND_PYTHON_MISSING", f"Inference Python does not exist: {python}")
    if not video.is_file():
        raise VideoImportError("VIDEO_MISSING", "The selected video file is no longer available.")
    if not model.is_file():
        raise VideoImportError("MODEL_WEIGHTS_NOT_FOUND", f"Local YOLO weights were not found: {model}")
    output = Path(output_root).resolve()
    logs = Path(log_root).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{uuid.uuid4()}.log"
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(Path(project_root).resolve()) + (os.pathsep + prior if prior else "")
    argv = build_pipeline_argv(
        backend_python=python, project_root=project_root, video_path=video,
        output_root=output, model_path=model,
    )
    started = time.monotonic()
    existing_manifests = set(discover_run_manifests(output))
    latest_path: Path | None = None
    manifest: dict[str, Any] = {}
    timed_out = False
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv, cwd=Path(project_root).resolve(), env=env, stdout=log,
            stderr=subprocess.STDOUT, shell=False, start_new_session=True,
        )
        try:
            while process.poll() is None:
                manifests = discover_run_manifests(output)
                new_manifests = [item for item in manifests if item not in existing_manifests]
                if new_manifests:
                    candidate = new_manifests[0]
                    try:
                        current = json.loads(candidate.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        current = {}
                    if current.get("video", {}).get("sha256") == sha256_file(video):
                        latest_path, manifest = candidate, current
                        if progress is not None:
                            progress(current)
                if time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                        process.wait(timeout=20)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=8)
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                    break
                time.sleep(0.75)
            return_code = process.wait()
        finally:
            log.flush()
            os.fsync(log.fileno())
    manifests = [item for item in discover_run_manifests(output) if item not in existing_manifests]
    for candidate in manifests:
        try:
            current = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if current.get("video", {}).get("sha256") == sha256_file(video):
            latest_path, manifest = candidate, current
            break
    run_status = manifest.get("run_status")
    success = return_code == 0 and run_status == "COMPLETED"
    if timed_out and run_status not in {"FAILED", "PARTIAL"}:
        run_status = "INTERRUPTED"
    if return_code != 0 and run_status == "COMPLETED":
        success = False
    failure = manifest.get("failure") or {}
    last_stage = next((item for item in reversed(manifest.get("stages", [])) if item.get("status") not in {"COMPLETED", "SKIPPED"}), None)
    return {
        "success": success,
        "run_status": run_status or ("FAILED" if return_code else "UNKNOWN"),
        "returncode": return_code,
        "run_dir": str(latest_path.parent) if latest_path else None,
        "manifest_path": str(latest_path) if latest_path else None,
        "manifest": manifest,
        "failure_stage": (failure.get("stage") if failure else None) or (last_stage.get("name") if last_stage else "PROCESS_STARTUP"),
        "failure_type": failure.get("type") or ("TIMEOUT" if timed_out else "PROCESS_EXIT" if return_code else None),
        "failure_explanation": failure.get("explanation") or ("Pipeline exceeded the local runtime limit." if timed_out else None),
        "log_path": str(log_path),
        "argv": argv,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
