"""Local Streamlit application for reviewable object-centric video memory.

Workflow:
    Analyze — ingest video or reopen a saved analysis and select an entity.
    Explore — inspect/review exact observations and event candidates.
    Ask — retrieve WHERE / WHEN / BEFORE from reviewed memory.

The application targets a trusted single-user Linux workstation. The module
also exposes helper functions imported by the archived regression tests; those
interfaces are retained unchanged in this course-maintenance release.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from recallvision.memory.identity_overlay import OverlayError
from recallvision.memory.trust_review import TrustReviewError, TrustReviewOverlay
from recallvision.apps.run_manager import (
    MAX_VIDEO_BYTES, MAX_VIDEO_SECONDS, VideoImportError, config_fingerprints,
    discover_run_manifests, is_compatible_run, probe_video, run_pipeline_process,
    sha256_file as run_sha256, store_uploaded_video, validate_video_metadata,
)

PROJECT_ROOT = Path(os.environ.get("RECALLVISION_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_RUN = Path(os.environ.get(
    "RECALLVISION_DEFAULT_RUN_DIR",
    PROJECT_ROOT / "outputs/p1b_fresh_video/9d86f889e0fb2e3a/69a2aecf-1ca4-4e83-a4af-dd0783b0ea98",
)).resolve()
DEFAULT_HASHES = os.environ.get(
    "RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST",
    str(PROJECT_ROOT / "outputs/p2b1_identity_overlay/source_p1b_sha256_before.sha256"),
)
OUTPUT_BASE = Path(os.environ.get(
    "RECALLVISION_UI_OUTPUT_ROOT",
    PROJECT_ROOT / "outputs/p2b2_interactive_review",
)).resolve()
APP_DATA_ROOT = Path(os.environ.get(
    "RECALLVISION_APP_DATA_ROOT", PROJECT_ROOT / "outputs/recallvision_app",
)).resolve()
PIPELINE_OUTPUT_ROOT = APP_DATA_ROOT / "pipeline_runs"
UPLOAD_ROOT = APP_DATA_ROOT / "uploads"
JOB_LOG_ROOT = APP_DATA_ROOT / "logs"
MODEL_PATH = Path(os.environ.get("RECALLVISION_MODEL_PATH", str(PROJECT_ROOT / "models" / "yolo11n.pt"))).expanduser().resolve()
QUESTIONS = (
    "Where is the {class_name}?",
    "When was the {class_name} moved?",
    "Where was the {class_name} before?",
)


class BackendError(RuntimeError):
    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.status = status
        self.details = details or {}
        super().__init__(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_run() -> Path:
    if DEFAULT_RUN.is_dir():
        return DEFAULT_RUN
    manifests = sorted(PROJECT_ROOT.glob("outputs/p1b_fresh_video/**/run_manifest.json"))
    return manifests[-1].parent if manifests else DEFAULT_RUN


def default_hashes(run_dir: Path) -> str:
    return DEFAULT_HASHES if "p1b_fresh_video" in str(run_dir) and Path(DEFAULT_HASHES).is_file() else ""


def make_context(run_dir_text: str, hashes_text: str) -> dict[str, Any]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise BackendError("SOURCE_INCOMPLETE", "The selected folder has no run_manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video = manifest.get("video")
    run_id = manifest.get("run_id")
    if not isinstance(video, dict) or not isinstance(run_id, str):
        raise BackendError("SOURCE_INVALID", "The run manifest has no run ID or video record.")
    video_hash = video.get("sha256")
    raw_video_path = video.get("path")
    if not isinstance(video_hash, str) or not isinstance(raw_video_path, str):
        raise BackendError("SOURCE_INVALID", "The run manifest has no source video path or SHA-256.")
    video_path = Path(raw_video_path)
    if not video_path.is_absolute():
        video_path = (run_dir / video_path).resolve()
    if not video_path.is_file():
        raise BackendError("SOURCE_VIDEO_MISSING", f"Source video does not exist: {video_path}")
    if sha256(video_path) != video_hash:
        raise BackendError("STALE_SOURCE", "Source video hash differs from the run manifest.")
    hash_manifest = Path(hashes_text).expanduser().resolve() if hashes_text.strip() else None
    if hash_manifest is not None and not hash_manifest.is_file():
        raise BackendError("SOURCE_PROVENANCE_INCOMPLETE", f"Source hash manifest does not exist: {hash_manifest}")
    ui_root = (OUTPUT_BASE / video_hash[:16] / run_id).resolve()
    try:
        ui_root.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise BackendError("OUTPUT_IN_SOURCE_RUN", "Review outputs cannot be written inside the frozen run.")

    source_pin_paths = [
        run_dir / "p2a_r1/entity_conflicts.json",
        run_dir / "p2a_r1/continuation_candidates.json",
    ]
    pinned_outputs: set[Path] = set()
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for record in stage.get("outputs", []):
            if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
                continue
            output_path = Path(record["path"])
            pinned_outputs.add((output_path if output_path.is_absolute() else run_dir / output_path).resolve())
    missing_pins = [path.resolve() for path in source_pin_paths if path.resolve() not in pinned_outputs]
    hash_manifest_generated = False
    if hash_manifest is None and missing_pins:
        absent = [path for path in missing_pins if not path.is_file()]
        if absent:
            raise BackendError("SOURCE_INCOMPLETE", "P2A identity evidence artifacts are missing.", {"paths": [str(x) for x in absent]})
        ui_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = ui_root / "first_load_source_hashes.sha256"
        lines = [
            "# First-load snapshot for source artifacts omitted from the run manifest.",
            "# Detects later changes; it is not proof of integrity before this timestamp.",
            f"# created_at_utc={datetime.now(timezone.utc).isoformat()}",
        ]
        lines.extend(f"{sha256(path)}  {path}" for path in missing_pins)
        try:
            fd = os.open(snapshot_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        hash_manifest = snapshot_path
        hash_manifest_generated = True

    summary_path = run_dir / "p2a_r1/entity_consolidation_summary.json"
    if not summary_path.is_file() or not (run_dir / "p2a_r1/persistent_entities.json").is_file():
        raise BackendError("SOURCE_INCOMPLETE", "P2A entity artifacts are missing.")
    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "video_path": video_path,
        "video_sha256": video_hash,
        "video": video,
        "manifest": manifest,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "hash_manifest": hash_manifest,
        "hash_manifest_generated": hash_manifest_generated,
        "missing_pins": [str(path) for path in missing_pins],
        "ui_root": ui_root,
        "ledger": ui_root / "identity_operations.jsonl",
        "trust_ledger": ui_root / "trust_reviews.jsonl",
        "ui_log": ui_root / "ui_interaction_log.jsonl",
    }


def backend_python() -> Path:
    configured = os.environ.get("RECALLVISION_BACKEND_PYTHON", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(sys.executable).resolve()


def _saved_run_label(manifest_path: Path) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Unreadable run · {manifest_path.parent.name}"
    video = manifest.get("video", {})
    video_name = Path(str(video.get("path", "video"))).name
    status = str(manifest.get("run_status", "UNKNOWN"))
    return f"{video_name} · {status} · {manifest.get('run_id', manifest_path.parent.name)}"


def load_saved_run() -> None:
    selected = st.session_state.get("saved_run_choice")
    if not selected or selected == "":
        return
    manifest_path = Path(selected).resolve()
    st.session_state["run_path_input"] = str(manifest_path.parent)
    st.session_state["hash_path_input"] = default_hashes(manifest_path.parent)
    st.session_state["loaded_run_path"] = str(manifest_path.parent)
    st.session_state["loaded_hash_path"] = st.session_state["hash_path_input"]
    st.session_state["selected_entity_id"] = None
    st.session_state["asked_question_key"] = None
    st.session_state["main_navigation"] = "Analyze"
    st.cache_data.clear()


def _stage_status_text(manifest: dict[str, Any]) -> str:
    rows = [f"**Run:** {manifest.get('run_status', 'STARTING')}"]
    for stage in manifest.get("stages", []):
        if isinstance(stage, dict):
            rows.append(f"- {stage.get('name', 'Stage')}: **{stage.get('status', 'UNKNOWN')}**")
    if manifest.get("failure"):
        failure = manifest["failure"]
        rows.append(f"- Failure: `{failure.get('stage')}` · `{failure.get('type')}`")
    return "\n".join(rows)


def _compatible_saved_run(video_path: Path) -> Path | None:
    if not MODEL_PATH.is_file():
        return None
    expected = {
        "video_sha256": run_sha256(video_path),
        "model_sha256": run_sha256(MODEL_PATH),
        "configs": config_fingerprints(PROJECT_ROOT),
        "tracker": "bytetrack.yaml",
        "confidence_threshold": 0.25,
        "imgsz": 640,
        "device": "cpu",
    }
    for manifest_path in discover_run_manifests(PROJECT_ROOT / "outputs"):
        compatible, _reason = is_compatible_run(manifest_path, video_path, MODEL_PATH, expected)
        if compatible:
            return manifest_path
    return None


def _start_video_analysis(uploaded_file: Any) -> None:
    if uploaded_file is None:
        st.warning("Choose a local video before starting analysis.")
        return
    if int(getattr(uploaded_file, "size", 0)) > MAX_VIDEO_BYTES:
        st.error(f"This app accepts files up to {MAX_VIDEO_BYTES // (1024 * 1024)} MiB.")
        return
    try:
        stored_video, import_record = store_uploaded_video(UPLOAD_ROOT, uploaded_file.name, uploaded_file.getvalue())
        metadata = probe_video(stored_video, backend_python=backend_python(), project_root=PROJECT_ROOT)
        validate_video_metadata(metadata)
    except VideoImportError as error:
        st.error(f"Video import failed · {error.status}: {error}")
        return
    except OSError as error:
        st.error(f"Video import failed: {error}")
        return
    st.caption(
        f"{import_record['original_name']} · {import_record['size_bytes'] / (1024 * 1024):.1f} MiB · "
        f"{metadata['width']}×{metadata['height']} · {metadata['fps']:.2f} FPS · {metadata['duration']:.1f} s"
    )
    import_record["video_metadata"] = metadata
    import_record["imported_at"] = datetime.now(timezone.utc).isoformat()
    record_path = stored_video.parent / "import_record.json"
    try:
        with record_path.open("x", encoding="utf-8") as handle:
            json.dump(import_record, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        st.error("This upload directory already has an import record; no existing data was changed.")
        return
    compatible = _compatible_saved_run(stored_video)
    if compatible is not None:
        st.info(f"A completed run with matching video, model, settings, configurations, and artifact hashes is available: {compatible.parent.name}.")
        if st.button("Open matching saved analysis", key="open_compatible_run", type="primary"):
            import_record["reused_run_manifest"] = str(compatible)
            record_path.write_text(json.dumps(import_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            st.session_state["run_path_input"] = str(compatible.parent)
            st.session_state["hash_path_input"] = default_hashes(compatible.parent)
            st.session_state["loaded_run_path"] = str(compatible.parent)
            st.session_state["loaded_hash_path"] = st.session_state["hash_path_input"]
            st.session_state["selected_entity_id"] = None
            st.session_state["asked_question_key"] = None
            st.session_state["main_navigation"] = "Analyze"
            st.rerun()
        return
    ready, runtime = check_backend()
    if not ready:
        st.error("The inference environment is unavailable. Select the configured Python environment in Advanced.")
        with st.expander("Runtime details"):
            st.code(runtime)
        return
    if not MODEL_PATH.is_file():
        st.error(f"Local YOLO weights are missing: {MODEL_PATH}. No model download will be attempted.")
        return
    progress = st.empty()
    progress.markdown("**Starting offline analysis…**")
    try:
        result = run_pipeline_process(
            backend_python=backend_python(), project_root=PROJECT_ROOT,
            video_path=stored_video, output_root=PIPELINE_OUTPUT_ROOT, model_path=MODEL_PATH,
            log_root=JOB_LOG_ROOT, progress=lambda manifest: progress.markdown(_stage_status_text(manifest)),
        )
    except (OSError, VideoImportError) as error:
        st.error(f"Analysis could not start · {getattr(error, 'status', 'PROCESS_STARTUP')}: {error}")
        st.stop()
    import_record["pipeline_manifest"] = result.get("manifest_path")
    import_record["pipeline_run_status"] = result.get("run_status")
    import_record["pipeline_run_id"] = result.get("manifest", {}).get("run_id")
    record_path.write_text(json.dumps(import_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress.markdown(_stage_status_text(result.get("manifest", {})))
    if not result["success"]:
        st.error(
            f"Analysis did not complete · stage {result['failure_stage']} · {result['failure_type'] or result['run_status']}. "
            f"{result['failure_explanation'] or ''}"
        )
        st.caption(f"Failure state and logs were retained: {result.get('manifest_path') or result['log_path']}")
        st.stop()
    run_dir = result["run_dir"]
    st.session_state["run_path_input"] = run_dir
    st.session_state["hash_path_input"] = ""
    st.session_state["loaded_run_path"] = run_dir
    st.session_state["loaded_hash_path"] = ""
    st.session_state["selected_entity_id"] = None
    st.session_state["asked_question_key"] = None
    st.session_state["main_navigation"] = "Analyze"
    st.success("Analysis completed. Opening the saved video and candidate objects.")
    st.rerun()


def render_new_video_import() -> None:
    with st.expander("Analyze a new local video", expanded=False):
        st.caption(
            f"Offline analysis · up to {MAX_VIDEO_SECONDS:g} seconds and "
            f"{MAX_VIDEO_BYTES // (1024 * 1024)} MiB. The existing local YOLO model is used; uploads and runs are kept in a separate app folder."
        )
        uploaded = st.file_uploader(
            "Choose a video", type=["mp4", "mov", "mkv", "avi", "m4v", "webm", "mpeg", "mpg"],
            key="new_video_upload",
        )
        if uploaded is not None:
            st.caption(f"Selected: {Path(uploaded.name.replace(chr(92), '/')).name} · {uploaded.size / (1024 * 1024):.1f} MiB")
        if st.button("Analyze video", key="start_pipeline", type="primary", disabled=uploaded is None):
            _start_video_analysis(uploaded)
            st.stop()


def cli_args(context: dict[str, Any], command: str) -> list[str]:
    args = [
        str(backend_python()), "-m", "recallvision.cli.identity_overlay", command,
        "--run-dir", str(context["run_dir"]), "--ledger", str(context["ledger"]),
    ]
    if context["hash_manifest"] is not None:
        args.extend(["--source-hash-manifest", str(context["hash_manifest"])])
    return args


def run_cli(context: dict[str, Any], command: str, extra: list[str], timeout: int = 180) -> dict[str, Any]:
    python = backend_python()
    if not python.is_file():
        raise BackendError("BACKEND_PYTHON_MISSING", f"Backend Python not found: {python}")
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + previous if previous else "")
    try:
        result = subprocess.run(
            cli_args(context, command) + extra, cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise BackendError("BACKEND_TIMEOUT", f"Backend command {command} exceeded {timeout} seconds.") from error
    raw = result.stdout.strip() or result.stderr.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BackendError(
            "BACKEND_INVALID_OUTPUT", f"Backend command {command} returned invalid JSON.",
            {"returncode": result.returncode, "stdout": result.stdout[-1500:], "stderr": result.stderr[-1500:]},
        ) from error
    if result.returncode:
        raise BackendError(str(payload.get("status", "BACKEND_FAILED")),
                           str(payload.get("error", "Backend request was rejected.")),
                           payload.get("details", {}))
    return payload


def check_backend() -> tuple[bool, str]:
    python = backend_python()
    if not python.is_file():
        return False, f"Backend Python does not exist: {python}"
    try:
        result = subprocess.run([str(python), "-c", "import cv2"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=15, check=False)
    except subprocess.TimeoutExpired:
        return False, "OpenCV import check timed out."
    if result.returncode:
        return False, f"Backend Python must include OpenCV: {result.stderr.strip()[-400:]}"
    return True, str(python)


def context_for_cached(run_dir: str, ledger: str, hashes: str) -> dict[str, Any]:
    context = make_context(run_dir, hashes)
    context["ledger"] = Path(ledger)
    return context


@st.cache_data(ttl=5, show_spinner=False)
def candidates(run_dir: str, ledger: str, hashes: str) -> dict[str, Any]:
    return run_cli(context_for_cached(run_dir, ledger, hashes), "candidates", [])


@st.cache_data(ttl=5, show_spinner=False)
def history(run_dir: str, ledger: str, hashes: str) -> dict[str, Any]:
    return run_cli(context_for_cached(run_dir, ledger, hashes), "history", [])


@st.cache_data(ttl=5, show_spinner=False)
def mapping(run_dir: str, ledger: str, hashes: str) -> dict[str, Any]:
    return run_cli(context_for_cached(run_dir, ledger, hashes), "mapping", [])


@st.cache_data(ttl=5, show_spinner=False)
def inspect_pair(run_dir: str, ledger: str, hashes: str, entity_a: str, entity_b: str) -> dict[str, Any]:
    return run_cli(
        context_for_cached(run_dir, ledger, hashes), "inspect",
        ["--entity-a", entity_a, "--entity-b", entity_b],
    )


def safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def read_index(index_path: Path) -> list[dict[str, Any]] | None:
    if not index_path.is_file():
        return None
    try:
        records = json.loads(index_path.read_text(encoding="utf-8"))["evidence"]
        return records if isinstance(records, list) and all(Path(x["evidence_path"]).is_file() for x in records) else None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def write_index(index_path: Path, records: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(index_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"evidence": records}, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def entity_evidence(context: dict[str, Any], entity_id: str) -> list[dict[str, Any]]:
    key = (str(context["run_dir"]), entity_id)
    cache = st.session_state.setdefault("entity_evidence_cache", {})
    if key in cache and all(Path(x["evidence_path"]).is_file() for x in cache[key]):
        return cache[key]
    folder = context["ui_root"] / "review_evidence/entities" / safe_slug(entity_id)
    index_path = folder / "evidence_index.json"
    records = read_index(index_path)
    if records is None:
        if folder.exists():
            folder = folder.parent / f"{folder.name}-{uuid.uuid4().hex[:8]}"
            index_path = folder / "evidence_index.json"
        payload = run_cli(context, "evidence", ["--entity-id", entity_id, "--output-dir", str(folder)])
        records = payload.get("evidence", [])
        write_index(index_path, records)
    cache[key] = records
    return records


def pair_evidence(context: dict[str, Any], entity_a: str, entity_b: str, revision: int) -> list[dict[str, Any]]:
    pair = "--".join(sorted((safe_slug(entity_a), safe_slug(entity_b))))
    folder = context["ui_root"] / f"review_evidence/pairs/revision-{revision:06d}-{pair}"
    index_path = folder / "evidence_index.json"
    records = read_index(index_path)
    if records is not None:
        return records
    if folder.exists():
        folder = folder.parent / f"{folder.name}-{uuid.uuid4().hex[:8]}"
        index_path = folder / "evidence_index.json"
    payload = run_cli(context, "evidence", [
        "--entity-a", entity_a, "--entity-b", entity_b, "--output-dir", str(folder),
    ])
    records = payload.get("evidence", [])
    write_index(index_path, records)
    return records


def append_ui_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {"event_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), **event}
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_ui_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except json.JSONDecodeError:
            continue
    return result


def format_time(value: Any) -> str:
    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return "—"


def entity_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Persistent ID": item["persistent_entity"].get("persistent_id"),
            "Class": item["persistent_entity"].get("class_name"),
            "Observations": item.get("observation_count_in_memory"),
            "First frame": item.get("first_frame"),
            "Last frame": item.get("last_frame"),
            "First time": format_time(item["persistent_entity"].get("start_time")),
            "Last time": format_time(item["persistent_entity"].get("end_time")),
            "Track IDs": ", ".join(item.get("source_track_ids", [])),
            "Auto review": "PENDING_HUMAN_REVIEW",
        }
        for item in items
    ]


def find_entity(items: list[dict[str, Any]], entity_id: str) -> dict[str, Any]:
    for item in items:
        if item["persistent_entity"].get("persistent_id") == entity_id:
            return item
    raise BackendError("ENTITY_NOT_FOUND", f"Persistent entity not found: {entity_id}")


def show_evidence(records: list[dict[str, Any]], columns: int = 3) -> None:
    if not records:
        st.info("No source observations are available for this entity.")
        return
    for start in range(0, len(records), columns):
        row = st.columns(columns)
        for column, item in zip(row, records[start:start + columns]):
            with column:
                st.image(item["evidence_path"], use_container_width=True)
                st.caption(f"frame {item.get('frame_index')} · {format_time(item.get('timestamp'))} · track {item.get('track_id')}")


def candidate_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Score": x.get("score"),
            "Decision": x.get("rejection_reason") or ("accepted" if x.get("accepted") else "rejected"),
            "Hard gates": x.get("hard_gate_pass"),
            "Appearance": x.get("appearance_similarity"),
            "Spatial distance": x.get("center_distance"),
            "Temporal gap (s)": x.get("temporal_gap"),
        }
        for x in items
    ]


def json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def latest_rebuild(context: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    root = context["ui_root"] / "rebuilds"
    if not root.is_dir():
        return None, None
    for path in sorted(root.glob("**/rebuild_manifest.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("run_id") == context["run_id"] and value.get("source_video_sha256") == context["video_sha256"]:
            return path, value
    return None, None


def rebuild_current(context: dict[str, Any], entity_id: str, class_name: str, label: str) -> dict[str, Any]:
    current_revision = history(str(context["run_dir"]), str(context["ledger"]), str(context["hash_manifest"] or "")).get("overlay_revision", 0)
    output_root = context["ui_root"] / "rebuilds" / f"{label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    questions = [template.format(class_name=class_name) for template in QUESTIONS]
    args = [
        "--output-root", str(output_root), "--target-class", class_name,
        "--target-entity-id", entity_id, "--trust-ledger", str(context["trust_ledger"]),
    ]
    for question in questions:
        args.extend(["--question", question])
    start = time.perf_counter()
    result = run_cli(context, "rebuild", args, timeout=900)
    elapsed = time.perf_counter() - start
    manifest_path = Path(result.get("output_dir", output_root / f"revision-{current_revision:06d}")) / "rebuild_manifest.json"
    if result.get("run_status") != "COMPLETED":
        append_ui_event(context["ui_log"], {
            "event": "REBUILD_FAILED", "source_run_id": context["run_id"],
            "overlay_revision": current_revision, "target_entity_id": entity_id,
            "elapsed_seconds": elapsed, "manifest_path": str(manifest_path), "failure": result.get("failure"),
        })
        raise BackendError("REBUILD_FAILED", str(result.get("failure") or "Memory rebuild did not complete."))
    append_ui_event(context["ui_log"], {
        "event": "REBUILD_COMPLETED", "source_run_id": context["run_id"],
        "overlay_revision": result.get("overlay_revision"), "target_entity_id": entity_id,
        "elapsed_seconds": elapsed, "manifest_path": str(manifest_path), "run_status": result.get("run_status"),
    })
    return {"manifest_path": manifest_path, "manifest": result, "elapsed_seconds": elapsed}


def set_feedback(level: str, message: str) -> None:
    st.session_state["review_feedback"] = {"level": level, "message": message}


def handle_relation(context: dict[str, Any], relation: str, entity_a: str, entity_b: str,
                    evidence_a: dict[str, Any], evidence_b: dict[str, Any], revision: int,
                    all_entities: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    try:
        result = run_cli(context, "confirm", [
            "--entity-a", entity_a, "--entity-b", entity_b, "--relation", relation,
            "--evidence-frame", str(evidence_a["frame_index"]), "--evidence-path", str(evidence_a["evidence_path"]),
            "--evidence-frame", str(evidence_b["frame_index"]), "--evidence-path", str(evidence_b["evidence_path"]),
            "--expected-revision", str(revision),
        ])
    except BackendError as error:
        append_ui_event(context["ui_log"], {
            "event": "RELATION_REJECTED", "source_run_id": context["run_id"],
            "source_video_sha256": context["video_sha256"], "entity_a": entity_a, "entity_b": entity_b,
            "relation": relation, "expected_overlay_revision": revision, "status": error.status,
            "message": str(error), "details": error.details, "elapsed_seconds": time.perf_counter() - started,
        })
        st.cache_data.clear()
        set_feedback("error", f"Rejected: {error.status}. {error}")
        st.rerun()
        return
    append_ui_event(context["ui_log"], {
        "event": "RELATION_RECORDED", "source_run_id": context["run_id"],
        "source_video_sha256": context["video_sha256"], "entity_a": entity_a, "entity_b": entity_b,
        "relation": relation, "operation_id": result.get("operation_id"),
        "overlay_revision": result.get("overlay_revision"), "elapsed_seconds": time.perf_counter() - started,
    })
    st.cache_data.clear()
    st.session_state["selected_entity_id"] = entity_a
    st.session_state["pending_navigation"] = "Explore"
    st.session_state["asked_question_key"] = None
    try:
        entity = find_entity(all_entities, entity_a)["persistent_entity"]
        rebuilt = rebuild_current(context, entity_a, entity["class_name"], "relation")
    except BackendError as error:
        set_feedback("warning", f"Relation recorded, but rebuild failed: {error}")
    else:
        set_feedback("success", f"{relation} recorded at revision {result.get('overlay_revision')}; memory rebuilt in {rebuilt['elapsed_seconds']:.2f} s.")
    st.rerun()


def handle_revoke(context: dict[str, Any], relation: dict[str, Any], revision: int, class_name: str) -> None:
    try:
        result = run_cli(context, "revoke", [
            "--operation-id", relation["operation_id"], "--expected-revision", str(revision),
        ])
    except BackendError as error:
        append_ui_event(context["ui_log"], {
            "event": "REVOCATION_REJECTED", "source_run_id": context["run_id"],
            "target_operation_id": relation["operation_id"], "status": error.status,
            "message": str(error), "details": error.details, "expected_overlay_revision": revision,
        })
        st.cache_data.clear()
        set_feedback("error", f"Revocation rejected: {error.status}. {error}")
        st.rerun()
        return
    append_ui_event(context["ui_log"], {
        "event": "RELATION_REVOKED", "source_run_id": context["run_id"],
        "target_operation_id": relation["operation_id"], "revoke_operation_id": result.get("operation_id"),
        "overlay_revision": result.get("overlay_revision"),
    })
    st.cache_data.clear()
    st.session_state["selected_entity_id"] = relation["entity_a"]
    st.session_state["pending_navigation"] = "Explore"
    st.session_state["asked_question_key"] = None
    try:
        rebuilt = rebuild_current(context, relation["entity_a"], class_name, "revoke")
    except BackendError as error:
        set_feedback("warning", f"Revocation recorded, but rebuild failed: {error}")
    else:
        set_feedback("success", f"Relation revoked at revision {result.get('overlay_revision')}; memory rebuilt in {rebuilt['elapsed_seconds']:.2f} s.")
    st.rerun()


def trust_overlay_for_context(context: dict[str, Any]) -> TrustReviewOverlay:
    return TrustReviewOverlay(
        context["run_dir"], context["trust_ledger"],
        identity_ledger_path=context["ledger"],
        source_hash_manifest=context["hash_manifest"],
    )


def render_trust_observation_frame(context: dict[str, Any], candidate: dict[str, Any]) -> Path:
    output = context["ui_root"] / "trust_evidence" / f"{safe_slug(candidate['raw_observation_id'])}.jpg"
    if not output.is_file():
        payload = run_cli(context, "trust-evidence", [
            "--trust-ledger", str(context["trust_ledger"]),
            "--observation-id", candidate["raw_observation_id"],
            "--output", str(output),
        ])
        output = Path(payload["evidence_path"])
    return output


def _rebuild_after_trust_action(context: dict[str, Any], source_entity_id: str, class_name: str, label: str) -> None:
    st.cache_data.clear()
    st.session_state["asked_question_key"] = None
    try:
        rebuilt = rebuild_current(context, source_entity_id, class_name, label)
    except (BackendError, TrustReviewError, OverlayError) as error:
        set_feedback("warning", f"Review saved, but trusted history could not be rebuilt: {error}")
    else:
        set_feedback("success", f"Review saved. Trusted history was recomputed in {rebuilt['elapsed_seconds']:.2f} s.")
    st.rerun()


def selected_trust_counts(
    context: dict[str, Any], selected_id: str, mapping_data: dict[str, Any],
) -> dict[str, int]:
    trust = trust_overlay_for_context(context)
    effective_id = mapping_data.get("mapping", {}).get(selected_id, {}).get("effective_entity_id", selected_id)
    statuses = [item["trust_status"] for item in trust.observation_candidates(effective_id)]
    return {status: statuses.count(status) for status in ("SUPPORTED", "UNCERTAIN", "MISSING")}


def render_trust_review(
    context: dict[str, Any],
    selected_id: str,
    selected_class: str,
    mapping_data: dict[str, Any],
    label: str,
) -> None:
    """Small exact-frame review surface; no frame range is inferred."""
    try:
        trust = trust_overlay_for_context(context)
        snapshot = trust.snapshot()
    except (TrustReviewError, OverlayError, OSError, ValueError) as error:
        st.error(f"Evidence review is unavailable: {error}")
        return
    effective_id = mapping_data.get("mapping", {}).get(selected_id, {}).get("effective_entity_id", selected_id)
    observations = trust.observation_candidates(effective_id)
    if not observations:
        st.info("No source observations are available for this object to review.")
        return

    with st.expander("Review exact video evidence", expanded=False):
        st.caption("Each decision applies only to the selected frame. Unchecked frames remain uncertain; the system does not fill gaps between reviewed frames.")
        selected_observation_id = st.selectbox(
            "Choose an observation frame",
            [item["raw_observation_id"] for item in observations],
            format_func=lambda value: next(
                f"{format_time(next(row for row in observations if row['raw_observation_id'] == value)['timestamp'])} · frame {next(row for row in observations if row['raw_observation_id'] == value)['frame_index']} · {next(row for row in observations if row['raw_observation_id'] == value)['trust_status']}"
                for row in observations if row["raw_observation_id"] == value
            ),
            key=f"trust_observation_{safe_slug(effective_id)}",
        )
        selected_observation = trust.observation_by_id[selected_observation_id]
        st.image(render_trust_observation_frame(context, selected_observation), use_container_width=True)
        st.caption(
            f"Frame {selected_observation['frame_index']} at {format_time(selected_observation['timestamp'])} · "
            f"source: {selected_observation['source_algorithm']} · "
            f"automatic identity: {selected_observation['identity_provenance']}"
        )
        with st.form(f"trust_observation_form_{safe_slug(effective_id)}"):
            decision = st.radio(
                "What does this exact frame show?",
                ["SUPPORTED", "UNCERTAIN", "MISSING"],
                format_func=lambda value: {
                    "SUPPORTED": "This detection belongs to the selected object",
                    "UNCERTAIN": "I cannot tell from this frame",
                    "MISSING": "Wrong detection or wrong identity",
                }[value],
                horizontal=False,
                key=f"trust_observation_decision_{safe_slug(effective_id)}",
            )
            reason = st.text_input("Review note", key=f"trust_observation_reason_{safe_slug(effective_id)}")
            acknowledged = st.checkbox(
                f"I reviewed frame {selected_observation['frame_index']} only.",
                key=f"trust_observation_ack_{safe_slug(effective_id)}",
            )
            save_observation = st.form_submit_button("Save frame review", type="primary")
        if save_observation:
            if not acknowledged:
                st.warning("Confirm that you inspected this exact frame before saving.")
            elif not reason.strip():
                st.warning("Add a short note so the review decision is auditable.")
            else:
                try:
                    trust.record_observation_review(
                        selected_observation_id,
                        decision,
                        reviewed_frame_indices=[selected_observation["frame_index"]],
                        reason=reason,
                        expected_revision=snapshot["overlay_revision"],
                    )
                except (TrustReviewError, OverlayError) as error:
                    set_feedback("error", f"Frame review rejected: {error}")
                    st.rerun()
                _rebuild_after_trust_action(context, selected_id, selected_class, "trust-frame")

        st.markdown("#### Review a movement candidate")
        source_path = context["run_dir"] / "p2d" / "move_object_events.json"
        source_events = []
        if source_path.is_file():
            try:
                source_events = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source_events = []
        entity_sources = {
            source_id for source_id, item in mapping_data.get("mapping", {}).items()
            if item.get("effective_entity_id") == effective_id
        }
        event_candidates = [
            item for item in source_events
            if isinstance(item, dict) and item.get("event_type") == "MOVE_OBJECT"
            and set(item.get("object_ids", [])) & entity_sources
        ]
        if not event_candidates:
            st.info("There are no saved automatic movement candidates for this object. Lack of a candidate is not proof that the object never moved.")
        else:
            reviewed_observation_candidates = [
                item for item in observations
                if item["source_entity_id"] in entity_sources
            ]
            event_id = st.selectbox(
                "Automatic movement candidate",
                [item["event_id"] for item in event_candidates],
                format_func=lambda value: next(
                    f"{format_time(next(item for item in event_candidates if item['event_id'] == value).get('start_time'))} – {format_time(next(item for item in event_candidates if item['event_id'] == value).get('end_time'))} · "
                    f"{snapshot['events'].get(value, {}).get('status', 'REQUIRES REVIEW')}"
                    for item in event_candidates if item["event_id"] == value
                ),
                key=f"trust_event_{safe_slug(effective_id)}",
            )
            chosen_event = next(item for item in event_candidates if item["event_id"] == event_id)
            event_frames = sorted({
                frame for evidence in chosen_event.get("evidence", []) if isinstance(evidence, dict)
                for frame in evidence.get("frame_indices", []) if isinstance(frame, int)
            })
            reviewed_ids = [
                item["raw_observation_id"] for item in reviewed_observation_candidates
                if item["frame_index"] in event_frames
            ]
            columns = st.columns(min(2, len(event_frames)) or 1)
            for index, frame in enumerate(event_frames):
                evidence_candidates = [
                    item for item in reviewed_observation_candidates if item["frame_index"] == frame
                ]
                with columns[index % len(columns)]:
                    st.markdown(f"**Evidence frame {frame}**")
                    if evidence_candidates:
                        st.image(render_trust_observation_frame(context, trust.observation_by_id[evidence_candidates[0]["raw_observation_id"]]), use_container_width=True)
                        st.caption(format_time(evidence_candidates[0]["timestamp"]))
                    else:
                        st.warning("No matching saved observation exists for this event frame.")
            with st.form(f"trust_event_form_{safe_slug(effective_id)}_{safe_slug(event_id)}"):
                event_decision = st.radio(
                    "Do these reviewed boundary frames support a position change?",
                    ["SUPPORTED", "UNCERTAIN", "MISSING"],
                    format_func=lambda value: {
                        "SUPPORTED": "Yes, the before/after positions differ",
                        "UNCERTAIN": "I cannot determine this",
                        "MISSING": "No, this candidate is not supported",
                    }[value],
                    key=f"trust_event_decision_{safe_slug(event_id)}",
                )
                event_reason = st.text_input("Review note", key=f"trust_event_reason_{safe_slug(event_id)}")
                event_ack = st.checkbox(
                    "I reviewed every displayed evidence frame. This does not confirm the path or exact movement onset between them.",
                    key=f"trust_event_ack_{safe_slug(event_id)}",
                )
                save_event = st.form_submit_button("Save event review", type="primary")
            if save_event:
                if not event_ack:
                    st.warning("Confirm that you inspected all evidence frames before saving.")
                elif len(event_frames) < 2 or len(reviewed_ids) < 2:
                    st.warning("This candidate lacks two distinct reviewed observation frames and cannot be confirmed.")
                elif not event_reason.strip():
                    st.warning("Add a short note so the event decision is auditable.")
                else:
                    try:
                        trust.record_event_review(
                            event_id,
                            event_decision,
                            reviewed_observation_ids=reviewed_ids,
                            reviewed_frame_indices=event_frames,
                            reason=event_reason,
                            expected_revision=snapshot["overlay_revision"],
                        )
                    except (TrustReviewError, OverlayError) as error:
                        set_feedback("error", f"Event review rejected: {error}")
                        st.rerun()
                    _rebuild_after_trust_action(context, selected_id, selected_class, "trust-event")

        active_reviews = [
            item for item in snapshot["history"]["operations"]
            if item.get("action") in {"OBSERVATION_REVIEW", "EVENT_REVIEW"}
            and item.get("revocation_status") == "ACTIVE"
            and item.get("source_identity_status") == "CURRENT"
            and item.get("effective_entity_id") == effective_id
        ]
        if active_reviews:
            st.markdown("#### Recent review decisions")
            st.caption(f"Trust review revision {snapshot['overlay_revision']} · {len(active_reviews)} active decision(s)")
            revoke_id = st.selectbox(
                "Choose a review to undo",
                [item["operation_id"] for item in active_reviews],
                format_func=lambda value: next(
                    f"{item.get('action', '').replace('_', ' ').title()} · {item.get('trust_status', item.get('event_review_status'))} · "
                    f"frame(s) {item.get('reviewed_frame_indices', [])}"
                    for item in active_reviews if item["operation_id"] == value
                ),
                key=f"trust_revoke_{safe_slug(effective_id)}",
            )
            if st.button("Undo selected review", key=f"trust_undo_{safe_slug(effective_id)}"):
                try:
                    trust.revoke(revoke_id, expected_revision=snapshot["overlay_revision"])
                except (TrustReviewError, OverlayError) as error:
                    set_feedback("error", f"Undo rejected: {error}")
                    st.rerun()
                _rebuild_after_trust_action(context, selected_id, selected_class, "trust-undo")


def show_timeline(timelines: list[dict[str, Any]], class_name: str, target_id: str | None) -> None:
    chosen = [x for x in timelines if x.get("persistent_id") == target_id] if target_id else [x for x in timelines if x.get("class_name") == class_name]
    if not chosen:
        st.info("No timeline is available for the selected entity in this view.")
        return
    rows = []
    for timeline in chosen:
        for index, segment in enumerate(timeline.get("segments", []), 1):
            rows.append({
                "Entity": timeline.get("persistent_id"), "Segment": index,
                "Start": format_time(segment.get("start_time")), "End": format_time(segment.get("end_time")),
                "Start frame": segment.get("start_frame"), "End frame": segment.get("end_frame"),
                "Zone": segment.get("zone") or segment.get("to_zone") or segment.get("state"),
                "Observations": segment.get("observation_count"),
            })
    st.subheader("State timeline")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_memory(context: dict[str, Any], entities: list[dict[str, Any]], selected_class: str,
                mapping_data: dict[str, Any], revision: int, selected_source_id: str | None) -> None:
    st.subheader("Effective identity mapping")
    source_mapping = mapping_data.get("mapping", {})
    rows = []
    for source_id, value in source_mapping.items():
        entity = find_entity(entities, source_id)["persistent_entity"]
        if entity.get("class_name") != selected_class:
            continue
        rows.append({
            "Source entity": source_id, "Effective entity": value.get("effective_entity_id"),
            "Identity provenance": value.get("identity_provenance"),
            "Confirmation operations": ", ".join(value.get("human_confirmation_operation_ids", [])),
            "Automatic review": value.get("source_identity_review_status"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("HUMAN_CONFIRMED overlay results remain separate from automatic P2A review status.")
    source_timelines = json_list(context["run_dir"] / "p2b/object_state_timelines.json")
    if not selected_source_id:
        st.info("Choose one entity explicitly before viewing its query history. No candidate is selected automatically.")
        show_timeline(source_timelines, selected_class, None)
        return
    selected_effective_id = source_mapping.get(selected_source_id, {}).get("effective_entity_id", selected_source_id)
    manifest_path, rebuild = latest_rebuild(context)
    if not rebuild:
        st.info("No overlay rebuild exists yet. The source P1-B timeline is shown as the automatic baseline.")
        show_timeline(source_timelines, selected_class, selected_source_id)
        return
    if rebuild.get("overlay_revision") != revision:
        st.warning(f"Latest rebuild revision {rebuild.get('overlay_revision')} differs from current overlay revision {revision}; showing the automatic baseline until it is recomputed.")
        show_timeline(source_timelines, selected_class, selected_source_id)
        return
    target = rebuild.get("effective_target_entity_id")
    if target != selected_effective_id:
        st.info("The latest rebuild belongs to a different entity. Recompute after choosing this candidate to view its current history.")
        show_timeline(source_timelines, selected_class, selected_source_id)
        return
    st.markdown(
        f"**{rebuild.get('run_status')} · revision {rebuild.get('overlay_revision')} · "
        f"effective target {target}**"
    )
    provenance = rebuild.get("target_identity_provenance") or {}
    st.info(
        f"Identity: {provenance.get('identity_provenance', 'UNKNOWN')} · source IDs "
        f"{', '.join(provenance.get('source_entity_ids', [])) or 'unresolved'} · source review "
        f"{provenance.get('source_identity_review_status', 'UNKNOWN')}"
    )
    base = manifest_path.parent
    show_timeline(json_list(base / "p2b/object_state_timelines.json"), selected_class, target)
    events = json_list(base / "p2c/zone_change_events.json") + json_list(base / "p2d/move_object_events.json")
    event_rows = []
    for event in events:
        if target not in event.get("object_ids", []):
            continue
        event_rows.append({
            "Type": event.get("event_type"), "Start": format_time(event.get("start_time")),
            "End": format_time(event.get("end_time")),
            "Frames": ", ".join(str(frame) for item in event.get("evidence", []) for frame in item.get("frame_indices", [])),
        })
    st.subheader("Events from this rebuild")
    st.dataframe(event_rows, use_container_width=True, hide_index=True)
    st.caption("No rows means this target has no supported event in the current memory revision.")
    st.subheader("Historical queries and evidence")
    for query in rebuild.get("query_results", []):
        path = Path(query["path"])
        if not path.is_file():
            st.warning(f"Query output missing: {path}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        st.markdown(f"**{query.get('question')}** — {result.get('status')}")
        st.write(result.get("answer", result.get("message", "No supported answer.")))
        details = result.get("overlay_provenance", {})
        st.caption(
            f"revision {details.get('overlay_revision')} · {details.get('identity_provenance')} · "
            f"source IDs {', '.join(details.get('source_entity_ids', [])) or 'unresolved'}"
        )
        images = [item for item in result.get("evidence_paths", []) if Path(item).is_file()]
        if images:
            show_evidence([{"evidence_path": item, "frame_index": result.get("frame_index"),
                            "timestamp": result.get("last_seen_timestamp"), "track_id": "query evidence"}
                           for item in images], columns=min(3, len(images)))


def entity_display_labels(items: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        items,
        key=lambda item: (
            str(item["persistent_entity"].get("class_name", "")),
            float(item["persistent_entity"].get("start_time") or 0),
            str(item["persistent_entity"].get("persistent_id", "")),
        ),
    )
    counters: dict[str, int] = {}
    labels: dict[str, str] = {}
    for item in ordered:
        entity = item["persistent_entity"]
        class_name = str(entity.get("class_name") or "Object")
        counters[class_name] = counters.get(class_name, 0) + 1
        entity_id = str(entity.get("persistent_id"))
        labels[entity_id] = (
            f"{class_name.title()} {counters[class_name]} · "
            f"{format_time(entity.get('start_time'))}–{format_time(entity.get('end_time'))}"
        )
    return labels


def identity_suggestions(anchor_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor = find_entity(items, anchor_id)
    fragment_owner: dict[str, str] = {}
    for item in items:
        entity_id = str(item["persistent_entity"].get("persistent_id"))
        for fragment_id in item["persistent_entity"].get("fragment_ids", []):
            fragment_owner[str(fragment_id)] = entity_id
    anchor_fragments = set(anchor["persistent_entity"].get("fragment_ids", []))
    suggestions: dict[str, list[dict[str, Any]]] = {}
    for candidate in anchor.get("continuation_candidates", []):
        source = str(candidate.get("source_fragment_id", ""))
        target = str(candidate.get("target_fragment_id", ""))
        if source in anchor_fragments:
            other_id = fragment_owner.get(target)
        elif target in anchor_fragments:
            other_id = fragment_owner.get(source)
        else:
            continue
        if not other_id or other_id == anchor_id:
            continue
        suggestions.setdefault(other_id, []).append(candidate)
    results = []
    for entity_id, records in suggestions.items():
        scores = [float(item["score"]) for item in records if item.get("score") is not None]
        gaps = [abs(float(item.get("temporal_gap") or 0)) for item in records]
        results.append({
            "entity_id": entity_id,
            "best_score": max(scores) if scores else None,
            "nearest_gap": min(gaps) if gaps else None,
            "records": records,
        })
    results.sort(key=lambda item: (
        item["best_score"] is None,
        -(item["best_score"] or 0),
        item["nearest_gap"] if item["nearest_gap"] is not None else float("inf"),
        item["entity_id"],
    ))
    return results


def current_rebuild(
    context: dict[str, Any], mapping_data: dict[str, Any], revision: int, source_entity_id: str,
) -> tuple[Path | None, dict[str, Any] | None, str]:
    expected = mapping_data.get("mapping", {}).get(source_entity_id, {}).get("effective_entity_id", source_entity_id)
    manifest_path, manifest = latest_rebuild(context)
    if not manifest or not manifest_path:
        return None, None, expected
    if manifest.get("run_status") != "COMPLETED":
        return None, None, expected
    if manifest.get("overlay_revision") != revision:
        return None, None, expected
    if manifest.get("effective_target_entity_id") != expected:
        return None, None, expected
    if context.get("trust_ledger"):
        try:
            trust_revision = TrustReviewOverlay(
                context["run_dir"], context["trust_ledger"],
                identity_ledger_path=context["ledger"],
                source_hash_manifest=context["hash_manifest"],
            ).snapshot()["overlay_revision"]
        except (TrustReviewError, OverlayError, OSError, ValueError):
            return None, None, expected
        if (manifest.get("trust_review") or {}).get("overlay_revision") != trust_revision:
            return None, None, expected
    return manifest_path, manifest, expected


def friendly_zone(value: Any) -> str:
    text = str(value or "").strip()
    for suffix in ("_region", "_zone"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text.replace("_", " ").title() if text else "Location estimate unavailable"


def jump_to_video(timestamp: Any) -> None:
    try:
        st.session_state["video_start_time"] = max(0.0, float(timestamp))
    except (TypeError, ValueError):
        st.session_state["video_start_time"] = 0.0
    st.session_state["analyze_media_view"] = "Original video"
    st.session_state["main_navigation"] = "Analyze"


def navigate_to(page: str) -> None:
    st.session_state["main_navigation"] = page


def render_evidence_cards(records: list[dict[str, Any]], key_prefix: str, limit: int = 4) -> None:
    valid = [
        item for item in records
        if item.get("evidence_path") and Path(str(item["evidence_path"])).is_file()
    ]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, Any]] = set()
    for item in valid:
        marker = (str(item["evidence_path"]), item.get("frame_index"), item.get("timestamp"))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
        if len(unique) >= limit:
            break
    if not unique:
        st.info("No viewable frame evidence is available for this result.")
        return
    for start in range(0, len(unique), 2):
        row = st.columns(min(2, len(unique) - start))
        for index, (column, item) in enumerate(zip(row, unique[start:start + 2]), start):
            with column:
                with st.container(border=True):
                    st.image(item["evidence_path"], use_container_width=True)
                    frame = item.get("frame_index")
                    timestamp = item.get("timestamp")
                    st.caption(f"{format_time(timestamp)} · frame {frame if frame is not None else '—'}")
                    if timestamp is not None:
                        st.button(
                            "View this moment in the video",
                            key=f"{key_prefix}_jump_{index}_{frame}",
                            on_click=jump_to_video,
                            args=(timestamp,),
                        )


def render_timeline(segments: list[dict[str, Any]], key_prefix: str) -> None:
    if not segments:
        st.info("No state history is available for this object yet.")
        return
    st.markdown("#### Object history")
    st.caption("Area labels are approximate. A boundary change alone does not confirm physical movement.")
    for index, segment in enumerate(segments, 1):
        start = segment.get("start_time")
        end = segment.get("end_time")
        label = friendly_zone(segment.get("zone") or segment.get("to_zone") or segment.get("state"))
        with st.container(border=True):
            st.markdown(f"**{index}. {label}**")
            st.caption(f"{format_time(start)} – {format_time(end)}")
            st.button(
                f"View around {format_time(start)}",
                key=f"{key_prefix}_segment_{index}",
                on_click=jump_to_video,
                args=(start,),
            )


def query_key(question: str) -> str:
    normalized = question.lower()
    if "before" in normalized:
        return "before"
    if "when" in normalized or "move" in normalized:
        return "when"
    return "where"


def read_query_result(manifest_path: Path, rebuild: dict[str, Any], key: str) -> dict[str, Any] | None:
    for record in rebuild.get("query_results", []):
        if query_key(str(record.get("question", ""))) != key:
            continue
        result_path = Path(str(record.get("path", "")))
        if not result_path.is_absolute():
            result_path = manifest_path.parent / result_path
        if not result_path.is_file():
            return None
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
    return None


def render_query_answer(result: dict[str, Any], question_key: str) -> None:
    status = str(result.get("status", ""))
    event_status = str(result.get("target_event_status", result.get("event_status", "")))
    if status == "AMBIGUOUS_TARGET":
        st.warning("Several objects of this type were found. Select the specific object in Analyze before asking.")
    elif status in {"TARGET_NOT_RESOLVED", "IDENTITY_AMBIGUOUS"}:
        st.warning("The object identity is ambiguous, so this answer is withheld until a specific history is selected or reviewed.")
    elif status == "EVENT_REQUIRES_REVIEW":
        st.info(result.get("answer", "A movement candidate needs review before it can support a history answer."))
    elif status == "INSUFFICIENT_TRUSTED_EVIDENCE":
        st.info(result.get("answer", "The available video frames have not been reviewed enough for a trusted answer."))
    elif status in {"TARGET_NOT_FOUND", "NO_TARGET_OBSERVATIONS"}:
        st.info("No matching target observations were detected in this video. This is different from an unreviewed candidate.")
    elif status == "NO_REFERENCE_EVENT" or (event_status == "NO_SUPPORTED_EVENT" and question_key == "before"):
        st.info("No reviewed movement event is available, so there is no supported earlier location to report. This does not prove the object never moved.")
    elif status in {"NO_SUPPORTED_EVENT", "NO_EVENT", "NO_SUPPORTED_MOVE"} or (
        question_key in {"when", "before"} and event_status == "NO_SUPPORTED_EVENT"
    ):
        st.info("There is no reviewed movement event for this object. Lack of a supported event does not prove that no movement occurred.")
    elif result.get("answer"):
        st.markdown(f"### {result['answer']}")
    else:
        st.info("I could not form a reliable answer from the available video evidence.")

    candidate = result.get("automatic_candidate_result")
    if candidate:
        with st.expander("Unreviewed automatic candidate", expanded=False):
            st.write(candidate.get("answer") or candidate.get("status") or "A candidate answer is available.")
            st.caption("This output is from automatic observations and is not used as a confirmed memory answer.")
    provenance = result.get("overlay_provenance") or {}
    if provenance.get("identity_provenance") == "HUMAN_CONFIRMED":
        st.caption("This answer uses an identity link that a user confirmed. The automatic identity review remains separate.")
    if provenance.get("source_identity_review_status") == "PENDING_HUMAN_REVIEW":
        st.caption("The automatic identity match still needs review; the answer keeps that status.")
    if result.get("evidence"):
        render_evidence_cards(result["evidence"], f"query_{question_key}")


def render_event_cards(events: list[dict[str, Any]], entity_id: str, key_prefix: str) -> None:
    supported = [item for item in events if entity_id in item.get("object_ids", [])]
    st.markdown("#### Automatic event candidates")
    if not supported:
        st.info("No automatic movement candidate is stored for this object. This does not prove that it never moved.")
        return
    for index, event in enumerate(supported, 1):
        event_type = event.get("event_type")
        start = event.get("start_time")
        end = event.get("end_time")
        with st.container(border=True):
            if event_type == "MOVE_OBJECT":
                st.markdown("**Movement candidate · requires evidence review**")
            elif event_type == "ZONE_CHANGE":
                st.markdown("**Observed area changed**")
                st.caption("Area estimates can vary near boundaries; this record alone does not confirm a physical move.")
            else:
                st.markdown("**Observed event**")
            before = friendly_zone(event.get("zone_before"))
            after = friendly_zone(event.get("zone_after"))
            if event.get("zone_before") or event.get("zone_after"):
                st.write(f"{before} → {after}")
            st.caption(f"{format_time(start)} – {format_time(end)}")
            refs = event.get("evidence", [])
            timestamp = next((ref.get("timestamp") for ref in refs if ref.get("timestamp") is not None), start)
            if timestamp is not None:
                st.button(
                    "View event evidence",
                    key=f"{key_prefix}_event_{index}",
                    on_click=jump_to_video,
                    args=(timestamp,),
                )


def render_active_identity_links(
    context: dict[str, Any], entities: list[dict[str, Any]], labels: dict[str, str],
    run_history: dict[str, Any], anchor_id: str,
) -> None:
    active = [
        relation for relation in run_history.get("active_relations", [])
        if anchor_id in (relation.get("entity_a"), relation.get("entity_b"))
    ]
    if not active:
        return
    st.markdown("#### Saved identity decisions")
    for index, relation in enumerate(active, 1):
        other_id = relation["entity_b"] if relation["entity_a"] == anchor_id else relation["entity_a"]
        if relation.get("relation") == "SAME_INSTANCE":
            st.markdown(
                f"**{labels.get(anchor_id, 'Object')}** is confirmed as the same object as "
                f"**{labels.get(other_id, 'another clip')}**."
            )
        else:
            st.markdown(
                f"**{labels.get(anchor_id, 'Object')}** is recorded as different from "
                f"**{labels.get(other_id, 'another clip')}**."
            )
        with st.form(f"revoke_identity_{index}_{safe_slug(str(relation.get('operation_id')))}"):
            st.checkbox("Undo this decision and rebuild the history.", key=f"ack_revoke_{index}")
            undo = st.form_submit_button("Undo identity decision")
        if undo:
            if st.session_state.get(f"ack_revoke_{index}"):
                st.session_state["selected_entity_id"] = relation["entity_a"]
                entity = find_entity(entities, relation["entity_a"])["persistent_entity"]
                handle_revoke(
                    context, relation, int(run_history.get("overlay_revision", 0)),
                    str(entity.get("class_name", "")),
                )
            else:
                st.warning("Check the box to confirm undo and rebuild.")


def render_identity_review(
    context: dict[str, Any], entities: list[dict[str, Any]], labels: dict[str, str],
    run_history: dict[str, Any], anchor_id: str | None,
) -> None:
    st.subheader("Review two video clips")
    if not anchor_id:
        st.info("Choose an object in Analyze first. Identity review will keep that object as the comparison starting point.")
        st.button("Choose an object", key="identity_choose_object", on_click=navigate_to, args=("Analyze",))
        return

    st.markdown(f"**Current object: {labels.get(anchor_id, 'Selected object')}**")
    render_active_identity_links(context, entities, labels, run_history, anchor_id)
    suggestions = identity_suggestions(anchor_id, entities)
    suggestion_ids = [item["entity_id"] for item in suggestions]
    all_ids = [str(item["persistent_entity"].get("persistent_id")) for item in entities]
    ordered_ids = suggestion_ids + [item for item in all_ids if item not in suggestion_ids and item != anchor_id]
    if suggestions:
        st.info(f"{len(suggestions)} possible clip link(s) need review. Suggestions are not confirmed identities.")
    else:
        st.caption("No direct automatic link suggestion is available. You can still compare another object from this video.")
    candidate_id = st.selectbox(
        "Choose a clip to compare",
        ordered_ids,
        index=None,
        placeholder="Select another object",
        format_func=lambda value: labels.get(str(value), str(value)),
        key="identity_candidate",
    )
    st.caption("Automatic suggestions use existing evidence to order candidates. You decide whether the clips show the same object.")
    if not candidate_id:
        return
    if candidate_id == anchor_id:
        st.error("Choose a different object to compare.")
        return

    try:
        pair = inspect_pair(
            str(context["run_dir"]), str(context["ledger"]),
            str(context["hash_manifest"] or ""), anchor_id, str(candidate_id),
        )
        pair_images = pair_evidence(context, anchor_id, str(candidate_id), int(run_history.get("overlay_revision", 0)))
    except BackendError as error:
        st.error(f"These clips could not be compared: {error}")
        return

    entity_a = pair["entity_a"]
    entity_b = pair["entity_b"]
    persistent_a = entity_a.get("persistent_entity", {})
    persistent_b = entity_b.get("persistent_entity", {})
    same_class = persistent_a.get("class_name") == persistent_b.get("class_name")
    if not same_class:
        st.warning("These clips are different object categories. Same Object is unavailable across categories.")
    conflicts = pair.get("spatial_conflicts", [])
    if conflicts:
        st.error(
            "The source contains simultaneous, spatially separated detections for these clips. "
            "The backend checks this conflict before saving a Same Object decision."
        )

    records_a = [item for item in pair_images if item.get("persistent_id") == anchor_id]
    records_b = [item for item in pair_images if item.get("persistent_id") == candidate_id]
    track_names = set()
    columns = st.columns(2, gap="large")
    for column, entity_id, details, persistent, records in (
        (columns[0], anchor_id, entity_a, persistent_a, records_a),
        (columns[1], str(candidate_id), entity_b, persistent_b, records_b),
    ):
        track_names.update(
            str(ref.get("track_id")) for ref in details.get("evidence_frame_options", [])
            if ref.get("track_id") is not None
        )
        with column:
            with st.container(border=True):
                st.markdown(f"### {labels.get(entity_id, 'Object clip')}")
                st.caption(f"Seen {format_time(persistent.get('start_time'))} – {format_time(persistent.get('end_time'))}")
                if records:
                    frame = st.selectbox(
                        "Evidence moment",
                        records,
                        format_func=lambda value: f"{format_time(value.get('timestamp'))} · frame {value.get('frame_index')}",
                        key=f"pair_frame_{safe_slug(entity_id)}_{int(run_history.get('overlay_revision', 0))}",
                    )
                    st.image(frame["evidence_path"], use_container_width=True)
                    st.caption(f"{format_time(frame.get('timestamp'))} · frame {frame.get('frame_index')}")
                    st.button(
                        "Open this moment in the video",
                        key=f"pair_jump_{safe_slug(entity_id)}",
                        on_click=jump_to_video,
                        args=(frame.get("timestamp"),),
                    )
                else:
                    st.info("No frame evidence is available for this clip.")

    candidate_records = pair.get("continuation_candidates", [])
    st.caption(
        "An automatic association hint is available. It helps prioritize review and never saves a decision."
        if candidate_records else "No direct automatic link was recorded between these two clips."
    )
    with st.expander("Advanced details for this comparison"):
        st.write(
            f"Saved detections in these clips: {entity_a.get('observation_count_in_memory', 0)} and "
            f"{entity_b.get('observation_count_in_memory', 0)}"
        )
        st.write(f"Source tracks: {', '.join(sorted(track_names)) if track_names else 'Not available'}")
        if candidate_records:
            st.dataframe(candidate_rows(candidate_records), use_container_width=True, hide_index=True)
        if conflicts:
            st.json(conflicts)
        st.caption(f"Automatic identity status: {context['summary'].get('p2a_r1_identity_validation_status', 'UNKNOWN')}")

    pair_token = f"{safe_slug(anchor_id)}_{safe_slug(str(candidate_id))}"
    with st.form(f"identity_decision_{pair_token}"):
        st.checkbox(
            "I reviewed both clips and understand this changes the effective history.",
            key=f"ack_identity_{pair_token}",
        )
        left, middle, right = st.columns(3)
        with left:
            same = st.form_submit_button("Same Object", disabled=not same_class, type="primary")
        with middle:
            different = st.form_submit_button("Different Objects")
        with right:
            unsure = st.form_submit_button("Not Sure")
    if same or different:
        if not st.session_state.get(f"ack_identity_{pair_token}"):
            st.warning("Review both clips and check the confirmation box before saving this decision.")
        elif records_a and records_b:
            handle_relation(
                context, "SAME_INSTANCE" if same else "DIFFERENT_INSTANCE",
                anchor_id, str(candidate_id), records_a[0], records_b[0],
                int(run_history.get("overlay_revision", 0)), entities,
            )
        else:
            st.error("The review cannot be saved until both clips have frame evidence.")
    elif unsure:
        append_ui_event(context["ui_log"], {
            "event": "IDENTITY_REVIEW_NOT_SURE",
            "source_run_id": context["run_id"],
            "source_video_sha256": context["video_sha256"],
            "entity_a": anchor_id,
            "entity_b": str(candidate_id),
            "overlay_revision": int(run_history.get("overlay_revision", 0)),
        })
        st.info("No identity change was saved. You can revisit this comparison later.")




def render_advanced(
    context: dict[str, Any], entities: list[dict[str, Any]], mapping_data: dict[str, Any],
    run_history: dict[str, Any], selected_id: str | None,
) -> None:
    st.subheader("Advanced details")
    st.caption("Technical identifiers, provenance, and operation records are available here.")
    with st.expander("Source run and integrity"):
        st.write(f"Run ID: {context['run_id']}")
        st.write(f"Video SHA-256: {context['video_sha256']}")
        st.write(f"Source file: {context['video_path']}")
        st.write(f"Overlay revision: {run_history.get('overlay_revision', 0)}")
        st.write(f"Automatic identity review: {context['summary'].get('p2a_r1_identity_validation_status', 'UNKNOWN')}")
        if context.get("hash_manifest"):
            st.write(f"Source hash manifest: {context['hash_manifest']}")
        if context.get("hash_manifest_generated"):
            st.warning("First-load snapshot detects later changes but does not prove earlier file integrity.")
    with st.expander("Persistent IDs, tracks, and mapping"):
        st.dataframe(entity_rows(entities), use_container_width=True, hide_index=True)
        st.json(mapping_data.get("mapping", {}))
        if selected_id:
            selected = find_entity(entities, selected_id)
            st.json({
                "selected_entity": selected["persistent_entity"],
                "source_track_ids": selected.get("source_track_ids", []),
                "continuation_candidates": selected.get("continuation_candidates", []),
                "spatial_conflicts": selected.get("spatial_conflicts", []),
            })
    with st.expander("Identity operation history"):
        st.metric("Overlay revision", int(run_history.get("overlay_revision", 0)))
        operations = run_history.get("operations", [])
        if operations:
            st.dataframe([
                {
                    "Action": item.get("action"), "Relation": item.get("relation"),
                    "Entity A": item.get("entity_a"), "Entity B": item.get("entity_b"),
                    "Operation": item.get("operation_id"), "Target operation": item.get("target_operation_id"),
                    "Revision": item.get("overlay_revision"), "Status": item.get("revocation_status"),
                    "Time": item.get("confirmation_timestamp"),
                }
                for item in reversed(operations[-50:])
            ], use_container_width=True, hide_index=True)
        else:
            st.info("No identity operations have been recorded.")
        events = read_ui_events(context["ui_log"])
        if events:
            st.dataframe([
                {
                    "Action": item.get("event"), "Status": item.get("status") or item.get("run_status"),
                    "Revision": item.get("overlay_revision"), "Source A": item.get("entity_a"),
                    "Source B": item.get("entity_b"), "Time": item.get("created_at"),
                    "Duration (s)": item.get("elapsed_seconds"),
                }
                for item in reversed(events[-50:])
            ], use_container_width=True, hide_index=True)
    with st.expander("Run manifest"):
        st.json(context["manifest"])



def clear_selected_object() -> None:
    st.session_state["selected_entity_id"] = None
    st.session_state["asked_question_key"] = None


def load_selected_run() -> None:
    requested_run = st.session_state.get("run_path_input", "")
    requested_hash = st.session_state.get("hash_path_input", "")
    previous_run = st.session_state.get("loaded_run_path", "")
    previous_hash = st.session_state.get("loaded_hash_path", "")
    if requested_run != previous_run and requested_hash == previous_hash and requested_hash in {"", DEFAULT_HASHES}:
        requested_hash = default_hashes(Path(requested_run).expanduser())
        st.session_state["hash_path_input"] = requested_hash
    st.session_state["loaded_run_path"] = requested_run
    st.session_state["loaded_hash_path"] = requested_hash
    st.session_state.pop("entity_evidence_cache", None)
    st.session_state["selected_entity_id"] = None
    st.session_state["asked_question_key"] = None
    st.session_state["main_navigation"] = "Analyze"
    st.cache_data.clear()



def main() -> None:
    """Initialize the local application and render its stateful workflow."""
    st.set_page_config(page_title="RecallVision", page_icon="🔎", layout="wide")
    st.markdown(
        "<style>"
        ".block-container{max-width:1440px;padding-top:1.25rem;padding-bottom:3rem;}"
        ".rv-tagline{font-size:1.04rem;color:var(--text-color);margin-top:-.65rem;margin-bottom:1.2rem;}"
        "div[data-testid='stMetric']{background:var(--secondary-background-color);border:1px solid rgba(250,250,250,.14);border-radius:14px;padding:.75rem 1rem;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("RecallVision")
    st.markdown(
        '<div class="rv-tagline">Find an object in your video, follow its history, and ask what happened.</div>',
        unsafe_allow_html=True,
    )

    initial_run = default_run()
    st.session_state.setdefault("run_path_input", str(initial_run))
    st.session_state.setdefault("hash_path_input", default_hashes(initial_run))
    st.session_state.setdefault("loaded_run_path", str(initial_run))
    st.session_state.setdefault("loaded_hash_path", default_hashes(initial_run))
    st.session_state.setdefault("main_navigation", "Analyze")
    st.session_state.setdefault("selected_entity_id", None)
    st.session_state.setdefault("video_start_time", 0.0)
    st.session_state.setdefault("analyze_media_view", "Original video")
    st.session_state.setdefault("asked_question_key", None)

    with st.sidebar:
        st.markdown("### Video")
        saved_manifests = discover_run_manifests(PROJECT_ROOT / "outputs")
        if saved_manifests:
            saved_options = [""] + [str(path) for path in saved_manifests]
            st.selectbox(
                "Saved app analyses", saved_options,
                format_func=lambda value: "Choose a saved run…" if not value else _saved_run_label(Path(value)),
                key="saved_run_choice",
            )
            st.button("Open saved analysis", key="open_saved_analysis", on_click=load_saved_run, disabled=not st.session_state.get("saved_run_choice"))
        else:
            st.caption("Saved analyses from this app will appear here.")
        st.caption("Opening a saved run never changes its original observations or events.")
        with st.expander("Open another analyzed video", expanded=False):
            with st.form("load_run_form"):
                st.text_input("Completed run directory", key="run_path_input")
                st.text_input("Optional source hash manifest", key="hash_path_input")
                st.form_submit_button("Load video", on_click=load_selected_run)

    feedback = st.session_state.get("review_feedback")
    if feedback:
        getattr(st, feedback["level"])(feedback["message"])
    try:
        context = make_context(st.session_state["loaded_run_path"], st.session_state["loaded_hash_path"])
        ready, runtime = check_backend()
        if not ready:
            st.error("The memory service is unavailable. Check the local analysis environment in Advanced.")
            with st.expander("Technical details"):
                st.code(runtime)
            st.stop()
        entity_data = candidates(
            str(context["run_dir"]), str(context["ledger"]), str(context["hash_manifest"] or ""),
        )
        entities = entity_data.get("entities", [])
        if not entities:
            st.warning("This video has no saved object candidates to explore.")
            st.markdown("### Start with another local video")
            render_new_video_import()
            st.stop()
        run_history = history(
            str(context["run_dir"]), str(context["ledger"]), str(context["hash_manifest"] or ""),
        )
        mapping_data = mapping(
            str(context["run_dir"]), str(context["ledger"]), str(context["hash_manifest"] or ""),
        )
    except BackendError as error:
        st.error("The selected analysis could not be opened.")
        with st.expander("Technical details"):
            st.code(f"{error.status}: {error}")
            if error.details:
                st.json(error.details)
        st.markdown("### Start with a local video")
        render_new_video_import()
        st.stop()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        st.error("The selected analysis is incomplete or unreadable.")
        with st.expander("Technical details"):
            st.exception(error)
        st.stop()

    classes = sorted({
        str(item["persistent_entity"].get("class_name"))
        for item in entities if item["persistent_entity"].get("class_name")
    })
    if not classes:
        st.warning("This run contains no supported object categories.")
        st.stop()
    if st.session_state.get("target_class") not in classes:
        st.session_state["target_class"] = "cup" if "cup" in classes else classes[0]
    labels = entity_display_labels(entities)
    all_ids = [str(item["persistent_entity"].get("persistent_id")) for item in entities]
    selected_id = st.session_state.get("selected_entity_id")
    if selected_id not in all_ids:
        selected_id = None
        st.session_state["selected_entity_id"] = None
        st.session_state["asked_question_key"] = None
    selected_item = find_entity(entities, selected_id) if selected_id else None
    selected_class = (
        str(selected_item["persistent_entity"].get("class_name"))
        if selected_item else str(st.session_state["target_class"])
    )
    selected_label = labels.get(selected_id, "No object selected") if selected_id else "No object selected"
    metadata = context["video"].get("metadata", {})
    video_name = Path(context["video_path"]).name
    current_revision = int(run_history.get("overlay_revision", 0))
    st.caption(
        f"Loaded video: {video_name} · {format_time(metadata.get('duration'))} · "
        f"Overlay revision {current_revision}"
    )
    if selected_id:
        st.success(f"Current object: {selected_label}")
    else:
        st.info("Choose an object in Analyze to carry the same selection into Explore, Ask, and identity review.")

    pages = ["Analyze", "Explore", "Ask", "Review identity", "Advanced"]
    pending_navigation = st.session_state.pop("pending_navigation", None)
    if pending_navigation in pages:
        st.session_state["main_navigation"] = pending_navigation
    st.radio("Main navigation", pages, horizontal=True, key="main_navigation", label_visibility="collapsed")
    page = st.session_state["main_navigation"]

    if page == "Analyze":
        st.subheader("Analyze video and choose an object")
        render_new_video_import()
        st.caption("Saved detections are candidate observations. Review exact frames in Explore before using them as trusted evidence.")
        selected_class = st.selectbox(
            "Object category", classes, key="target_class", on_change=clear_selected_object,
        )
        class_entities = [
            item for item in entities
            if item["persistent_entity"].get("class_name") == selected_class
        ]
        class_ids = [str(item["persistent_entity"].get("persistent_id")) for item in class_entities]
        if selected_id and selected_id not in class_ids:
            st.session_state["selected_entity_id"] = None
            st.session_state["asked_question_key"] = None
            selected_id = None
        if selected_id:
            try:
                counts = selected_trust_counts(context, selected_id, mapping_data)
                st.caption(
                    f"Exact-frame evidence · {counts['SUPPORTED']} reviewed as supported · "
                    f"{counts['UNCERTAIN']} uncertain or unreviewed · {counts['MISSING']} reviewed as missing."
                )
                st.caption("MISSING means a specific saved detection was rejected; it does not label undetected frames.")
            except (TrustReviewError, OverlayError, OSError, ValueError) as error:
                st.warning(f"Trust status is unavailable for this object: {error}")
        video_col, objects_col = st.columns([2.15, 1], gap="large")
        with video_col:
            st.markdown("#### Video")
            st.radio(
                "View",
                ["Original video", "Detection evidence"],
                horizontal=True,
                key="analyze_media_view",
                label_visibility="collapsed",
            )
            if st.session_state["analyze_media_view"] == "Original video":
                st.video(
                    context["video_path"].read_bytes(),
                    start_time=float(st.session_state.get("video_start_time", 0.0)),
                )
                st.caption("Use the player controls to play, pause, and seek.")
            elif selected_id:
                try:
                    render_evidence_cards(entity_evidence(context, selected_id), "analyze_evidence", limit=4)
                except BackendError as error:
                    st.info("Frame evidence could not be loaded for this object.")
                    with st.expander("Technical details"):
                        st.code(f"{error.status}: {error}")
            else:
                st.info("Choose an object to view its detection evidence.")

        with objects_col:
            st.markdown(f"#### Objects found · {len(class_entities)}")
            st.caption("Select one object to explore its saved history.")
            for item in class_entities:
                entity_id = str(item["persistent_entity"]["persistent_id"])
                label = labels.get(entity_id, selected_class.title())
                try:
                    records = entity_evidence(context, entity_id)
                    thumbnail = records[0] if records else None
                except BackendError:
                    thumbnail = None
                with st.container(border=True):
                    image_col, info_col = st.columns([0.9, 1.45], vertical_alignment="center")
                    with image_col:
                        if thumbnail and Path(thumbnail["evidence_path"]).is_file():
                            st.image(thumbnail["evidence_path"], use_container_width=True)
                        else:
                            st.caption("No preview")
                    with info_col:
                        st.markdown(f"**{label.split(' · ')[0]}**")
                        persistent = item["persistent_entity"]
                        st.caption(
                            f"{format_time(persistent.get('start_time'))} – "
                            f"{format_time(persistent.get('end_time'))}"
                        )
                        provenance = mapping_data.get("mapping", {}).get(entity_id, {})
                        if provenance.get("identity_provenance") == "HUMAN_CONFIRMED":
                            st.caption("History includes a user-confirmed link")
                        elif context["summary"].get("p2a_r1_identity_validation_status") == "PENDING_HUMAN_REVIEW":
                            st.caption("Identity may need review")
                        if entity_id == selected_id:
                            st.caption("Selected")
                        if st.button(
                            "Select object" if entity_id != selected_id else "Selected object",
                            key=f"select_entity_{safe_slug(entity_id)}",
                            disabled=entity_id == selected_id,
                            type="primary" if entity_id == selected_id else "secondary",
                        ):
                            st.session_state["selected_entity_id"] = entity_id
                            st.session_state["asked_question_key"] = None
                            st.rerun()
            if len(class_entities) > 1 and not selected_id:
                st.info("Several objects match this category. Select the one you want to follow.")

    elif page == "Explore":
        st.subheader("Explore object history")
        if not selected_id:
            st.info("Choose an object in Analyze to view its history.")
            st.button("Choose an object", key="explore_choose_object", on_click=navigate_to, args=("Analyze",))
        else:
            try:
                counts = selected_trust_counts(context, selected_id, mapping_data)
                st.caption(
                    f"Evidence review · {counts['SUPPORTED']} supported · {counts['UNCERTAIN']} uncertain or unreviewed · "
                    f"{counts['MISSING']} explicitly rejected observations."
                )
                st.caption("These counts apply to saved detections only; no state is inferred for frames without detections.")
            except (TrustReviewError, OverlayError, OSError, ValueError) as error:
                st.warning(f"Trust status is unavailable for this object: {error}")
            mapping_entry = mapping_data.get("mapping", {}).get(selected_id, {})
            provenance = mapping_entry.get("identity_provenance", "AUTOMATIC_SOURCE")
            if provenance == "HUMAN_CONFIRMED":
                st.info("This history includes a user-confirmed identity link. Automatic identity review remains separate.")
            elif context["summary"].get("p2a_r1_identity_validation_status") == "PENDING_HUMAN_REVIEW":
                st.warning("Automatic matching may split one real object into separate histories. Review a suggested link if the clips appear continuous.")
            else:
                st.caption("This history follows the automatic identity mapping for this video.")

            manifest_path, rebuild, effective_id = current_rebuild(
                context, mapping_data, current_revision, selected_id,
            )
            if rebuild and manifest_path:
                timelines = json_list(manifest_path.parent / "p2b/object_state_timelines.json")
                events = (
                    json_list(manifest_path.parent / "p2c/zone_change_events.json")
                    + json_list(manifest_path.parent / "p2d/move_object_events.json")
                )
                if rebuild.get("trust_review"):
                    st.caption(
                        f"Showing {rebuild['trust_review'].get('trusted_observation_count', 0)} explicitly supported observations · "
                        f"trust review revision {rebuild['trust_review'].get('overlay_revision')} · unreviewed frames stay uncertain."
                    )
                else:
                    st.caption(f"History refreshed for the current identity revision ({current_revision}).")
            elif current_revision == 0:
                timelines = json_list(context["run_dir"] / "p2b/object_state_timelines.json")
                events = (
                    json_list(context["run_dir"] / "p2c/zone_change_events.json")
                    + json_list(context["run_dir"] / "p2d/move_object_events.json")
                )
                effective_id = selected_id
                st.warning("Automatic history candidate · observations have not been reviewed as facts. Check exact frames below; the system does not fill gaps.")
            else:
                timelines = []
                events = []
                st.warning("Identity decisions changed. Refresh this object's history before using earlier results.")

            selected_timeline = next(
                (item for item in timelines if item.get("persistent_id") == effective_id),
                None,
            )
            render_timeline(
                selected_timeline.get("segments", []) if selected_timeline else [],
                f"explore_{safe_slug(selected_id)}",
            )
            render_event_cards(events, effective_id, f"explore_{safe_slug(selected_id)}")
            render_trust_review(context, selected_id, selected_class, mapping_data, selected_label)

            unresolved = [
                item for item in identity_suggestions(selected_id, entities)
                if any(not record.get("accepted", False) for record in item["records"])
            ]
            if unresolved:
                st.warning(f"There are {len(unresolved)} possible identity link(s) to review.")
            else:
                st.caption("No unresolved direct identity suggestion is available.")
            st.button(
                "Review identity links" if unresolved else "Review identity",
                key="open_identity_review",
                type="primary" if unresolved else "secondary",
                on_click=navigate_to,
                args=("Review identity",),
            )

            if not rebuild and current_revision > 0:
                if st.button("Refresh this object's history", key="refresh_history"):
                    try:
                        result = rebuild_current(context, selected_id, selected_class, "explore")
                        st.cache_data.clear()
                        set_feedback("success", f"History refreshed in {result['elapsed_seconds']:.1f} seconds.")
                        st.rerun()
                    except BackendError as error:
                        set_feedback("error", f"History could not be refreshed: {error}")
                        st.rerun()

    elif page == "Ask":
        st.subheader("Ask about this object")
        if not selected_id:
            st.info("Choose an object in Analyze first. Questions will use that same selection.")
            st.button("Choose an object", key="ask_choose_object", on_click=navigate_to, args=("Analyze",))
        else:
            st.markdown(f"#### {selected_label}")
            prompts = [
                ("where", "Where was this object last seen?"),
                ("when", "When did this object move?"),
                ("before", "Where was it before moving?"),
            ]
            columns = st.columns(3)
            for column, (key, label) in zip(columns, prompts):
                with column:
                    if st.button(label, key=f"ask_{key}", use_container_width=True):
                        st.session_state["asked_question_key"] = key
                        manifest_path, rebuild, _ = current_rebuild(
                            context, mapping_data, current_revision, selected_id,
                        )
                        if not rebuild:
                            try:
                                rebuilt = rebuild_current(context, selected_id, selected_class, "ask")
                                st.cache_data.clear()
                                set_feedback("success", f"History checked in {rebuilt['elapsed_seconds']:.1f} seconds.")
                            except BackendError as error:
                                set_feedback("error", f"I could not check this video: {error}")
                        st.rerun()

            asked = st.session_state.get("asked_question_key")
            manifest_path, rebuild, effective_id = current_rebuild(
                context, mapping_data, current_revision, selected_id,
            )
            if asked and rebuild and manifest_path:
                result = read_query_result(manifest_path, rebuild, asked)
                if result:
                    prompt = next(label for key, label in prompts if key == asked)
                    st.markdown(f"#### {prompt}")
                    render_query_answer(result, asked)
                else:
                    st.info("No saved answer is available for this question. Ask again to refresh this object's history.")
            elif asked:
                st.info("The current history is not ready yet. Select the question again to build answers for this object.")
            else:
                st.caption("Choose a question to check this object's saved video evidence.")

    elif page == "Review identity":
        st.subheader("Review identity")
        st.caption("Compare video evidence and decide whether two clips belong to the same object.")
        render_identity_review(context, entities, labels, run_history, selected_id)

    else:
        render_advanced(context, entities, mapping_data, run_history, selected_id)


if __name__ == "__main__":
    main()
