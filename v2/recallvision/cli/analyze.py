"""Run the formal P1B real-video perception analysis."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from recallvision.memory.store import ObservationMemoryStore
from recallvision.perception.analysis import summarize_observations
from recallvision.perception.ultralytics_backend import (
    UltralyticsPerceptionBackend,
    read_video_metadata,
    sha256_file,
)

def run(
    *,
    video_path: Path,
    output_dir: Path,
    model_name: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    video_id: str | None = None,
    source_url: str | None = None,
    emit_json: bool = True,
) -> int:
    """Run one analysis pass and write the P1B artifacts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write P1B artifacts into a non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_video_metadata(video_path)
    video_sha256 = sha256_file(video_path)
    selected_device = device or _default_device()
    selected_video_id = video_id or f"sha256:{video_sha256}"
    local_model_path = _resolve_local_model(model_name)
    start_time = time.monotonic()
    analysis_start_time = datetime.now(timezone.utc).isoformat()
    backend = UltralyticsPerceptionBackend(
        model_name=str(local_model_path),
        tracker=tracker,
        confidence_threshold=confidence_threshold,
        imgsz=imgsz,
        device=selected_device,
    )
    memory_path = output_dir / "observations.jsonl"
    # Create exclusively before processing so an old run can never be appended.
    with memory_path.open("x", encoding="utf-8"):
        pass
    store = ObservationMemoryStore(memory_path)
    observations = []
    for observation in backend.process_video(video_path, video_id=selected_video_id):
        store.append(observation)
        observations.append(observation)

    summary = summarize_observations(observations)
    weights_path = backend.model_weights_path or local_model_path
    run_metadata = {
        "formal_run": True,
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "video_id": selected_video_id,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "video_source_url": source_url,
        "video_fps": metadata["fps"],
        "video_width": metadata["width"],
        "video_height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "duration": metadata["duration"],
        "model_name": model_name,
        "model_version": _package_version("ultralytics"),
        "model_weights_path": str(weights_path) if weights_path else None,
        "model_weights_sha256": sha256_file(weights_path) if weights_path else None,
        "tracker": tracker,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": None,
        "imgsz": imgsz,
        "device": selected_device,
        "analysis_start_time": analysis_start_time,
        "analysis_runtime_seconds": time.monotonic() - start_time,
        "perception_status": (
            "PASS_OBSERVATIONS_WRITTEN"
            if summary["total_observations"]
            else "COMPLETED_NO_OBSERVATIONS"
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if emit_json:
        print(json.dumps({"run_metadata": run_metadata, "analysis_summary": summary}, indent=2, sort_keys=True))
    return 0


def _resolve_local_model(model_name: str) -> Path:
    requested = Path(model_name).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, Path(__file__).resolve().parents[2] / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"local model weights not found for {model_name!r}; pass --model with an existing checkpoint. "
        "Automatic model downloads are disabled."
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _git_value(*arguments: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-url", default=None)
    arguments = parser.parse_args()
    return run(
        video_path=arguments.video,
        output_dir=arguments.output,
        model_name=arguments.model,
        tracker=arguments.tracker,
        confidence_threshold=arguments.conf,
        imgsz=arguments.imgsz,
        device=arguments.device,
        source_url=arguments.source_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
