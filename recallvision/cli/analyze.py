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

VIDEO_SOURCE_URL = "https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/"


def run(
    *,
    video_path: Path,
    output_dir: Path,
    model_name: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
) -> int:
    """Run one analysis pass and write the P1B artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_video_metadata(video_path)
    video_sha256 = sha256_file(video_path)
    selected_device = device or _default_device()
    video_id = video_path.stem
    start_time = time.monotonic()
    analysis_start_time = datetime.now(timezone.utc).isoformat()
    backend = UltralyticsPerceptionBackend(
        model_name=model_name,
        tracker=tracker,
        confidence_threshold=confidence_threshold,
        imgsz=imgsz,
        device=selected_device,
    )
    memory_path = output_dir / "observations.jsonl"
    store = ObservationMemoryStore(memory_path)
    observations = []
    for observation in backend.process_video(video_path, video_id=video_id):
        store.append(observation)
        observations.append(observation)

    summary = summarize_observations(observations)
    weights_path = backend.model_weights_path
    run_metadata = {
        "formal_run": True,
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "video_id": video_id,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "video_source_url": VIDEO_SOURCE_URL,
        "video_fps": metadata["fps"],
        "video_width": metadata["width"],
        "video_height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "duration": metadata["duration"],
        "model_name": model_name,
        "model_version": importlib.metadata.version("ultralytics"),
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
            "PASS_TARGET_DETECTED"
            if summary["bottle_observation_count"]
            else "FAIL_TARGET_NOT_DETECTED"
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
    print(json.dumps({"run_metadata": run_metadata, "analysis_summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["bottle_observation_count"] else 2


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
    arguments = parser.parse_args()
    return run(
        video_path=arguments.video,
        output_dir=arguments.output,
        model_name=arguments.model,
        tracker=arguments.tracker,
        confidence_threshold=arguments.conf,
        imgsz=arguments.imgsz,
        device=arguments.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
