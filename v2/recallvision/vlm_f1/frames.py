"""Raw source-frame extraction and exact canonical reference cropping."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recallvision.memory.schema import BoundingBox

from .errors import VlmF1InputError


@dataclass(frozen=True)
class VideoInfo:
    """Source-video metadata needed for frame/time validation."""

    fps: float
    width: int
    height: int
    frame_count: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video dimensions must be positive")
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("frame_count must be positive or None")


def probe_video(video_path: Path) -> VideoInfo:
    """Read source metadata with ffprobe, without loading a model/backend."""

    if not video_path.is_file():
        raise VlmF1InputError(f"raw video is unavailable: {video_path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
        raw_count = stream.get("nb_frames")
        frame_count = int(raw_count) if raw_count not in (None, "N/A") else None
        return VideoInfo(
            fps=fps,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_count=frame_count,
        )
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
        raise VlmF1InputError(f"unable to probe raw video: {video_path}: {error}") from error


def extract_raw_frame(video_path: Path, frame_index: int, destination: Path) -> None:
    """Decode one unannotated source frame into a deterministic JPEG file."""

    if frame_index < 0:
        raise VlmF1InputError("frame_index must be non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise VlmF1InputError(f"unable to extract raw frame {frame_index}: {error}") from error
    if not destination.is_file() or destination.stat().st_size == 0:
        raise VlmF1InputError(f"raw source frame {frame_index} was not produced")


def crop_reference_card(
    source_frame: Path,
    bbox: BoundingBox,
    video_info: VideoInfo,
    destination: Path,
) -> dict[str, Any]:
    """Crop exactly the canonical bbox, clipping only to image boundaries."""

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment-specific dependency
        raise VlmF1InputError("Pillow is required only for reference-card cropping") from error

    x1 = max(0, min(video_info.width, math.floor(bbox.x1)))
    y1 = max(0, min(video_info.height, math.floor(bbox.y1)))
    x2 = max(0, min(video_info.width, math.ceil(bbox.x2)))
    y2 = max(0, min(video_info.height, math.ceil(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        raise VlmF1InputError("canonical reference bbox is empty after clipping")
    try:
        with Image.open(source_frame) as image:
            if image.size != (video_info.width, video_info.height):
                raise VlmF1InputError(
                    f"source frame dimensions {image.size} do not match probed video "
                    f"{video_info.width}x{video_info.height}"
                )
            card = image.convert("RGB").crop((x1, y1, x2, y2))
            destination.parent.mkdir(parents=True, exist_ok=True)
            card.save(destination, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
    except VlmF1InputError:
        raise
    except OSError as error:
        raise VlmF1InputError(f"unable to create reference card: {error}") from error
    return {
        "clipped": bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > video_info.width or bbox.y2 > video_info.height,
        "crop_bbox": [x1, y1, x2, y2],
        "rounding_policy": "floor_x1_y1_ceil_x2_y2",
    }
