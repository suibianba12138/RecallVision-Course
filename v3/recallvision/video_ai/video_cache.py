from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Any


class VideoAnalysisCache:
    """Content-addressed local cache for source frames and model analysis JSON."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.frames_root = self.root / "frames"
        self.analysis_root = self.root / "analysis"
        self.metadata_root = self.root / "metadata"

    @staticmethod
    def _digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _safe_video_key(video_hash: str) -> str:
        return hashlib.sha256(video_hash.encode("utf-8")).hexdigest()

    def _atomic_write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def get_metadata(self, video_hash: str) -> dict[str, Any] | None:
        path = self.metadata_root / f"{self._safe_video_key(video_hash)}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put_metadata(self, video_hash: str, metadata: dict[str, Any]) -> None:
        path = self.metadata_root / f"{self._safe_video_key(video_hash)}.json"
        self._atomic_write(path, (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"))

    def _analysis_path(self, video_hash: str, route: str, query: str, model_fingerprint: str, config: dict[str, Any] | None) -> Path:
        normalized_query = re.sub(r"\s+", " ", query).strip().casefold()
        key = self._digest({
            "video_hash": video_hash,
            "route": route,
            "query": normalized_query,
            "model_fingerprint": model_fingerprint,
            "config": config or {},
        })
        return self.analysis_root / self._safe_video_key(video_hash) / f"{key}.json"

    def get_analysis(
        self, video_hash: str, route: str, query: str, model_fingerprint: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        path = self._analysis_path(video_hash, route, query, model_fingerprint, config)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        result = value.get("result")
        return result if isinstance(result, dict) else value

    def put_analysis(
        self,
        video_hash: str,
        route: str,
        query: str,
        model_fingerprint: str,
        result: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        path = self._analysis_path(video_hash, route, query, model_fingerprint, config)
        envelope = {
            "video_hash": video_hash,
            "route": route,
            "query": re.sub(r"\s+", " ", query).strip(),
            "model_fingerprint": model_fingerprint,
            "config": config or {},
            "result": result,
        }
        payload = (json.dumps(envelope, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        self._atomic_write(path, payload)

    def list_analyses(self, video_hash: str) -> list[dict[str, Any]]:
        directory = self.analysis_root / self._safe_video_key(video_hash)
        records: list[dict[str, Any]] = []
        if not directory.is_dir():
            return records
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("video_hash") == video_hash:
                records.append(value)
        return records

    def get_or_create_frame(self, video_hash: str, frame_index: int, decode: Callable[[], bytes]) -> bytes:
        video_key = self._safe_video_key(video_hash)
        path = self.frames_root / video_key / f"{int(frame_index):010d}.jpg"
        try:
            return path.read_bytes()
        except FileNotFoundError:
            data = decode()
            if not isinstance(data, bytes) or not data:
                raise ValueError("Frame decoder must return non-empty bytes")
            self._atomic_write(path, data)
            return data

    def get_or_create_frames(
        self, video_hash: str, frame_indices: list[int],
        decode_many: Callable[[list[int]], dict[int, bytes]],
    ) -> dict[int, bytes]:
        """Reuse cached JPEGs and decode only misses through one caller-owned decoder."""
        video_key = self._safe_video_key(video_hash)
        indices = sorted({int(index) for index in frame_indices})
        if any(index < 0 for index in indices):
            raise ValueError("frame indices must be nonnegative")
        paths = {
            index: self.frames_root / video_key / f"{index:010d}.jpg"
            for index in indices
        }
        frames: dict[int, bytes] = {}
        missing: list[int] = []
        for index, path in paths.items():
            try:
                frames[index] = path.read_bytes()
            except FileNotFoundError:
                missing.append(index)
        if missing:
            decoded = decode_many(missing)
            if not isinstance(decoded, dict):
                raise ValueError("Batch frame decoder must return a frame-index to JPEG mapping")
            for index in missing:
                data = decoded.get(index)
                if not isinstance(data, bytes) or not data:
                    raise ValueError(f"Batch frame decoder returned no JPEG bytes for frame {index}")
                self._atomic_write(paths[index], data)
                frames[index] = data
        return frames

