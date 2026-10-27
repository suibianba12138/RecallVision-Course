#!/usr/bin/env python3
"""Verify the three original Mixkit source clips referenced by V3's demo memory pack.

This is an optional, read-only check. It does not download files to the
repository or modify the saved course memory. Network access is required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = {
    "025bb4c46ec15ccfa44dbff8375911d4450ff2a85708ff9cef8e894f3bbc1dd1":
        "https://assets.mixkit.co/njpruhts4frcepd3xnlpi1h7xor0",
    "38b6f9fee3c0e22fb7547dfeab92a391ca01476a1da6f1e3d6ed4f807e2a1e38":
        "https://assets.mixkit.co/active_storage/video_items/100072/1719941812/100072-video-720.mp4",
    "a6b0fcd0eb43be64ee5525ed5dc2ad90ec3707bacd6318472d122d5f7021e10d":
        "https://assets.mixkit.co/videos/43699/43699-720.mp4",
}
PACK = Path(__file__).resolve().parents[1] / "v3" / "demo" / "course_memory_pack_v1.json"
MAX_SOURCE_SIZE = 50 * 1024 * 1024


def verify() -> None:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    expected = {entry["sha256"] for entry in pack["videos"]}
    if expected != set(SOURCES):
        raise ValueError("Media manifest differs from the course demo memory pack.")
    for digest, url in SOURCES.items():
        request = Request(url, headers={"User-Agent": "RecallVision-Course/1.0"})
        hasher = hashlib.sha256()
        count = 0
        with urlopen(request, timeout=60) as response:
            if response.headers.get_content_type() != "video/mp4":
                raise ValueError("Unexpected response content type.")
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_SOURCE_SIZE:
                    raise ValueError("Unexpected media file size.")
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"Original media hash mismatch: {url}")
        print(f"PASS {digest[:12]}  {count:,} bytes  {url}")


if __name__ == "__main__":
    verify()
