"""Read basic video metadata in the configured perception environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recallvision.perception.ultralytics_backend import read_video_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Read video dimensions and timing.")
    parser.add_argument("--video", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        metadata = read_video_metadata(arguments.video)
    except Exception as error:
        print(json.dumps({"status": "VIDEO_PROBE_FAILED", "error": str(error), "details": {}}))
        return 2
    print(json.dumps({"status": "OK", "metadata": metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
