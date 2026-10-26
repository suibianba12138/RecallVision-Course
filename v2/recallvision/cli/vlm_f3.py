"""Authorize a validated VLM-F2 interpretation into a derived MOVE_OBJECT event."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from recallvision.vlm_f3.authorization import authorize_attribution, write_artifacts
from recallvision.vlm_f3.errors import VlmF3InputError
from recallvision.vlm_f3.schema import AUTHORIZED


def run(
    *,
    source_events_path: Path,
    event_id: str,
    f1_package: Path,
    f2_sidecar_path: Path,
    f2_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    result = authorize_attribution(
        source_events_path=source_events_path,
        event_id=event_id,
        f1_package=f1_package,
        f2_sidecar_path=f2_sidecar_path,
        f2_report_path=f2_report_path,
    )
    output_dir = (
        output_root
        / _safe_video_id(result["provenance"]["video_id"])
        / _safe_event_id(result["authorization_record"]["source_event_id"])
    )
    written = write_artifacts(result, output_dir)
    return {
        "authorization_status": result["authorization_status"],
        "authorization_reason": result["authorization_reason"],
        "output_dir": written["output_dir"],
        "derived_event_path": written["derived_event_path"],
        "authorization_report_path": written["authorization_report_path"],
        "source_manifest_path": written["source_manifest_path"],
        "derived_event_sha256": written["derived_event_sha256"],
        "authorization_report_sha256": written["authorization_report_sha256"],
        "source_manifest_sha256": written["source_manifest_sha256"],
        "vlm_inference": False,
        "provider_dependency_added": False,
        "source_event_mutated": False,
        "who_moved_object_implemented": False,
    }


def _safe_video_id(video_id: str) -> str:
    if not isinstance(video_id, str) or not video_id:
        raise VlmF3InputError("bound video_id is not usable for an output directory")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("._")
    if not safe or safe in {".", ".."}:
        raise VlmF3InputError("bound video_id cannot produce a safe output directory")
    return safe


def _safe_event_id(event_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", event_id).strip("._")
    if not safe or safe in {".", ".."}:
        raise VlmF3InputError("event_id cannot produce a stable output directory")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--f1-package", type=Path, required=True)
    parser.add_argument("--f2-sidecar", type=Path, required=True)
    parser.add_argument("--f2-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            source_events_path=arguments.source_events,
            event_id=arguments.event_id,
            f1_package=arguments.f1_package,
            f2_sidecar_path=arguments.f2_sidecar,
            f2_report_path=arguments.f2_report,
            output_root=arguments.output_root,
        )
    except VlmF3InputError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["authorization_status"] == AUTHORIZED else 2


if __name__ == "__main__":
    raise SystemExit(main())
