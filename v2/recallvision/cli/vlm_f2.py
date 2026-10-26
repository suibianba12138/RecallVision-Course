"""Ingest one external VLM interpretation against an immutable VLM-F1 package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from recallvision.vlm_f2.errors import VlmF2ValidationError
from recallvision.vlm_f2.ingest import ingest_interpretation, write_sidecar


def run(
    *,
    f1_package: Path,
    interpretation_path: Path,
    output_root: Path,
    identity_assertion_path: Path | None = None,
) -> dict[str, Any]:
    """Write F2 sidecars under ``output_root/<video>/<safe-event-id>``."""

    source = _load_object(interpretation_path, "interpretation")
    assertion = (
        _load_object(identity_assertion_path, "identity assertion")
        if identity_assertion_path is not None
        else None
    )
    result = ingest_interpretation(
        f1_package,
        source,
        identity_assertion=assertion,
    )
    validated = result["validated_agent_interpretation"]
    output_dir = output_root / validated["video_id"] / _safe_event_id(validated["move_object_event_id"])
    written = write_sidecar(result, output_dir)
    return {
        "f1_package": str(f1_package),
        "package_sha256": validated["package_sha256"],
        "video_id": validated["video_id"],
        "move_object_event_id": validated["move_object_event_id"],
        "output_dir": written["sidecar_dir"],
        "source_interpretation_path": written["source_interpretation_path"],
        "validated_agent_interpretation_path": written["validated_agent_interpretation_path"],
        "validation_report_path": written["validation_report_path"],
        "sidecar_sha256": written["sidecar_sha256"],
        "attribution_ready": written["attribution_ready"],
        "vlm_inference": False,
        "provider_dependency": False,
        "event_mutation": False,
        "person_ids_populated": False,
        "who_moved_object_implemented": False,
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VlmF2ValidationError(f"unable to load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise VlmF2ValidationError(f"{label} must be a JSON object")
    return value


def _safe_event_id(event_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", event_id).strip("_")
    if not safe:
        raise VlmF2ValidationError("event_id cannot produce a stable output directory")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-package", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path, required=True)
    parser.add_argument("--identity-assertion", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            f1_package=arguments.f1_package,
            interpretation_path=arguments.interpretation,
            identity_assertion_path=arguments.identity_assertion,
            output_root=arguments.output_root,
        )
    except VlmF2ValidationError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
