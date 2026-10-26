"""Artifact-only P3B WHERE_OBJECT_BEFORE CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from recallvision.cli.p3b import run


EVENT_ID = "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).parents[1]
    return (
        root / "outputs/p2d_r1/pixabay_bottle/move_object_events.json",
        root / "outputs/p2b/pixabay_bottle/object_state_timelines.json",
    )


def test_p3b_cli_answers_implicit_and_explicit_development_queries(tmp_path: Path) -> None:
    events_path, timelines_path = _paths()
    implicit = run(
        events_path=events_path,
        timelines_path=timelines_path,
        target_class="bottle",
        output_dir=tmp_path / "implicit",
    )
    explicit = run(
        events_path=events_path,
        timelines_path=timelines_path,
        target_class="bottle",
        reference_event_id=EVENT_ID,
        output_dir=tmp_path / "explicit",
    )

    assert implicit["status"] == "OK"
    assert implicit["persistent_id"] == "obj:bottle:0001"
    assert implicit["reference_event_id"] == EVENT_ID
    assert implicit["from_source_segment_id"] == (
        "p1b_pixabay_bottle#obj:bottle:0001#segment=1"
    )
    assert implicit["zone"] == "left_region"
    assert implicit["last_observed_frame"] == 92
    assert implicit["last_observed_timestamp"] == 3.069733333333333
    assert implicit == explicit
    assert "was last observed" in implicit["answer"]
    assert json.loads((tmp_path / "implicit/query_result.json").read_text()) == implicit


def test_p3b_cli_imports_no_perception_or_p2d_evaluator() -> None:
    import subprocess
    import sys

    code = """
import sys
import recallvision.cli.p3b
forbidden = {"ultralytics", "torch", "cv2", "recallvision.events.movement_evidence"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("P3B_CLI_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "P3B_CLI_IMPORT_PASS"
