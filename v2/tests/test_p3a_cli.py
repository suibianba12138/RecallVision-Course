"""Artifact-only P3A query CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from recallvision.cli.p3a import run


def test_p3a_cli_answers_development_bottle_from_frozen_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    result = run(
        events_path=root / "outputs/p2d_r1/pixabay_bottle/move_object_events.json",
        timelines_path=root / "outputs/p2b/pixabay_bottle/object_state_timelines.json",
        question="When was the bottle moved?",
        output_dir=tmp_path,
    )

    assert result["status"] == "MOVEMENT_FOUND"
    assert result["matched_event_count"] == 1
    assert result["persistent_id"] == "obj:bottle:0001"
    assert result["event_ids"] == ["move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"]
    assert result["time_intervals"] == [
        {
            "event_id": result["event_ids"][0],
            "start_time": 3.069733333333333,
            "end_time": 3.3033,
        }
    ]
    assert result["frame_intervals"] == [
        {"event_id": result["event_ids"][0], "start_frame": 92, "end_frame": 99}
    ]
    assert "3.07 s and 3.30 s" in result["answer"]
    assert "3.77" not in result["answer"]
    assert json.loads((tmp_path / "query_result.json").read_text())["status"] == "MOVEMENT_FOUND"


def test_p3a_cli_query_imports_no_perception_or_movement_detector() -> None:
    code = """
import sys
import recallvision.cli.p3a
forbidden = {"ultralytics", "torch", "cv2", "recallvision.events.movement_evidence"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("P3A_QUERY_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "P3A_QUERY_IMPORT_PASS"
