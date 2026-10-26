"""Integration test for query CLI memory reload without perception imports."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def test_query_cli_reads_memory_and_writes_query_result(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/p1a_observations.jsonl")
    memory_path = tmp_path / "observations.jsonl"
    metadata_path = tmp_path / "run_metadata.json"
    shutil.copyfile(fixture, memory_path)
    metadata_path.write_text(
        json.dumps(
            {
                "video_id": "fixture_video_001",
                "confidence_threshold": 0.25,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recallvision.cli.query",
            "--memory",
            str(memory_path),
            "--question",
            "Where is the bottle?",
            "--metadata",
            str(metadata_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    query_result = json.loads((tmp_path / "query_result.json").read_text(encoding="utf-8"))
    assert query_result["status"] == "OK"
    assert query_result["answer"] == "The bottle is in the right region."
    assert query_result["frame_index"] == 240
    assert query_result["evidence_image"] is None
    assert "ultralytics" not in result.stdout
