"""Command-line interface for one unified video run."""

from __future__ import annotations

import sys
from pathlib import Path


def test_pipeline_cli_forwards_target_selectors_and_reports_completed_run(monkeypatch, capsys) -> None:
    from recallvision.cli import pipeline

    seen = {}

    def fake_run_pipeline(**kwargs):
        seen.update(kwargs)
        return {"run_status": "COMPLETED", "run_id": "run-1", "run_dir": "/tmp/run-1"}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recallvision-pipeline",
            "--video", "/tmp/new.mp4",
            "--output-root", "/tmp/recallvision-runs",
            "--target-class", "cake",
            "--target-entity-id", "obj:cake:0002",
            "--question", "Where is the cake?",
            "--model", "/tmp/yolo11n.pt",
        ],
    )

    exit_code = pipeline.main()

    assert exit_code == 0
    assert seen["video_path"] == Path("/tmp/new.mp4")
    assert seen["target_class"] == "cake"
    assert seen["target_entity_id"] == "obj:cake:0002"
    assert seen["question"] == "Where is the cake?"
    assert seen["model_path"] == "/tmp/yolo11n.pt"
    assert '"run_status": "COMPLETED"' in capsys.readouterr().out


def test_pipeline_cli_returns_nonzero_for_partial_run(monkeypatch) -> None:
    from recallvision.cli import pipeline

    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda **_kwargs: {"run_status": "PARTIAL", "run_id": "run-2"},
    )
    monkeypatch.setattr(sys, "argv", ["recallvision-pipeline", "--video", "/tmp/in.mp4", "--output-root", "/tmp/out"])

    assert pipeline.main() == 3



def test_pipeline_cli_collects_repeated_questions(monkeypatch, capsys) -> None:
    from recallvision.cli import pipeline

    seen = {}
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda **kwargs: (seen.update(kwargs) or {"run_status": "COMPLETED", "run_id": "run-queries"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recallvision-pipeline",
            "--video", "/tmp/new.mp4",
            "--output-root", "/tmp/recallvision-runs",
            "--question", "Where is the cake?",
            "--question", "When was the cake moved?",
        ],
    )

    assert pipeline.main() == 0
    assert seen["questions"] == ["Where is the cake?", "When was the cake moved?"]
    capsys.readouterr()
