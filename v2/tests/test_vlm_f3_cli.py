"""CLI boundary and real development-artifact tests for VLM-F3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SOURCE_EVENTS = ROOT / "outputs/p2d_r1/pixabay_bottle/move_object_events.json"
F1_PACKAGE = ROOT / "outputs/vlm_f1/p1b_pixabay_bottle/move-object_p1b_pixabay_bottle_obj_bottle_0001_1-_2"
F2_DIR = ROOT / "outputs/vlm_f2/p1b_pixabay_bottle/move-object_p1b_pixabay_bottle_obj_bottle_0001_1-_2"
F2_SIDECAR = F2_DIR / "validated_agent_interpretation.json"
F2_REPORT = F2_DIR / "validation_report.json"
F2_CONTROL_FIXTURE = ROOT / "tests/fixtures/vlm_f2/control_no_supported_agent.json"
EVENT_ID = "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"


def test_vlm_f3_cli_imports_without_vlm_or_perception_runtime() -> None:
    code = """
import sys
import recallvision.cli.vlm_f3
forbidden = {"openai", "anthropic", "transformers", "vllm", "ollama", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("VLM_F3_CLI_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "VLM_F3_CLI_IMPORT_PASS"


@pytest.mark.skipif(
    not all(path.exists() for path in (SOURCE_EVENTS, F1_PACKAGE, F2_SIDECAR, F2_REPORT)),
    reason="canonical local P2D/F1/F2 development artifacts are not present",
)
def test_real_development_authorization_preserves_p2d_and_writes_derived_event(tmp_path: Path) -> None:
    source_before = SOURCE_EVENTS.read_bytes()
    outputs: list[dict[str, object]] = []
    for name in ("out-a", "out-b"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "recallvision.cli.vlm_f3",
                "--source-events",
                str(SOURCE_EVENTS),
                "--event-id",
                EVENT_ID,
                "--f1-package",
                str(F1_PACKAGE),
                "--f2-sidecar",
                str(F2_SIDECAR),
                "--f2-report",
                str(F2_REPORT),
                "--output-root",
                str(tmp_path / name),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(result.stdout))

    first, second = outputs
    assert first["authorization_status"] == "AUTHORIZED"
    assert first["derived_event_sha256"] == second["derived_event_sha256"]
    assert first["authorization_report_sha256"] == second["authorization_report_sha256"]
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    derived = json.loads(Path(first["derived_event_path"]).read_text())
    source = json.loads(SOURCE_EVENTS.read_text())[0]
    assert derived["person_ids"] == ["obj:person:0001"]
    assert {key: value for key, value in derived.items() if key != "person_ids"} == {
        key: value for key, value in source.items() if key != "person_ids"
    }
    assert SOURCE_EVENTS.read_bytes() == source_before


@pytest.mark.skipif(
    not all(path.exists() for path in (SOURCE_EVENTS, F1_PACKAGE, F2_CONTROL_FIXTURE)),
    reason="canonical local P2D/F1/control fixture artifacts are not present",
)
def test_real_f2_abstention_control_cannot_authorize(tmp_path: Path) -> None:
    f2_root = tmp_path / "f2-control"
    f2_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recallvision.cli.vlm_f2",
            "--f1-package",
            str(F1_PACKAGE),
            "--interpretation",
            str(F2_CONTROL_FIXTURE),
            "--output-root",
            str(f2_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    f2_output = json.loads(f2_result.stdout)
    f3_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recallvision.cli.vlm_f3",
            "--source-events",
            str(SOURCE_EVENTS),
            "--event-id",
            EVENT_ID,
            "--f1-package",
            str(F1_PACKAGE),
            "--f2-sidecar",
            f2_output["validated_agent_interpretation_path"],
            "--f2-report",
            f2_output["validation_report_path"],
            "--output-root",
            str(tmp_path / "f3-control"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert f3_result.returncode == 2
    output = json.loads(f3_result.stdout)
    assert output["authorization_status"] == "BLOCKED_NOT_ATTRIBUTION_READY"
    assert output["derived_event_path"] is None
