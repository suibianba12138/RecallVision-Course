"""CLI and real F1-package integration tests for VLM-F2."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "outputs/vlm_f1/p1b_pixabay_bottle/move-object_p1b_pixabay_bottle_obj_bottle_0001_1-_2"
FIXTURES = ROOT / "tests/fixtures/vlm_f2"
EXPECTED_PACKAGE_SHA256 = "74e7fdd4a937c0c3f6aae2a3583c4f760eb1b384034dde15bc307f1bf22f4503"


def test_vlm_f2_cli_imports_without_vlm_or_perception_runtime() -> None:
    code = """
import sys
import recallvision.cli.vlm_f2
forbidden = {"openai", "anthropic", "transformers", "vllm", "ollama", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("VLM_F2_CLI_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "VLM_F2_CLI_IMPORT_PASS"


@pytest.mark.skipif(
    not PACKAGE.is_dir(), reason="canonical local F1 development package is not present"
)
def test_real_f1_development_package_and_external_fixture_validate_end_to_end(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        "-m",
        "recallvision.cli.vlm_f2",
        "--f1-package",
        str(PACKAGE),
        "--interpretation",
        str(FIXTURES / "development_supported_agent.json"),
        "--identity-assertion",
        str(FIXTURES / "development_identity_assertion.json"),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = json.loads(result.stdout)

    assert output["package_sha256"] == EXPECTED_PACKAGE_SHA256
    assert output["attribution_ready"] is True
    sidecar = json.loads(Path(output["validated_agent_interpretation_path"]).read_text())
    assert sidecar["validated_status"] == "VALIDATED_SUPPORTED_AGENT"
    assert sidecar["mapped_persistent_person_id"] == "obj:person:0001"
    assert sidecar["identity_gate_status"] == "PASS_EVENT_LOCAL"


@pytest.mark.skipif(
    not PACKAGE.is_dir(), reason="canonical local F1 development package is not present"
)
def test_real_f1_control_fixture_abstains_without_identity(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        "-m",
        "recallvision.cli.vlm_f2",
        "--f1-package",
        str(PACKAGE),
        "--interpretation",
        str(FIXTURES / "control_no_supported_agent.json"),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = json.loads(result.stdout)
    sidecar = json.loads(Path(output["validated_agent_interpretation_path"]).read_text())

    assert output["attribution_ready"] is False
    assert sidecar["source_status"] == "NO_SUPPORTED_AGENT"
    assert sidecar["mapped_persistent_person_id"] is None
    assert sidecar["interaction_supported"] is True
    assert sidecar["causal_support"] is False
