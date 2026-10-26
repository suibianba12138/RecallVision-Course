"""CLI boundary tests for VLM-F1."""

from __future__ import annotations

import subprocess
import sys


def test_vlm_f1_cli_imports_without_vlm_or_perception_runtime() -> None:
    code = """
import sys
import recallvision.cli.vlm_f1
forbidden = {"openai", "anthropic", "transformers", "vllm", "ollama", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("VLM_F1_CLI_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "VLM_F1_CLI_IMPORT_PASS"
