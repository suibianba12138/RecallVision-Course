"""Ensure core query imports do not load optional perception libraries."""

import subprocess
import sys


def test_core_query_imports_do_not_import_optional_perception_dependencies() -> None:
    code = """
import sys
import recallvision.memory.store
import recallvision.query.object_location
forbidden = {"ultralytics", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("CORE_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "CORE_IMPORT_PASS"
