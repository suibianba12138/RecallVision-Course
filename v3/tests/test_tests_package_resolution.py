from __future__ import annotations

import importlib.util
from pathlib import Path


def test_tests_helpers_resolve_from_this_checkout() -> None:
    spec = importlib.util.find_spec("tests")
    expected = Path(__file__).resolve().parent / "__init__.py"

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == expected
