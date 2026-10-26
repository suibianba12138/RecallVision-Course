from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def _luminance(color: str) -> float:
    components = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in components
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_streamlit_theme_uses_dark_surfaces_with_readable_text() -> None:
    config = tomllib.loads((PROJECT_ROOT / ".streamlit/config.toml").read_text())
    theme = config["theme"]

    assert theme["base"] == "dark"
    for surface in (theme["backgroundColor"], theme["secondaryBackgroundColor"]):
        assert _contrast_ratio(theme["textColor"], surface) >= 4.5
    assert _contrast_ratio(theme["textColor"], theme["primaryColor"]) >= 4.5


def test_app_css_does_not_override_dark_theme_with_light_surfaces() -> None:
    source = (PROJECT_ROOT / "recallvision/apps/identity_review.py").read_text()

    assert "background:#f6f7f9" not in source
    assert "background:#ffffff" not in source
    assert "background:#fff" not in source
    assert "color:#17233a" not in source
    assert "color:#5a6577" not in source
    assert "color:var(--text-color)" in re.sub(r"\s+", "", source)
    assert "background:var(--secondary-background-color)" in re.sub(r"\s+", "", source)
