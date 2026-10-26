from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_contextual_identity_review_displays_and_rejects_known_conflict(monkeypatch, tmp_path: Path) -> None:
    project = Path("/home/nsk/RecallVision-v2-p1")
    run = project / "outputs/p1b_fresh_video/9d86f889e0fb2e3a/69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"
    monkeypatch.setenv("RECALLVISION_PROJECT_ROOT", str(project))
    monkeypatch.setenv("RECALLVISION_DEFAULT_RUN_DIR", str(run))
    monkeypatch.setenv("RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST", str(project / "outputs/p2b1_identity_overlay/source_p1b_sha256_before.sha256"))
    monkeypatch.setenv("RECALLVISION_BACKEND_PYTHON", "/home/nsk/anaconda3/envs/hsmujoco_py312/bin/python")
    monkeypatch.setenv("RECALLVISION_UI_OUTPUT_ROOT", str(tmp_path / "ui-output"))

    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=120).run()
    app.button(key="select_entity_obj_cup_0006").click().run()
    app.radio(key="main_navigation").set_value("Explore").run()
    app.button(key="open_identity_review").click().run()
    app.selectbox(key="identity_candidate").select("obj:cup:0007").run()
    assert not app.exception
    assert any("simultaneous, spatially separated detections" in item.value for item in app.error)
    assert len(app.get("imgs")) >= 2

    app.checkbox(key="ack_identity_obj_cup_0006_obj_cup_0007").check().run()
    next(button for button in app.button if button.label == "Same Object").click().run()
    assert not app.exception
    assert any("SPATIAL_CONFLICT" in item.value for item in app.error)
    app.radio(key="main_navigation").set_value("Advanced").run()
    assert any("No identity operations" in item.value for item in app.info)
    log = tmp_path / "ui-output" / "9d86f889e0fb2e3a" / "69a2aecf-1ca4-4e83-a4af-dd0783b0ea98" / "ui_interaction_log.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(event.get("status") == "SPATIAL_CONFLICT" for event in events)
