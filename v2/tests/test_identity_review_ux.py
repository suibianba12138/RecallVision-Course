from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _app(monkeypatch, tmp_path: Path) -> AppTest:
    project = Path("/home/nsk/RecallVision-v2-p1")
    monkeypatch.setenv("RECALLVISION_PROJECT_ROOT", str(project))
    monkeypatch.setenv(
        "RECALLVISION_DEFAULT_RUN_DIR",
        str(project / "outputs/p1b_fresh_video/9d86f889e0fb2e3a/69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"),
    )
    monkeypatch.setenv(
        "RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST",
        str(project / "outputs/p2b1_identity_overlay/source_p1b_sha256_before.sha256"),
    )
    monkeypatch.setenv("RECALLVISION_BACKEND_PYTHON", "/home/nsk/anaconda3/envs/hsmujoco_py312/bin/python")
    monkeypatch.setenv("RECALLVISION_UI_OUTPUT_ROOT", str(tmp_path / "ui-output"))
    return AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=90).run()


def test_task_navigation_shares_selected_object_and_hides_raw_ids(monkeypatch, tmp_path: Path) -> None:
    app = _app(monkeypatch, tmp_path)

    assert not app.exception
    assert any(title.value == "RecallVision" for title in app.title)
    navigation = app.radio(key="main_navigation")
    assert navigation.options == ["Analyze", "Explore", "Ask", "Review identity", "Advanced"]

    app.selectbox(key="target_class").select("cup").run()
    app.button(key="select_entity_obj_cup_0005").click().run()
    assert app.session_state["selected_entity_id"] == "obj:cup:0005"
    assert any("cup 5" in item.value.lower() for item in app.success)
    assert all("obj:cup:" not in item.value for item in app.markdown)

    app.radio(key="main_navigation").set_value("Explore").run()
    assert app.session_state["selected_entity_id"] == "obj:cup:0005"
    assert any("cup 5" in item.value.lower() for item in app.success)

    app.radio(key="main_navigation").set_value("Ask").run()
    labels = {button.label for button in app.button}
    assert "Where was this object last seen?" in labels
    assert "When did this object move?" in labels
    assert "Where was it before moving?" in labels


def test_identity_review_uses_current_object_context(monkeypatch, tmp_path: Path) -> None:
    app = _app(monkeypatch, tmp_path)
    app.selectbox(key="target_class").select("cup").run()
    app.button(key="select_entity_obj_cup_0005").click().run()
    app.radio(key="main_navigation").set_value("Explore").run()
    app.button(key="open_identity_review").click().run()
    app.selectbox(key="identity_candidate").select("obj:cup:0006").run()

    assert not app.exception
    assert app.radio(key="main_navigation").value == "Review identity"
    assert app.session_state["selected_entity_id"] == "obj:cup:0005"
    assert app.selectbox(key="identity_candidate").value == "obj:cup:0006"
    assert any("automatic suggestions" in item.value.lower() for item in app.caption)
    assert any(button.label == "Same Object" for button in app.button)
    assert any(button.label == "Different Objects" for button in app.button)
    assert any(button.label == "Not Sure" for button in app.button)
    assert all("saved detections" not in item.value.lower() for item in app.caption)

    app.button(key="FormSubmitter:identity_decision_obj_cup_0005_obj_cup_0006-Not Sure").click().run()
    assert any("No identity change was saved" in item.value for item in app.info)
    ui_log = tmp_path / "ui-output" / "9d86f889e0fb2e3a" / "69a2aecf-1ca4-4e83-a4af-dd0783b0ea98" / "ui_interaction_log.jsonl"
    assert any('"IDENTITY_REVIEW_NOT_SURE"' in line for line in ui_log.read_text().splitlines())

    app.selectbox(key="identity_candidate").select("obj:bottle:0001").run()
    assert any("different object categories" in item.value for item in app.warning)
    same_button = next(button for button in app.button if button.label == "Same Object")
    assert same_button.disabled
