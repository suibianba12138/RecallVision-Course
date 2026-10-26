from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_multiple_entity_class_requires_explicit_object_selection(monkeypatch, tmp_path: Path) -> None:
    project = Path("/home/nsk/RecallVision-v2-p1")
    run = project / "outputs/p2b2_interactive_review/independent_runs/2d8e01c0fcbfac89/41d7358c-d49a-45c4-a361-cae10aa3b074"
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

    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=90).run()
    app.text_input(key="run_path_input").set_value(str(run)).run()
    app.button(key="FormSubmitter:load_run_form-Load video").click().run()
    app.selectbox(key="target_class").select("dining table").run()

    assert not app.exception
    assert sum(button.label == "Select object" for button in app.button) == 2
    assert app.session_state["selected_entity_id"] is None
    assert any("Several objects match this category" in item.value for item in app.info)

    app.radio(key="main_navigation").set_value("Explore").run()
    assert any("Choose an object in Analyze" in item.value for item in app.info)
    app.radio(key="main_navigation").set_value("Ask").run()
    assert any("Choose an object in Analyze first" in item.value for item in app.info)
    assert app.session_state["selected_entity_id"] is None

    app.radio(key="main_navigation").set_value("Analyze").run()
    app.button(key="select_entity_obj_dining_table_0001").click().run()
    assert app.session_state["selected_entity_id"] == "obj:dining table:0001"
