from __future__ import annotations

import json
from pathlib import Path

import pytest


def _configure(monkeypatch, tmp_path: Path, project: Path) -> None:
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


def test_streamlit_app_loads_real_p1b_candidates_without_running_perception(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    project = Path("/home/nsk/RecallVision-v2-p1")
    _configure(monkeypatch, tmp_path, project)
    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=90).run()

    assert not app.exception
    assert any(title.value == "RecallVision" for title in app.title)
    assert any("candidate observations" in item.value for item in app.caption)
    assert len(app.get("file_uploader")) == 1
    assert app.radio(key="main_navigation").options == ["Analyze", "Explore", "Ask", "Review identity", "Advanced"]
    app.selectbox(key="target_class").select("cup").run()
    assert len([button for button in app.button if button.label == "Select object"]) == 7
    assert all("obj:cup:" not in item.value for item in app.markdown)


def test_app_loads_independent_pipeline_run_and_asks_about_person(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    project = Path("/home/nsk/RecallVision-v2-p1")
    run = project / "outputs/p2b2_interactive_review/independent_runs/2d8e01c0fcbfac89/41d7358c-d49a-45c4-a361-cae10aa3b074"
    _configure(monkeypatch, tmp_path, project)
    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=120).run()
    app.text_input(key="run_path_input").set_value(str(run)).run()
    app.button(key="FormSubmitter:load_run_form-Load video").click().run()

    assert not app.exception
    assert "person" in app.selectbox(key="target_class").options
    app.selectbox(key="target_class").select("person").run()
    app.button(key="select_entity_obj_person_0001").click().run()
    assert app.session_state["selected_entity_id"] == "obj:person:0001"
    assert len(app.get("imgs")) >= 1
    assert app.text_input(key="hash_path_input").value == ""

    app.radio(key="main_navigation").set_value("Advanced").run()
    assert any("First-load snapshot" in item.value for item in app.warning)
    app.radio(key="main_navigation").set_value("Ask").run()
    app.button(key="ask_when").click().run()
    assert not app.exception
    assert any("not enough explicitly reviewed" in item.value for item in app.info)

    output = tmp_path / "ui-output" / "2d8e01c0fcbfac89" / "41d7358c-d49a-45c4-a361-cae10aa3b074"
    manifests = list(output.glob("rebuilds/**/rebuild_manifest.json"))
    assert len(manifests) == 1
    rebuilt = json.loads(manifests[0].read_text())
    assert rebuilt["run_status"] == "COMPLETED"
    assert rebuilt["identity_review_status"] == "PENDING_HUMAN_REVIEW"
    results = [json.loads(Path(item["path"]).read_text()) for item in rebuilt["query_results"]]
    assert [item["status"] for item in results] == [
        "INSUFFICIENT_TRUSTED_EVIDENCE", "INSUFFICIENT_TRUSTED_EVIDENCE", "INSUFFICIENT_TRUSTED_EVIDENCE",
    ]
    assert all(item["identity_review_status"] == "PENDING_HUMAN_REVIEW" for item in results)


def test_app_without_saved_default_still_exposes_local_video_import(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    project = Path("/home/nsk/RecallVision-v2-p1")
    _configure(monkeypatch, tmp_path, project)
    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()
    monkeypatch.setenv("RECALLVISION_PROJECT_ROOT", str(empty_project))
    monkeypatch.setenv("RECALLVISION_DEFAULT_RUN_DIR", str(empty_project / "no-such-run"))
    monkeypatch.setenv("RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST", "")
    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=90).run()

    assert not app.exception
    assert any("Start with a local video" in item.value for item in app.markdown)
    assert len(app.get("file_uploader")) == 1
