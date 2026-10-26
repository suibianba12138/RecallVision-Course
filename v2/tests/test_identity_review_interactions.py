from __future__ import annotations

import json
import hashlib
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_same_object_confirmation_rebuild_and_undo_through_ui(monkeypatch, tmp_path: Path) -> None:
    project = Path("/home/nsk/RecallVision-v2-p1")
    run = project / "outputs/p1b_fresh_video/9d86f889e0fb2e3a/69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"
    monkeypatch.setenv("RECALLVISION_PROJECT_ROOT", str(project))
    monkeypatch.setenv("RECALLVISION_DEFAULT_RUN_DIR", str(run))
    monkeypatch.setenv("RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST", str(project / "outputs/p2b1_identity_overlay/source_p1b_sha256_before.sha256"))
    monkeypatch.setenv("RECALLVISION_BACKEND_PYTHON", "/home/nsk/anaconda3/envs/hsmujoco_py312/bin/python")
    monkeypatch.setenv("RECALLVISION_UI_OUTPUT_ROOT", str(tmp_path / "ui-output"))

    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=180).run()
    app.button(key="select_entity_obj_cup_0005").click().run()
    app.radio(key="main_navigation").set_value("Explore").run()
    app.button(key="open_identity_review").click().run()
    app.selectbox(key="identity_candidate").select("obj:cup:0006").run()
    assert not app.exception
    assert len(app.get("imgs")) >= 2
    assert any(button.label == "Same Object" for button in app.button)

    app.checkbox(key="ack_identity_obj_cup_0005_obj_cup_0006").check().run()
    next(button for button in app.button if button.label == "Same Object").click().run()
    assert not app.exception
    assert any("SAME_INSTANCE recorded" in item.value for item in app.success)
    assert app.radio(key="main_navigation").value == "Explore"
    assert any("user-confirmed identity link" in item.value.lower() for item in app.info)

    assert not any(button.label.startswith("View around") for button in app.button)

    # A fresh browser session avoids carrying removed identity form widgets into
    # the AppTest rerun while still exercising the visible Undo control.
    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=180).run()
    app.button(key="select_entity_obj_cup_0005").click().run()
    app.radio(key="main_navigation").set_value("Review identity").run()
    assert any("same object as" in item.value.lower() for item in app.markdown)
    assert any(box.key == "ack_revoke_1" for box in app.checkbox)
    app.checkbox(key="ack_revoke_1").check().run()
    next(button for button in app.button if button.label == "Undo identity decision").click().run()
    assert not app.exception
    assert any("Relation revoked" in item.value for item in app.success)
    assert app.radio(key="main_navigation").value == "Explore"

    root = tmp_path / "ui-output" / "9d86f889e0fb2e3a" / "69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"
    manifests = list(root.glob("rebuilds/**/rebuild_manifest.json"))
    rebuilds = [json.loads(path.read_text()) for path in manifests]
    assert len(rebuilds) == 2
    assert all(item["run_status"] == "COMPLETED" for item in rebuilds)
    relation_rebuild = next(item for item in rebuilds if item["overlay_revision"] == 1)
    revoked_rebuild = next(item for item in rebuilds if item["overlay_revision"] == 2)
    assert relation_rebuild["target_identity_provenance"]["identity_provenance"] == "HUMAN_CONFIRMED"
    assert relation_rebuild["trust_review"]["trusted_observation_count"] == 0
    assert relation_rebuild["identity_review_status"] == revoked_rebuild["identity_review_status"] == "PENDING_HUMAN_REVIEW"
    assert revoked_rebuild["target_identity_provenance"]["identity_provenance"] == "AUTOMATIC_SOURCE"
    mapping = json.loads((next(path for path in manifests if json.loads(path.read_text())["overlay_revision"] == 2).parent / "effective_identity/effective_entity_mapping.json").read_text())["mapping"]
    assert mapping["obj:cup:0005"]["effective_entity_id"] != mapping["obj:cup:0006"]["effective_entity_id"]
    ledger = [json.loads(line) for line in (root / "identity_operations.jsonl").read_text().splitlines()]
    assert [item["action"] for item in ledger] == ["RELATION", "REVOKE"]
    assert not (root / "trust_reviews.jsonl").exists()


def test_exact_frame_review_rebuilds_only_the_reviewed_observation(monkeypatch, tmp_path: Path) -> None:
    project = Path("/home/nsk/RecallVision-v2-p1")
    run = project / "outputs/p1b_fresh_video/9d86f889e0fb2e3a/69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"
    monkeypatch.setenv("RECALLVISION_PROJECT_ROOT", str(project))
    monkeypatch.setenv("RECALLVISION_DEFAULT_RUN_DIR", str(run))
    monkeypatch.setenv("RECALLVISION_DEFAULT_SOURCE_HASH_MANIFEST", str(project / "outputs/p2b1_identity_overlay/source_p1b_sha256_before.sha256"))
    monkeypatch.setenv("RECALLVISION_BACKEND_PYTHON", "/home/nsk/anaconda3/envs/hsmujoco_py312/bin/python")
    monkeypatch.setenv("RECALLVISION_UI_OUTPUT_ROOT", str(tmp_path / "ui-output"))
    source_observations = run / "p2a_r1" / "observations_consolidated.jsonl"
    source_bytes = source_observations.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_observation_count = sum(1 for line in source_bytes.splitlines() if line.strip())

    app = AppTest.from_file(str(project / "recallvision/apps/identity_review.py"), default_timeout=180).run()
    app.button(key="select_entity_obj_cup_0005").click().run()
    app.radio(key="main_navigation").set_value("Explore").run()
    assert app.text_input(key="trust_observation_reason_obj_cup_0005")
    app.text_input(key="trust_observation_reason_obj_cup_0005").set_value("The selected cup is visibly present.").run()
    app.checkbox(key="trust_observation_ack_obj_cup_0005").check().run()
    next(button for button in app.button if button.label == "Save frame review").click().run()

    assert not app.exception
    assert any("Trusted history was recomputed" in item.value for item in app.success)
    root = tmp_path / "ui-output" / "9d86f889e0fb2e3a" / "69a2aecf-1ca4-4e83-a4af-dd0783b0ea98"
    trust_ops = [json.loads(line) for line in (root / "trust_reviews.jsonl").read_text().splitlines()]
    assert len(trust_ops) == 1
    assert trust_ops[0]["action"] == "OBSERVATION_REVIEW"
    assert trust_ops[0]["trust_status"] == "SUPPORTED"
    assert trust_ops[0]["reviewed_interval"]["coverage"] == "EXACT_REVIEWED_FRAMES_ONLY"
    rebuild = json.loads(next(root.glob("rebuilds/**/rebuild_manifest.json")).read_text())
    assert rebuild["trust_review"]["trusted_observation_count"] == 1
    trusted_lines = (Path(rebuild["output_dir"]) / "effective_identity/trusted_observation_reviews.jsonl").read_text().splitlines()
    assert len(trusted_lines) == source_observation_count
    assert sum(json.loads(line)["trust_status"] == "SUPPORTED" for line in trusted_lines) == 1
    assert sum(json.loads(line)["trust_status"] == "UNCERTAIN" for line in trusted_lines) == source_observation_count - 1
    assert hashlib.sha256(source_observations.read_bytes()).hexdigest() == source_hash
