"""P3C CLI artifact wiring tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recallvision.cli.p3c import run


ROOT = Path(__file__).resolve().parents[1]
F3_ROOT = ROOT / "outputs/vlm_f3"
ENTITIES = ROOT / "outputs/p2a_r1/pixabay_bottle/persistent_entities.json"


@pytest.mark.skipif(not F3_ROOT.is_dir() or not ENTITIES.is_file(), reason="development F3/P2A artifacts unavailable")
def test_p3c_cli_writes_expected_success_artifact(tmp_path: Path) -> None:
    first = run(
        f3_root=F3_ROOT,
        persistent_entities_path=ENTITIES,
        question="Who moved the bottle?",
        video_id="p1b_pixabay_bottle",
        target_entity_id="obj:bottle:0001",
        output_dir=tmp_path,
    )
    second = run(
        f3_root=F3_ROOT,
        persistent_entities_path=ENTITIES,
        question="Who moved the bottle?",
        video_id="p1b_pixabay_bottle",
        target_entity_id="obj:bottle:0001",
        output_dir=tmp_path,
    )

    assert first == second
    assert first["status"] == "SUCCESS"
    assert first["query_type"] == "WHO_MOVED_OBJECT"
    assert first["resolved_event_id"] == "move-object:p1b_pixabay_bottle:obj:bottle:0001:1->2"
    assert first["person_ids"] == ["obj:person:0001"]
    assert first["frame_interval"] == {"start_frame": 92, "end_frame": 99}
    assert first["query_contract"] == {
        "attributed_event_is_authoritative": True,
        "agent_neutral_p2d_used_for_person_answer": False,
        "f1_used_at_query_time": False,
        "f2_used_at_query_time": False,
        "f3_authorization_run_at_query_time": False,
        "vlm_used_at_query_time": False,
        "pose_used_at_query_time": False,
        "raw_video_used_at_query_time": False,
        "identity_reasoning_run_at_query_time": False,
        "brief_reason_parsed": False,
        "authorized_person_ids_read_directly": True,
        "source_event_mutated": False,
    }
    output_path = tmp_path / "who_moved_object.json"
    assert output_path.is_file()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["output_artifact"] == str(output_path)
    assert "output_sha256" not in written
