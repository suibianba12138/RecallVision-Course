from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from test_identity_overlay import make_run


def _add_move_event(run: Path) -> dict[str, object]:
    observations_path = run / "p2a_r1" / "observations_consolidated.jsonl"
    rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    manifest_path = run / "run_manifest.json"
    video_id = rows[0]["video_id"]
    event_dir = run / "p2d"
    event_dir.mkdir(exist_ok=True)
    event_path = event_dir / "move_object_events.json"
    event = {
        "event_id": "move-object:d2:cup:10->11",
        "event_type": "MOVE_OBJECT",
        "start_time": 1.0,
        "end_time": 1.1,
        "person_ids": [],
        "object_ids": ["obj:cup:0001"],
        "zone_before": "center_region",
        "zone_after": "center_region",
        "evidence": [
            {"video_id": video_id, "timestamp": 1.0, "frame_indices": [10]},
            {"video_id": video_id, "timestamp": 1.1, "frame_indices": [11]},
        ],
        "source_segment_ids": [],
    }
    event_path.write_text(json.dumps([event]))
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"].append({"name": "P2D", "outputs": [{
        "path": str(event_path.resolve()),
        "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }]})
    manifest_path.write_text(json.dumps(manifest))
    return event


def _trust(run: Path, tmp_path: Path):
    from recallvision.memory.trust_review import TrustReviewOverlay

    return TrustReviewOverlay(
        run,
        tmp_path / "trust" / "reviews.jsonl",
        identity_ledger_path=tmp_path / "identity" / "operations.jsonl",
    )


def test_event_review_cannot_override_boundary_observation_marked_missing(tmp_path: Path) -> None:
    from recallvision.memory.trust_review import TrustReviewError

    run = make_run(tmp_path)
    event = _add_move_event(run)
    trust = _trust(run, tmp_path)
    refs = trust.observation_candidates("obj:cup:0001")
    by_frame = {item["frame_index"]: item for item in refs}
    missing = by_frame[10]
    trust.record_observation_review(
        missing["raw_observation_id"], "MISSING",
        reviewed_frame_indices=[10], reason="Boundary frame does not show the target.",
        expected_revision=0,
    )

    with pytest.raises(TrustReviewError) as rejected:
        trust.record_event_review(
            str(event["event_id"]), "SUPPORTED",
            reviewed_observation_ids=[item["raw_observation_id"] for item in refs],
            reviewed_frame_indices=[10, 11], reason="Checked both event boundaries.",
            expected_revision=1,
        )

    assert rejected.value.status == "EVENT_BOUNDARY_OBSERVATION_REJECTED"
    assert rejected.value.details["observation_id"] == missing["raw_observation_id"]


def test_later_missing_boundary_blocks_previously_supported_event(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    event = _add_move_event(run)
    trust = _trust(run, tmp_path)
    refs = trust.observation_candidates("obj:cup:0001")
    by_frame = {item["frame_index"]: item for item in refs}
    approved = trust.record_event_review(
        str(event["event_id"]), "SUPPORTED",
        reviewed_observation_ids=[item["raw_observation_id"] for item in refs],
        reviewed_frame_indices=[10, 11], reason="The checked positions differ.",
        expected_revision=0,
    )
    trust.record_observation_review(
        by_frame[10]["raw_observation_id"], "MISSING",
        reviewed_frame_indices=[10], reason="Recheck shows this is a false detection.",
        expected_revision=1,
    )

    snapshot = trust.snapshot()
    decision = snapshot["events"][str(event["event_id"])]
    assert decision["review_status"] == "SUPPORTED"
    assert decision["status"] == "BLOCKED_BY_MISSING_OBSERVATION"
    assert decision["evidence_eligibility"] == "BLOCKED"
    assert by_frame[10]["raw_observation_id"] not in snapshot["supported_observation_ids"]
    assert approved["provenance"] == "HUMAN_REVIEW"

    from recallvision.pipeline.identity_overlay_rebuild import _reviewed_move_events
    assert _reviewed_move_events(
        trust_overlay=trust,
        trust_snapshot=snapshot,
        identity_mapping=trust.identity.snapshot()["effective_mapping"],
    ) == []



def test_event_review_is_invalidated_after_identity_relation_is_revoked(tmp_path: Path) -> None:
    from recallvision.memory.identity_overlay import IdentityOverlay
    from recallvision.memory.trust_review import TrustReviewOverlay

    run = make_run(tmp_path)
    event = _add_move_event(run)
    identity_path = tmp_path / "identity" / "operations.jsonl"
    identity = IdentityOverlay(run, identity_path)
    relation = identity.record_relation(
        "obj:cup:0001", "obj:cup:0002", "SAME_INSTANCE",
        evidence_refs=[
            {"video_id": identity.video_id, "frame_index": 11},
            {"video_id": identity.video_id, "frame_index": 20},
        ],
        expected_revision=0,
    )
    trust_path = tmp_path / "trust" / "reviews.jsonl"
    trust = TrustReviewOverlay(run, trust_path, identity_ledger_path=identity_path)
    refs = trust.observation_candidates("obj:cup:0001")
    trust.record_event_review(
        str(event["event_id"]), "SUPPORTED",
        reviewed_observation_ids=[item["raw_observation_id"] for item in refs],
        reviewed_frame_indices=[10, 11], reason="Reviewed under the merged identity.",
        expected_revision=0,
    )

    identity.revoke(relation["operation_id"], expected_revision=1)
    reopened = TrustReviewOverlay(run, trust_path, identity_ledger_path=identity_path)
    snapshot = reopened.snapshot()
    assert snapshot["source_identity_overlay_revision"] == 2
    assert snapshot["events"] == {}
    assert snapshot["stale_operation_ids"]


def test_revoked_event_review_stays_revoked_after_reopen(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    event = _add_move_event(run)
    trust_path = tmp_path / "trust" / "reviews.jsonl"
    identity_path = tmp_path / "identity" / "operations.jsonl"
    trust = __import__("recallvision.memory.trust_review", fromlist=["TrustReviewOverlay"]).TrustReviewOverlay(
        run, trust_path, identity_ledger_path=identity_path,
    )
    refs = trust.observation_candidates("obj:cup:0001")
    approved = trust.record_event_review(
        str(event["event_id"]), "SUPPORTED",
        reviewed_observation_ids=[item["raw_observation_id"] for item in refs],
        reviewed_frame_indices=[10, 11], reason="Checked both positions.",
        expected_revision=0,
    )
    trust.revoke(approved["operation_id"], expected_revision=1)

    reopened = __import__("recallvision.memory.trust_review", fromlist=["TrustReviewOverlay"]).TrustReviewOverlay(
        run, trust_path, identity_ledger_path=identity_path,
    )
    snapshot = reopened.snapshot()
    assert snapshot["events"] == {}
    assert snapshot["supported_observation_ids"] == []
    assert [item["action"] for item in snapshot["history"]["operations"]] == [
        "EVENT_REVIEW", "REVOKE",
    ]
