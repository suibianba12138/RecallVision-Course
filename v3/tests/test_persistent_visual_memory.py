from __future__ import annotations

import hashlib
import json
import sqlite3
import stat

import pytest

from recallvision.video_ai.persistent_memory import (
    PersistentMemoryStore,
    answer_from_memory,
    build_memory_prompt,
    normalize_memory_document,
    parse_memory_response,
    saved_answer_requires_visual_recheck,
)


def test_video_and_memories_survive_store_reopen(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"small local video fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source,
        source_name="clip.mp4",
        metadata={"duration": 8.0, "fps": 2.0},
        model_fingerprint="qwen-test",
    )
    memories = normalize_memory_document(
        {
            "scenes": [],
            "entities": [{
                "entity_id": "entity-1",
                "description": "red backpack",
                "observations": [{
                    "time_seconds": 3.0,
                    "location": "on the desk by the window",
                    "state": "resting",
                    "evidence_seconds": [3.0],
                    "confidence": "high",
                }],
            }],
            "events": [],
            "relations": [],
        },
        duration=8.0,
        model_fingerprint="qwen-test",
    )
    store.replace_memories(
        video["video_hash"], memories, raw_response="{}",
        analysis_metadata={"sampled_timestamps": [3.0], "generation_seconds": 12.0},
    )

    reopened = PersistentMemoryStore(tmp_path / "memory")
    restored_video = reopened.get_video(video["video_hash"])
    assert restored_video["analysis_status"] == "complete"
    assert restored_video["source_path"] == video["source_path"]
    assert restored_video["source_path"] != str(source)
    assert restored_video["analysis_metadata"]["sampled_timestamps"] == [3.0]
    assert reopened.list_memories(video["video_hash"])[0]["claim"] == (
        "red backpack is resting on the desk by the window"
    )


def test_local_memory_data_permissions_are_private(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "private-memory")
    private_directories = [store.root, store.videos_root, store.evidence_root]
    for directory in private_directories:
        assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(store.database_path.stat().st_mode) & 0o077 == 0


def test_normalize_memory_rejects_evidence_outside_video(tmp_path) -> None:
    document = {
        "scenes": [],
        "entities": [{
            "entity_id": "bag",
            "description": "red backpack",
            "observations": [{
                "time_seconds": 9.0,
                "location": "desk",
                "evidence_seconds": [9.0],
            }],
        }],
        "events": [],
        "relations": [],
    }
    memories = normalize_memory_document(document, duration=8.0, model_fingerprint="qwen-test")
    assert memories == []


def test_normalize_memory_binds_claims_to_an_actually_sampled_source_time() -> None:
    document = {
        "scenes": [],
        "entities": [{
            "entity_id": "bag",
            "description": "red backpack",
            "observations": [{
                "time_seconds": 1.0,
                "location": "desk",
                "state": "resting",
                "evidence_seconds": [1.0, 2.0],
                "confidence": "high",
            }],
        }],
        "events": [],
        "relations": [],
    }
    memories = normalize_memory_document(
        document, duration=8.0, model_fingerprint="qwen-test", sampled_timestamps=[0.0, 2.0, 4.0]
    )
    assert memories[0]["evidence_seconds"] == [2.0]
    assert memories[0]["start_seconds"] == 2.0


def test_memory_search_and_saved_answer_reuse(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"another local fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source,
        source_name="clip.mp4",
        metadata={"duration": 12.0, "fps": 4.0},
        model_fingerprint="qwen-test",
    )
    memories = normalize_memory_document(
        {
            "scenes": [],
            "entities": [{
                "entity_id": "bag",
                "description": "red backpack",
                "observations": [{
                    "time_seconds": 5.0,
                    "location": "on the desk",
                    "state": "resting",
                    "evidence_seconds": [5.0],
                    "confidence": "high",
                }],
            }, {
                "entity_id": "cup",
                "description": "gray cup",
                "observations": [{
                    "time_seconds": 8.0,
                    "location": "beside the keyboard",
                    "state": "resting",
                    "evidence_seconds": [8.0],
                    "confidence": "high",
                }],
            }],
            "events": [],
            "relations": [],
        },
        duration=12.0,
        model_fingerprint="qwen-test",
    )
    store.replace_memories(video["video_hash"], memories, raw_response="{}")
    match = store.search_memories(video["video_hash"], "Where was my backpack?")
    assert match[0]["label"] == "red backpack"
    assert match[0]["evidence_seconds"] == [5.0]
    assert match[0]["verification_status"] == "unverified"
    answer = answer_from_memory("Where was my backpack?", match)
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True

    store.save_answer(
        video["video_hash"],
        "Where was my backpack?",
        {"answer": "The backpack was on the desk.", "evidence_seconds": [5.0]},
        provenance={"route": "persistent_memory"},
    )
    reopened = PersistentMemoryStore(tmp_path / "memory")
    assert reopened.get_saved_answer(video["video_hash"], "where was my backpack?")["answer"] == (
        "The backpack was on the desk."
    )


def test_saved_answer_cache_is_scoped_to_evidence_verification_version(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"verification cache fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 2.0}, model_fingerprint="qwen-v1"
    )
    context_v1 = {"memory_schema": 2, "verification_config": "detector-cfg-1"}
    context_v2 = {"memory_schema": 2, "verification_config": "detector-cfg-2"}
    answer = {"answer": "candidate at 1.0 s", "verification_status": "corroborated"}
    store.save_answer(video["video_hash"], "Where is the bag?", answer,
                      provenance={"route": "memory"}, cache_context=context_v1)
    assert store.get_saved_answer(video["video_hash"], "Where is the bag?",
                                  cache_context=context_v1) == answer
    assert store.get_saved_answer(video["video_hash"], "Where is the bag?",
                                  cache_context=context_v2) is None


def test_memory_answer_abstains_when_matching_entities_are_ambiguous() -> None:
    matches = [{
        "kind": "entity_observation",
        "label": "red backpack",
        "claim": "red backpack is on the desk",
        "start_seconds": 4.0,
        "end_seconds": 4.0,
        "evidence_seconds": [4.0],
        "confidence": "high",
        "payload": {"entity_id": "left-bag"},
        "retrieval_score": 0.8,
    }, {
        "kind": "entity_observation",
        "label": "red backpack",
        "claim": "red backpack is on a chair",
        "start_seconds": 6.0,
        "end_seconds": 6.0,
        "evidence_seconds": [6.0],
        "confidence": "high",
        "payload": {"entity_id": "right-bag"},
        "retrieval_score": 0.78,
    }]
    answer = answer_from_memory("Where was the red backpack?", matches)
    assert answer["status"] == "ambiguous"
    assert answer["abstain"] is True


def test_last_seen_location_query_requires_targeted_verification() -> None:
    answer = answer_from_memory("Where did I leave my backpack?", [{
        "kind": "entity_observation", "label": "black backpack",
        "claim": "A black backpack is visible on the left side of the frame",
        "start_seconds": 0.0, "end_seconds": 10.886,
        "evidence_seconds": [0.0, 1.543, 3.128, 4.671, 6.215, 7.758, 9.343, 10.886],
        "confidence": "medium", "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }])
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True
    assert saved_answer_requires_visual_recheck("Where did I leave my backpack?", {
        "input_mode": "persistent visual memory", "answer": "The bag is on the desk.",
    }) is True
    assert saved_answer_requires_visual_recheck("Where did I leave my backpack?", {
        "input_mode": "Qwen3.5-2B", "answer": "The bag is on the desk.",
    }) is False
    assert answer_from_memory("Where was the backpack last?", [{
        "kind": "entity_observation", "label": "black backpack",
        "claim": "A black backpack is visible on the left side of the frame",
        "start_seconds": 0.0, "end_seconds": 0.0, "evidence_seconds": [0.0],
        "confidence": "high", "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }])["status"] == "insufficient"


def test_temporal_question_does_not_use_entity_observation_as_answer() -> None:
    answer = answer_from_memory("When did the red backpack move?", [{
        "kind": "entity_observation", "label": "red backpack",
        "claim": "red backpack is on the desk", "start_seconds": 3.0, "end_seconds": 3.0,
        "evidence_seconds": [3.0], "confidence": "high", "payload": {"entity_id": "bag"},
        "retrieval_score": 0.95,
    }])
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True


def test_general_location_memory_answer_does_not_claim_last_seen_time() -> None:
    answer = answer_from_memory("Where was my backpack?", [{
        "kind": "entity_observation", "label": "black backpack",
        "claim": "A black backpack is visible on the left side of the frame",
        "start_seconds": 0.0, "end_seconds": 0.0, "evidence_seconds": [0.0],
        "confidence": "medium", "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }])
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True
    assert "unverified" in answer["uncertainty"].casefold()


def test_unverified_model_claim_cannot_answer_object_location_as_a_fact() -> None:
    answer = answer_from_memory("Where was my backpack?", [{
        "kind": "entity_observation", "label": "black backpack",
        "claim": "A black backpack is visible on the left side of the frame",
        "start_seconds": 0.0, "end_seconds": 0.0, "evidence_seconds": [0.0],
        "confidence": "high", "verification_status": "unverified",
        "visibility_state": "unknown", "identity_status": "unverified",
        "uncertainty_reason": "No independent detector evidence was saved.",
        "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }])
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True
    assert "unverified" in answer["uncertainty"].casefold()


def test_object_inventory_returns_deduplicated_unverified_memory_candidates() -> None:
    answer = answer_from_memory("What objects appeared in the video?", [{
        "item_id": 1, "kind": "entity_observation", "label": "Device",
        "claim": "A black tablet is on the desk", "start_seconds": 0.0, "end_seconds": 0.0,
        "evidence_seconds": [0.0], "confidence": "high", "verification_status": "unverified",
        "visibility_state": "unknown", "identity_status": "unverified",
        "payload": {"entity_id": "tablet"}, "retrieval_score": 0.2,
    }, {
        "item_id": 2, "kind": "entity_observation", "label": "Device",
        "claim": "A black tablet is on the desk", "start_seconds": 2.0, "end_seconds": 2.0,
        "evidence_seconds": [2.0], "confidence": "high", "verification_status": "unverified",
        "visibility_state": "unknown", "identity_status": "unverified",
        "payload": {"entity_id": "tablet"}, "retrieval_score": 0.1,
    }, {
        "item_id": 3, "kind": "entity_observation", "label": "Object",
        "claim": "A blue mug appears beside the tablet", "start_seconds": 2.0, "end_seconds": 2.0,
        "evidence_seconds": [2.0], "confidence": "medium", "verification_status": "unverified",
        "visibility_state": "unknown", "identity_status": "unverified",
        "payload": {"entity_id": "mug"}, "retrieval_score": 0.1,
    }])
    assert answer["status"] == "memory_report"
    assert answer["verification_status"] == "unverified"
    assert "black tablet" in answer["answer"]
    assert "blue mug" in answer["answer"]
    assert answer["evidence_seconds"] == [0.0, 2.0]
    assert "candidate" in answer["answer"].casefold()


def test_event_summary_reports_unverified_event_candidate_with_sample_times() -> None:
    answer = answer_from_memory("What happened to the plant?", [{
        "item_id": 9, "kind": "event", "label": "planting action",
        "claim": "The person is placing the potted plant into a black pot",
        "start_seconds": 5.0, "end_seconds": 8.3, "evidence_seconds": [5.0, 6.7, 8.3],
        "confidence": "high", "verification_status": "unverified",
        "uncertainty_reason": "No event-level verification is saved.",
        "payload": {}, "retrieval_score": 0.85,
    }])
    assert answer["status"] == "memory_report"
    assert answer["verification_status"] == "unverified"
    assert "placing the potted plant" in answer["answer"]
    assert "5.00" in answer["answer"] and "8.30" in answer["answer"]
    assert answer["evidence_seconds"] == [5.0, 6.7, 8.3]
    assert "unverified" in answer["uncertainty"].casefold()


def test_general_location_memory_answer_uses_earliest_sample_for_same_entity() -> None:
    answer = answer_from_memory("Where was my backpack?", [{
        "kind": "entity_observation", "label": "red backpack", "claim": "red backpack is on the floor",
        "start_seconds": 8.0, "end_seconds": 8.0, "evidence_seconds": [8.0], "confidence": "high",
        "verification_status": "corroborated", "visibility_state": "present",
        "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }, {
        "kind": "entity_observation", "label": "red backpack", "claim": "red backpack is on the desk",
        "start_seconds": 2.0, "end_seconds": 2.0, "evidence_seconds": [2.0], "confidence": "high",
        "verification_status": "corroborated", "visibility_state": "present",
        "payload": {"entity_id": "bag"}, "retrieval_score": 0.95,
    }])
    assert answer["evidence_seconds"] == [2.0]
    assert "on the desk" in answer["answer"]


def test_memory_event_answer_requires_the_target_named_in_the_question() -> None:
    answer = answer_from_memory("When was the backpack moved?", [{
        "kind": "event", "label": "camera movement",
        "claim": "The camera pans slowly to the right", "start_seconds": 0.0,
        "end_seconds": 6.215, "evidence_seconds": [0.0, 6.215],
        "confidence": "high", "payload": {}, "retrieval_score": 0.67,
    }])
    assert answer["status"] == "insufficient"
    assert answer["abstain"] is True


def test_structured_memory_response_requires_all_collections() -> None:
    parsed = parse_memory_response(
        '```json\n{"scenes":[],"entities":[],"events":[],"relations":[]}\n```'
    )
    assert parsed["scenes"] == []
    try:
        parse_memory_response('{"entities":[]}')
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("incomplete memory JSON must be rejected")


def test_flat_memory_records_normalize_into_evidence_linked_items() -> None:
    document = parse_memory_response(
        '{"items":[{"kind":"entity_observation","entity_id":"bag",'
        '"label":"black backpack","claim":"at the left edge of the desk",'
        '"start_seconds":4.671,"end_seconds":4.671,"evidence_seconds":[4.671],'
        '"confidence":1.0},{"kind":"event","label":"camera pan",'
        '"claim":"camera moves left to right","start_seconds":0.0,"end_seconds":10.886,'
        '"evidence_seconds":[0.0,10.886],"confidence":"medium"}]}'
    )
    memories = normalize_memory_document(
        document, duration=10.928, model_fingerprint="qwen-test",
        sampled_timestamps=[0.0, 4.671, 10.886],
    )
    assert [item["kind"] for item in memories] == ["entity_observation", "event"]
    assert memories[0]["evidence_seconds"] == [4.671]
    assert memories[1]["evidence_seconds"] == [0.0, 10.886]
    assert memories[0]["confidence"] == "medium"


def test_model_top_level_array_is_normalized_as_flat_memory_items() -> None:
    parsed = parse_memory_response(
        '[{"kind":"event","label":"camera movement",'
        '"claim":"camera pans right","start_seconds":0,"end_seconds":6,'
        '"evidence_seconds":[0,6],"confidence":"high"}]'
    )
    assert parsed["items"][0]["kind"] == "event"


def test_object_description_misclassified_as_relation_is_recovered_as_entity(tmp_path) -> None:
    items = normalize_memory_document(
        {"items": [{
            "kind": "relation", "entity_id": "bag", "label": "Bag",
            "claim": "A black backpack is visible on the left side of the frame",
            "time_seconds": [0.0, 1.5], "start_seconds": 0.0, "end_seconds": 1.5,
            "evidence_seconds": [0.0, 1.5], "confidence": "high",
        }]},
        duration=5.0, model_fingerprint="qwen-test", sampled_timestamps=[0.0, 1.5],
    )
    assert len(items) == 2
    assert all(item["kind"] == "entity_observation" for item in items)
    assert [item["evidence_seconds"] for item in items] == [[0.0], [1.5]]
    assert all(item["payload"]["entity_id"] == "bag" for item in items)


def test_truncated_flat_memory_output_keeps_only_complete_items_and_marks_partial() -> None:
    parsed = parse_memory_response(
        '{"items":[{"kind":"event","label":"camera pan",'
        '"claim":"camera moves across the desk","start_seconds":0,"end_seconds":8,'
        '"evidence_seconds":[0,8],"confidence":"medium"},{"kind":"entity_observation"'
    )
    assert parsed["_partial"] is True
    assert len(parsed["items"]) == 1
    assert parsed["items"][0]["label"] == "camera pan"


def test_automatic_memory_prompt_requires_source_timed_structured_facts() -> None:
    prompt = build_memory_prompt()
    assert '"items"' in prompt
    assert "entity_observation" in prompt and "event" in prompt
    assert "scene" in prompt and "relation" in prompt
    assert "evidence_seconds" in prompt
    assert "Do not merge" in prompt
    assert "do not infer who caused" in prompt.casefold()
    assert "inspect every supplied timestamp" in prompt.casefold()
    assert "at most 4 events, 2 scenes, and 2 relations" in prompt.casefold()
    assert "at most 2 observations per entity" in prompt.casefold()
    assert "before and after time" in prompt.casefold()
    assert "timestamps printed before the images" in prompt.casefold()
    assert "minified JSON" in prompt
    assert '"start_seconds":0.0' not in prompt
    assert "include 3 to 4 distinctive entities when present" in prompt.casefold()
    assert "evidence from at least two supplied times" in prompt.casefold()


def test_invalid_replacement_keeps_previous_memory_and_model_change_marks_stale(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"persistent fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 8.0}, model_fingerprint="qwen-a"
    )
    memories = normalize_memory_document(
        {"scenes": [], "entities": [{
            "entity_id": "bag", "description": "red backpack", "observations": [{
                "time_seconds": 3.0, "location": "on desk", "evidence_seconds": [3.0], "confidence": "high"
            }]
        }], "events": [], "relations": []},
        duration=8.0, model_fingerprint="qwen-a",
    )
    store.replace_memories(video["video_hash"], memories, raw_response="complete")
    invalid = [{**memories[0], "evidence_seconds": [9.0]}]
    try:
        store.replace_memories(video["video_hash"], invalid, raw_response="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range evidence must fail")
    assert store.list_memories(video["video_hash"])[0]["claim"] == memories[0]["claim"]

    updated = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 8.0}, model_fingerprint="qwen-b"
    )
    assert updated["analysis_status"] == "pending"
    assert store.list_memories(video["video_hash"])[0]["claim"] == memories[0]["claim"]
    assert store.list_memories(video["video_hash"], model_fingerprint="qwen-b") == []
    replacement = normalize_memory_document(
        {"scenes": [], "entities": [{
            "entity_id": "bag", "description": "red backpack", "observations": [{
                "time_seconds": 4.0, "location": "on chair", "evidence_seconds": [4.0], "confidence": "medium"
            }]
        }], "events": [], "relations": []},
        duration=8.0, model_fingerprint="qwen-b",
    )
    store.replace_memories(video["video_hash"], replacement, raw_response="updated")
    restored = store.list_memories(video["video_hash"])
    assert len(restored) == 1
    assert restored[0]["claim"] == replacement[0]["claim"]
    assert restored[0]["model_fingerprint"] == "qwen-b"


def test_legacy_v1_migration_splits_sightings_and_marks_them_unverified(tmp_path) -> None:
    root = tmp_path / "legacy-memory"
    root.mkdir()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"legacy source")
    video_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    now = 1_700_000_000.0
    connection = sqlite3.connect(root / "visual_memory.sqlite3")
    connection.executescript("""
        CREATE TABLE schema_info(singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL);
        INSERT INTO schema_info(singleton, version) VALUES (1, 1);
        CREATE TABLE videos(
            video_hash TEXT PRIMARY KEY, source_name TEXT NOT NULL, source_path TEXT NOT NULL,
            metadata_json TEXT NOT NULL, model_fingerprint TEXT NOT NULL, schema_version INTEGER NOT NULL,
            analysis_metadata_json TEXT NOT NULL DEFAULT '{}', analysis_status TEXT NOT NULL DEFAULT 'pending',
            analysis_error TEXT, analysis_raw TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE memory_items(
            item_id INTEGER PRIMARY KEY AUTOINCREMENT, video_hash TEXT NOT NULL, kind TEXT NOT NULL,
            label TEXT NOT NULL, claim TEXT NOT NULL, start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
            evidence_seconds_json TEXT NOT NULL, confidence TEXT NOT NULL, model_fingerprint TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
    """)
    connection.execute(
        "INSERT INTO videos VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (video_hash, "clip.mp4", str(source), json.dumps({"duration": 4.0, "fps": 2.0}),
         "qwen-v1", 1, "{}", "complete", None, "raw-memory", now, now),
    )
    connection.execute(
        """INSERT INTO memory_items(video_hash,kind,label,claim,start_seconds,end_seconds,
           evidence_seconds_json,confidence,model_fingerprint,payload_json)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (video_hash, "entity_observation", "red backpack", "red backpack is visible",
         0.0, 1.5, "[0.0,1.5]", "high", "qwen-v1", '{"entity_id":"bag"}'),
    )
    connection.commit()
    connection.close()

    store = PersistentMemoryStore(root)
    restored = store.list_memories(video_hash)
    assert len(restored) == 2
    assert [item["evidence_seconds"] for item in restored] == [[0.0], [1.5]]
    assert all(item["start_seconds"] == item["end_seconds"] for item in restored)
    assert all(item["verification_status"] == "unverified" for item in restored)
    assert all(item["visibility_state"] == "unknown" for item in restored)
    assert all(item["identity_status"] == "unverified" for item in restored)
    assert all("legacy" in item["uncertainty_reason"].casefold() for item in restored)
    assert store.get_video(video_hash)["analysis_status"] == "complete"
    with store._db() as db:
        assert db.execute("SELECT version FROM schema_info WHERE singleton=1").fetchone()[0] == 2


def test_visual_verification_persists_exact_frame_reference_and_unknown_state(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"verification fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 3.0, "fps": 2.0}, model_fingerprint="qwen-v1"
    )
    item = normalize_memory_document({"items": [{
        "kind": "entity_observation", "entity_id": "bag", "label": "black backpack",
        "claim": "black backpack at left", "time_seconds": 1.5, "evidence_seconds": [1.5],
        "confidence": "high",
    }]}, duration=3.0, model_fingerprint="qwen-v1", sampled_timestamps=[1.5])[0]
    store.replace_memories(video["video_hash"], [item], raw_response="{}")
    item_id = store.list_memories(video["video_hash"])[0]["item_id"]
    store.update_visual_verification(item_id, {
        "verification_status": "unknown", "visibility_state": "unknown",
        "identity_status": "unverified",
        "uncertainty_reason": "No detector candidate; a miss cannot establish absence.",
        "visual_evidence": [{"frame_index": 3, "timestamp_seconds": 1.5, "detections": []}],
    }, verifier_fingerprint="grounding-dino:test", config_fingerprint="cfg-v1")
    reopened = PersistentMemoryStore(tmp_path / "memory").list_memories(video["video_hash"])[0]
    assert reopened["verification_status"] == "unknown"
    assert reopened["visibility_state"] == "unknown"
    assert reopened["identity_status"] == "unverified"
    assert reopened["visual_evidence"][0]["frame_index"] == 3
    assert reopened["verification_config_fingerprint"] == "cfg-v1"


def test_targeted_refinement_is_a_persistent_exact_frame_observation_and_idempotent(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"adaptive visibility fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 8.0}, model_fingerprint="qwen-v1"
    )
    result = {
        "verification_status": "corroborated",
        "visibility_state": "present",
        "identity_status": "unverified",
        "uncertainty_reason": "Crop review supports this exact frame; continuity is unknown.",
        "visual_evidence": [{"frame_index": 151, "timestamp_seconds": 5.035, "detections": [
            {"bbox_xyxy": [1.0, 90.0, 54.0, 480.0], "target_presence_corroborated": True}
        ]}],
    }
    kwargs = {
        "label": "black backpack", "target_description": "backpack",
        "timestamp_seconds": 5.035, "result": result, "model_fingerprint": "qwen-v1",
        "verifier_fingerprint": "dino-qwen-v1", "config_fingerprint": "adaptive-v1",
    }
    first = store.save_visual_observation(video["video_hash"], **kwargs)
    second = store.save_visual_observation(video["video_hash"], **kwargs)
    rows = PersistentMemoryStore(tmp_path / "memory").list_memories(video["video_hash"])

    assert first == second
    assert len(rows) == 1
    assert rows[0]["kind"] == "visual_observation"
    assert rows[0]["evidence_seconds"] == [5.035]
    assert rows[0]["verification_status"] == "corroborated"
    assert rows[0]["visibility_state"] == "present"
    assert rows[0]["identity_status"] == "unverified"


def test_targeted_refinement_rejects_mismatched_frame_timestamp(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"mismatched timestamp fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 8.0}, model_fingerprint="qwen-v1"
    )
    with pytest.raises(ValueError, match="match its exact-frame evidence"):
        store.save_visual_observation(video["video_hash"],
            label="backpack", target_description="backpack", timestamp_seconds=5.0,
            result={"visual_evidence": [{"timestamp_seconds": 5.2}]},
            model_fingerprint="qwen-v1", verifier_fingerprint="dino-qwen-v1",
            config_fingerprint="adaptive-v1")


def test_answer_cache_context_changes_when_visual_evidence_changes(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"evidence revision fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 2.0}, model_fingerprint="qwen-v1"
    )
    item = normalize_memory_document({"items": [{
        "kind": "entity_observation", "entity_id": "bag", "label": "black backpack",
        "claim": "black backpack at left", "time_seconds": 1.0, "evidence_seconds": [1.0],
        "confidence": "high",
    }]}, duration=2.0, model_fingerprint="qwen-v1")[0]
    store.replace_memories(video["video_hash"], [item], raw_response="{}")
    item_id = store.list_memories(video["video_hash"])[0]["item_id"]
    before = store.answer_cache_context(video["video_hash"], verification_config_fingerprint="cfg-v1")
    store.update_visual_verification(item_id, {
        "verification_status": "corroborated", "visibility_state": "present",
        "identity_status": "unverified", "visual_evidence": [{"timestamp_seconds": 1.0}],
    }, verifier_fingerprint="dino-v1", config_fingerprint="cfg-v1")
    after = store.answer_cache_context(video["video_hash"], verification_config_fingerprint="cfg-v1")
    assert before["evidence_revision"] != after["evidence_revision"]


def test_source_evidence_frame_is_durable_and_retrievable(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"frame fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 6.0}, model_fingerprint="qwen-a"
    )
    path = store.save_evidence_frame(
        video["video_hash"], frame_index=12, time_seconds=3.0, jpeg_bytes=b"jpeg-data"
    )
    reopened = PersistentMemoryStore(tmp_path / "memory")
    assert reopened.find_evidence_frame(video["video_hash"], 3.1) == path
    assert path.read_bytes() == b"jpeg-data"


def test_partial_generation_is_persisted_as_partial_not_complete(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"partial fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 5.0}, model_fingerprint="qwen-a"
    )
    items = normalize_memory_document(
        {"items": [{
            "kind": "entity_observation", "entity_id": "cup", "label": "gray cup",
            "claim": "beside keyboard", "time_seconds": 1.0,
            "evidence_seconds": [1.0], "confidence": "medium",
        }]},
        duration=5.0, model_fingerprint="qwen-a", sampled_timestamps=[1.0],
    )
    store.replace_memories(
        video["video_hash"], items, raw_response="partial",
        analysis_metadata={"partial_output": True}, complete=False,
    )
    assert store.get_video(video["video_hash"])["analysis_status"] == "partial"


def test_partial_regeneration_keeps_prior_memory_without_duplicate_claims(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"partial upgrade fixture")
    store = PersistentMemoryStore(tmp_path / "memory")
    video = store.register_video(
        source, source_name="clip.mp4", metadata={"duration": 8.0}, model_fingerprint="qwen-a"
    )
    previous = normalize_memory_document({
        "scenes": [], "entities": [{"entity_id": "bag", "description": "red bag", "observations": [{
            "time_seconds": 1.0, "location": "near the door", "evidence_seconds": [1.0],
        }]}], "events": [], "relations": [],
    }, duration=8.0, model_fingerprint="qwen-a")
    latest = normalize_memory_document({
        "items": [{"kind": "scene", "label": "entryway", "claim": "An entryway is visible.",
                   "start_seconds": 2.0, "end_seconds": 4.0, "evidence_seconds": [2.0, 4.0]}]
    }, duration=8.0, model_fingerprint="qwen-a")
    store.replace_memories(video["video_hash"], previous, raw_response="complete")
    store.replace_memories(video["video_hash"], latest, raw_response="partial", complete=False)
    store.replace_memories(video["video_hash"], latest, raw_response="partial", complete=False)
    claims = [item["claim"] for item in store.list_memories(video["video_hash"])]
    assert claims == [previous[0]["claim"], latest[0]["claim"]]
    assert store.get_video(video["video_hash"])["analysis_status"] == "partial"
