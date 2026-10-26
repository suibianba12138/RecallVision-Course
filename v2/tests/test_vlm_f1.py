"""Focused contract tests for the VLM-F1 evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recallvision.events.schema import EventRecord, EventType
from recallvision.memory.schema import BoundingBox, EvidenceRef, Observation
from recallvision.vlm_f1.builder import (
    VideoInfo,
    build_evidence_package,
    select_minimal_sequence,
)
from recallvision.vlm_f1.errors import VlmF1InputError, VlmOutputValidationError
from recallvision.vlm_f1.frames import crop_reference_card
from recallvision.vlm_f1.hashing import package_sha256
from recallvision.vlm_f1.validator import validate_vlm_output


VIDEO_ID = "fixture-video"
EVENT_ID = "move-object:fixture-video:obj:bottle:0001:1->2"
OBJECT_ID = "obj:bottle:0001"


def _event(event_type: EventType = EventType.MOVE_OBJECT) -> EventRecord:
    return EventRecord(
        event_id=EVENT_ID,
        event_type=event_type,
        start_time=92 / 30,
        end_time=99 / 30,
        object_ids=[OBJECT_ID],
        person_ids=[],
        evidence=[
            EvidenceRef(video_id=VIDEO_ID, timestamp=92 / 30, frame_indices=[92]),
            EvidenceRef(video_id=VIDEO_ID, timestamp=99 / 30, frame_indices=[99]),
        ],
    )


def _observation(
    *,
    frame_index: int,
    persistent_id: str,
    entity_type: str,
    timestamp: float | None = None,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 40.0),
) -> Observation:
    return Observation(
        video_id=VIDEO_ID,
        frame_index=frame_index,
        timestamp=frame_index / 30 if timestamp is None else timestamp,
        track_id=f"track-{persistent_id}-{frame_index}",
        persistent_id=persistent_id,
        entity_type=entity_type,
        class_name="person" if entity_type == "person" else "bottle",
        bbox=BoundingBox(*bbox),
        confidence=0.9,
        zone="center_region",
    )


def _diagnostic() -> dict[str, object]:
    return {
        "event_id": EVENT_ID,
        "left_support_frames": list(range(78, 92)),
        "right_support_frames": list(range(99, 114)),
        "right_support_timestamps": [113 / 30],
        "support_horizon_seconds": 0.50,
    }


def _fake_extractor(_video: Path, frame_index: int, destination: Path) -> None:
    destination.write_bytes(f"raw-frame-{frame_index}".encode("ascii"))


def _fake_cropper(
    source: Path,
    bbox: BoundingBox,
    _video_info: VideoInfo,
    destination: Path,
) -> dict[str, object]:
    destination.write_bytes(source.read_bytes() + b"|reference")
    return {
        "clipped": False,
        "crop_bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
    }


def _build_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture-video")
    objects = [
        _observation(frame_index=78, persistent_id=OBJECT_ID, entity_type="object"),
        _observation(frame_index=92, persistent_id=OBJECT_ID, entity_type="object"),
        _observation(frame_index=99, persistent_id=OBJECT_ID, entity_type="object"),
        _observation(frame_index=113, persistent_id=OBJECT_ID, entity_type="object"),
    ]
    persons = [
        _observation(frame_index=frame, persistent_id="obj:person:0001", entity_type="person")
        for frame in range(83, 114)
    ] + [
        _observation(frame_index=frame, persistent_id="obj:person:0002", entity_type="person")
        for frame in range(100, 107)
    ]
    return build_evidence_package(
        event=_event(),
        event_diagnostic=_diagnostic(),
        object_observations=objects,
        person_observations=persons,
        persistent_entity_ids={OBJECT_ID, "obj:person:0001", "obj:person:0002"},
        video_path=video,
        output_dir=tmp_path / "package",
        repository_head="base-head",
        video_info=VideoInfo(fps=30.0, width=100, height=80, frame_count=200),
        frame_extractor=_fake_extractor,
        reference_card_cropper=_fake_cropper,
    )


def test_valid_move_event_builds_closed_candidate_package(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)

    assert result.manifest["selected_frame_indices"] == [78, 85, 91, 92, 96, 99, 100, 107, 113]
    assert result.manifest["candidate_mapping"] == {
        "P1": "obj:person:0001",
        "P2": "obj:person:0002",
    }
    assert result.manifest["candidate_temporal_facts"]["P1"]["present_at_from_boundary"] is True
    assert result.manifest["candidate_temporal_facts"]["P2"]["present_at_from_boundary"] is False
    assert result.manifest["candidate_reference_cards"]["P1"]["reference_frame"] == 98
    assert result.manifest["candidate_reference_cards"]["P2"]["reference_frame"] == 103
    assert (result.root / "frames/frame_0096.jpg").read_bytes() == b"raw-frame-96"
    assert (result.root / "candidate_reference_cards/P1.jpg").exists()


def test_wrong_event_type_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(VlmF1InputError, match="MOVE_OBJECT"):
        build_evidence_package(
            event=_event(EventType.ZONE_CHANGE),
            event_diagnostic=_diagnostic(),
            object_observations=[],
            person_observations=[],
            video_path=tmp_path / "missing.mp4",
            output_dir=tmp_path / "package",
            repository_head="head",
            video_info=VideoInfo(fps=30.0, width=100, height=80, frame_count=200),
            frame_extractor=_fake_extractor,
            reference_card_cropper=_fake_cropper,
        )


def test_minimal_sequence_reproduces_frozen_development_positions() -> None:
    assert select_minimal_sequence(78, 92, 99, 113) == (
        78,
        85,
        91,
        92,
        96,
        99,
        100,
        107,
        113,
    )


def test_short_event_deduplicates_in_temporal_order() -> None:
    assert select_minimal_sequence(10, 10, 10, 10) == (10,)
    assert select_minimal_sequence(10, 11, 11, 12) == (10, 11, 12)


def test_detector_gap_keeps_raw_frame_without_interpolation(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)
    coverage = result.manifest["detector_coverage"]
    assert coverage["96"]["target_object_observation_available"] is False
    assert coverage["96"]["interpolated"] is False
    assert (result.root / "frames/frame_0096.jpg").read_bytes() == b"raw-frame-96"


def test_missing_usable_reference_is_recorded_without_fabrication(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture-video")
    invalid_person = _observation(
        frame_index=92,
        persistent_id="obj:person:0001",
        entity_type="person",
        bbox=(10.0, 10.0, 10.0, 20.0),
    )
    result = build_evidence_package(
        event=_event(),
        event_diagnostic=_diagnostic(),
        object_observations=[
            _observation(frame_index=78, persistent_id=OBJECT_ID, entity_type="object"),
            _observation(frame_index=113, persistent_id=OBJECT_ID, entity_type="object"),
        ],
        person_observations=[invalid_person],
        persistent_entity_ids={OBJECT_ID, "obj:person:0001"},
        video_path=video,
        output_dir=tmp_path / "package",
        repository_head="head",
        video_info=VideoInfo(fps=30.0, width=100, height=80, frame_count=200),
        frame_extractor=_fake_extractor,
        reference_card_cropper=_fake_cropper,
    )
    card = result.manifest["candidate_reference_cards"]["P1"]
    assert card["status"] == "MISSING"
    assert card["reference_frame"] is None
    assert not (result.root / "candidate_reference_cards/P1.jpg").exists()


def test_reference_crop_uses_exact_clipped_canonical_bbox(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source = tmp_path / "source.jpg"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(source)
    destination = tmp_path / "card.jpg"
    metadata = crop_reference_card(
        source,
        BoundingBox(-2.2, 10.2, 30.8, 90.1),
        VideoInfo(fps=30.0, width=100, height=80, frame_count=200),
        destination,
    )
    assert metadata["clipped"] is True
    assert metadata["crop_bbox"] == [0, 10, 31, 80]
    assert Image.open(destination).size == (31, 70)


def test_output_validator_enforces_closed_labels_and_evidence_frames() -> None:
    valid = {
        "status": "SUPPORTED_AGENT",
        "candidate_label": "P1",
        "interaction_supported": True,
        "causal_support": True,
        "evidence_frames": [92, 99],
        "brief_reason": "The supplied sequence supports candidate-caused displacement.",
    }
    assert validate_vlm_output(valid, candidate_labels=["P1", "P2"], supplied_frames=[92, 99]) == valid

    with pytest.raises(VlmOutputValidationError, match="candidate_label"):
        validate_vlm_output({**valid, "candidate_label": "P9"}, candidate_labels=["P1", "P2"], supplied_frames=[92, 99])
    with pytest.raises(VlmOutputValidationError, match="evidence_frames"):
        validate_vlm_output({**valid, "evidence_frames": [91]}, candidate_labels=["P1", "P2"], supplied_frames=[92, 99])
    with pytest.raises(VlmOutputValidationError, match="non-null"):
        validate_vlm_output({**valid, "status": "SUPPORTED_AGENT", "candidate_label": None}, candidate_labels=["P1"], supplied_frames=[92])
    with pytest.raises(VlmOutputValidationError, match="null"):
        validate_vlm_output({**valid, "status": "NO_SUPPORTED_AGENT", "candidate_label": "P1"}, candidate_labels=["P1"], supplied_frames=[92, 99])


def test_abstention_is_valid_without_attribution() -> None:
    result = {
        "status": "INSUFFICIENT_VISUAL_EVIDENCE",
        "candidate_label": None,
        "interaction_supported": False,
        "causal_support": False,
        "evidence_frames": [],
        "brief_reason": "The supplied sequence is insufficient.",
    }
    assert validate_vlm_output(result, candidate_labels=["P1"], supplied_frames=[92]) == result


def test_repeated_builds_have_same_package_hash(tmp_path: Path) -> None:
    first = _build_fixture(tmp_path / "first")
    second = _build_fixture(tmp_path / "second")
    assert first.manifest == second.manifest
    assert package_sha256(first.root) == package_sha256(second.root)
    assert json.loads((first.root / "manifest.json").read_text()) == first.manifest
