"""P2C observed zone-transition event contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest

from recallvision.events.schema import EventType
from recallvision.events.zone_transitions import ZoneTransitionEventBuilder
from recallvision.memory.schema import EvidenceRef
from recallvision.memory.state_timeline import ObjectStateRecord, ObjectStateTimeline


def evidence(video_id: str, timestamp: float, frame_index: int, label: str) -> EvidenceRef:
    return EvidenceRef(
        video_id=video_id,
        timestamp=timestamp,
        frame_indices=[frame_index],
        segment_ref=f"{video_id}#frame={frame_index}#{label}",
    )


def segment(
    *,
    persistent_id: str = "obj:bottle:0001",
    class_name: str = "bottle",
    video_id: str = "fixture_video_001",
    start_frame: int = 10,
    end_frame: int = 20,
    start_time: float = 1.0,
    end_time: float = 2.0,
    zone: str | None = "left_region",
    index: int = 1,
) -> ObjectStateRecord:
    return ObjectStateRecord(
        persistent_id=persistent_id,
        class_name=class_name,
        video_id=video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
        zone=zone,
        observation_count=2,
        first_evidence=evidence(video_id, start_time, start_frame, f"segment={index}:first"),
        last_evidence=evidence(video_id, end_time, end_frame, f"segment={index}:last"),
        source_track_ids=("track-1",),
        source_frame_indices=(start_frame, end_frame),
        last_confidence=0.9,
    )


def timeline(*segments: ObjectStateRecord, persistent_id: str = "obj:bottle:0001") -> ObjectStateTimeline:
    return ObjectStateTimeline(
        persistent_id=persistent_id,
        class_name="bottle",
        video_id="fixture_video_001",
        segments=tuple(segments),
    )


def test_same_zone_segments_do_not_emit_zone_change() -> None:
    events = ZoneTransitionEventBuilder().build(
        timeline(
            segment(zone="left_region", start_frame=10, end_frame=20, start_time=1.0, end_time=2.0),
            segment(zone="left_region", start_frame=21, end_frame=30, start_time=2.1, end_time=3.0, index=2),
        )
    )

    assert events == []


def test_simple_zone_change_is_bounded_by_actual_boundary_evidence() -> None:
    events = ZoneTransitionEventBuilder().build(
        timeline(
            segment(zone="left_region", start_frame=69, end_frame=92, start_time=2.3023, end_time=3.069733333333333),
            segment(zone="center_region", start_frame=99, end_frame=152, start_time=3.3033, end_time=5.071733333333333, index=2),
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type is EventType.ZONE_CHANGE
    assert event.object_ids == ["obj:bottle:0001"]
    assert event.zone_before == "left_region"
    assert event.zone_after == "center_region"
    assert event.start_time == 3.069733333333333
    assert event.end_time == 3.3033
    assert [item.frame_indices for item in event.evidence] == [[92], [99]]
    assert event.to_record()["event_type"] == "ZONE_CHANGE"


def test_multiple_transitions_only_connect_adjacent_segments() -> None:
    events = ZoneTransitionEventBuilder().build(
        timeline(
            segment(zone="left", start_frame=1, end_frame=2, start_time=0.1, end_time=0.2),
            segment(zone="center", start_frame=3, end_frame=4, start_time=0.3, end_time=0.4, index=2),
            segment(zone="right", start_frame=5, end_frame=6, start_time=0.5, end_time=0.6, index=3),
        )
    )

    assert [(event.zone_before, event.zone_after) for event in events] == [
        ("left", "center"),
        ("center", "right"),
    ]
    assert all(event.zone_before != "left" or event.zone_after != "right" for event in events)


def test_different_persistent_entities_are_not_connected() -> None:
    first = timeline(
        segment(persistent_id="obj:bottle:0001", zone="left"),
        segment(persistent_id="obj:bottle:0001", zone="center", start_frame=21, end_frame=30, start_time=2.1, end_time=3.0, index=2),
        persistent_id="obj:bottle:0001",
    )
    second = timeline(
        segment(persistent_id="obj:bottle:0002", zone="center"),
        segment(persistent_id="obj:bottle:0002", zone="right", start_frame=21, end_frame=30, start_time=2.1, end_time=3.0, index=2),
        persistent_id="obj:bottle:0002",
    )

    events = ZoneTransitionEventBuilder().build_all([second, first])
    assert [event.object_ids for event in events] == [
        ["obj:bottle:0001"],
        ["obj:bottle:0002"],
    ]
    assert all(event.object_ids == ["obj:bottle:0001"] or event.object_ids == ["obj:bottle:0002"] for event in events)


def test_unsorted_segments_are_sorted_without_mutating_source_timeline() -> None:
    first = segment(zone="left", start_frame=1, end_frame=2, start_time=0.1, end_time=0.2)
    second = segment(zone="center", start_frame=3, end_frame=4, start_time=0.3, end_time=0.4, index=2)
    source = timeline(second, first)

    events = ZoneTransitionEventBuilder().build(source)
    assert [(event.zone_before, event.zone_after) for event in events] == [("left", "center")]
    assert source.segments == (second, first)


@pytest.mark.parametrize(
    "bad_segments,match",
    [
        (
            (
                segment(start_frame=10, end_frame=20, start_time=2.0, end_time=1.0),
            ),
            "segment temporal bounds",
        ),
        (
            (
                segment(start_frame=10, end_frame=20, start_time=1.0, end_time=2.0),
                segment(start_frame=19, end_frame=30, start_time=1.9, end_time=3.0, index=2),
            ),
            "overlap",
        ),
        (
            (
                segment(zone=None),
            ),
            "zone",
        ),
    ],
)
def test_malformed_timeline_fails_instead_of_fabricating_event(bad_segments, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ZoneTransitionEventBuilder().build(timeline(*bad_segments))


def test_missing_temporal_evidence_fails_explicitly() -> None:
    malformed = segment(zone="left")
    malformed = ObjectStateRecord(
        **{**malformed.__dict__, "last_evidence": None},  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="last_evidence"):
        ZoneTransitionEventBuilder().build(timeline(malformed))


def test_provenance_is_explicit_and_no_move_object_is_emitted() -> None:
    event = ZoneTransitionEventBuilder().build(
        timeline(
            segment(zone="left"),
            segment(zone="center", start_frame=21, end_frame=30, start_time=2.1, end_time=3.0, index=2),
        )
    )[0]

    assert event.source_segment_ids == (
        "fixture_video_001#obj:bottle:0001#segment=1",
        "fixture_video_001#obj:bottle:0001#segment=2",
    )
    assert event.evidence[0].segment_ref.endswith("segment=1:last")
    assert event.evidence[1].segment_ref.endswith("segment=2:first")
    assert event.event_type is not EventType.MOVE_OBJECT


def test_event_generation_is_deterministic() -> None:
    source = timeline(
        segment(zone="left"),
        segment(zone="center", start_frame=21, end_frame=30, start_time=2.1, end_time=3.0, index=2),
    )
    first = [event.to_record() for event in ZoneTransitionEventBuilder().build(source)]
    second = [event.to_record() for event in ZoneTransitionEventBuilder().build(source)]

    assert first == second


def test_event_core_does_not_import_perception_dependencies() -> None:
    code = """
import sys
import recallvision.events.zone_transitions
forbidden = {"ultralytics", "torch", "cv2"}
assert forbidden.isdisjoint(sys.modules), sorted(forbidden & set(sys.modules))
print("P2C_EVENT_CORE_IMPORT_PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "P2C_EVENT_CORE_IMPORT_PASS"
