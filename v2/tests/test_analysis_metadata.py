"""Tests for analysis summary metadata generated without a model."""

from recallvision.memory.schema import BoundingBox, Observation
from recallvision.perception.analysis import summarize_observations


def make_observation(
    *,
    class_name: str,
    timestamp: float,
    track_id: str,
) -> Observation:
    return Observation(
        video_id="video-1",
        frame_index=int(timestamp * 10),
        timestamp=timestamp,
        track_id=track_id,
        persistent_id=f"track:{track_id}",
        entity_type="object",
        class_name=class_name,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        confidence=0.8,
        zone="center_region",
    )


def test_analysis_summary_records_bottle_counts_and_time_range() -> None:
    summary = summarize_observations(
        [
            make_observation(class_name="bottle", timestamp=1.0, track_id="7"),
            make_observation(class_name="book", timestamp=2.0, track_id="8"),
            make_observation(class_name="bottle", timestamp=3.0, track_id="7"),
        ]
    )

    assert summary["total_observations"] == 3
    assert summary["observations_by_class"] == {"bottle": 2, "book": 1}
    assert summary["bottle_observation_count"] == 2
    assert summary["unique_bottle_track_ids"] == ["7"]
    assert summary["first_bottle_timestamp"] == 1.0
    assert summary["last_bottle_timestamp"] == 3.0


def test_p1b_run_succeeds_for_non_bottle_observations_without_fixed_source_url(
    monkeypatch, tmp_path
) -> None:
    import json

    import recallvision.cli.analyze as analyze

    video = tmp_path / "cake.mp4"
    video.write_bytes(b"synthetic video bytes")
    cake = make_observation(class_name="cake", timestamp=1.0, track_id="cake-track-1")
    seen = {}

    class FakeBackend:
        model_weights_path = None

        def __init__(self, **_kwargs):
            pass

        def process_video(self, _video_path, *, video_id):
            seen["video_id"] = video_id
            from dataclasses import replace

            yield replace(cake, video_id=video_id)

    monkeypatch.setattr(analyze, "UltralyticsPerceptionBackend", FakeBackend)
    monkeypatch.setattr(analyze, "read_video_metadata", lambda _path: {
        "fps": 10.0, "width": 20, "height": 20, "frame_count": 10, "duration": 1.0
    })
    monkeypatch.setattr(analyze, "sha256_file", lambda _path: "abc123")
    monkeypatch.setattr(analyze.importlib.metadata, "version", lambda _name: "test-version")
    monkeypatch.setattr(analyze, "_default_device", lambda: "cpu")
    output_dir = tmp_path / "run"

    exit_code = analyze.run(
        video_path=video,
        output_dir=output_dir,
        model_name=str(video),
        emit_json=False,
    )

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    summary = json.loads((output_dir / "analysis_summary.json").read_text())
    assert exit_code == 0
    assert seen["video_id"] == "sha256:abc123"
    assert metadata["video_source_url"] is None
    assert metadata["perception_status"] == "PASS_OBSERVATIONS_WRITTEN"
    assert summary["observations_by_class"] == {"cake": 1}


def test_p1b_refuses_to_append_to_existing_run_artifacts(monkeypatch, tmp_path) -> None:
    import recallvision.cli.analyze as analyze

    video = tmp_path / "cake.mp4"
    video.write_bytes(b"synthetic video bytes")
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    observations = output_dir / "observations.jsonl"
    observations.write_text("old-run-record\n", encoding="utf-8")
    monkeypatch.setattr(analyze, "read_video_metadata", lambda _path: {
        "fps": 10.0, "width": 20, "height": 20, "frame_count": 10, "duration": 1.0
    })
    monkeypatch.setattr(analyze, "sha256_file", lambda _path: "abc123")
    monkeypatch.setattr(analyze, "_default_device", lambda: "cpu")
    monkeypatch.setattr(analyze, "UltralyticsPerceptionBackend", lambda **_kwargs: None)

    try:
        analyze.run(video_path=video, output_dir=output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected existing run artifacts to be protected")

    assert observations.read_text(encoding="utf-8") == "old-run-record\n"
