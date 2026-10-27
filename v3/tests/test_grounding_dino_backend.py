from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import pytest

MODULE_PATH = "recallvision.perception.grounding_dino_backend"


def backend_module():
    spec = importlib.util.find_spec(MODULE_PATH)
    assert spec is not None, "Grounding DINO target-conditioned backend must be implemented"
    return importlib.import_module(MODULE_PATH)


class FakeProcessor:
    load_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def __call__(self, *, images, text, return_tensors):
        self.last_prompt = text
        return {"input_ids": [[1, 2, 3]]}

    def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
        self.postprocess_kwargs = kwargs
        return [{"boxes": [[2, 3, 20, 18]], "scores": [0.8], "text_labels": ["headphones"]}]


class FakeModel:
    load_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **inputs):
        return object()


class FakeTorch:
    @staticmethod
    def is_tensor(value):
        return False

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_backend_loads_only_verified_local_assets(tmp_path, monkeypatch):
    module = backend_module()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    expected_sha = sha256(checkpoint.read_bytes()).hexdigest()
    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForZeroShotObjectDetection=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "4.52.2")

    detector = module.GroundingDinoTargetDetector(
        asset_root=tmp_path,
        revision="a" * 40,
        expected_checkpoint_sha256=expected_sha,
    )

    assert detector.device == "cpu"
    assert FakeProcessor.load_kwargs == {"local_files_only": True}
    assert FakeModel.load_kwargs == {"local_files_only": True}
    assert detector.model.eval()


def test_backend_returns_standard_records_with_frame_and_timestamp(tmp_path, monkeypatch):
    module = backend_module()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    expected_sha = sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForZeroShotObjectDetection=FakeModel,
    ))
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "4.52.2")
    detector = module.GroundingDinoTargetDetector(
        asset_root=tmp_path,
        revision="a" * 40,
        expected_checkpoint_sha256=expected_sha,
    )

    detections = detector.detect(
        SimpleNamespace(size=(32, 24)),
        video_id="clip-1",
        video_sha256="b" * 64,
        target_description="White Headphones",
        frame_index=24,
        timestamp_seconds=1.0,
    )

    record = detections[0].as_record()
    assert record["target_description"] == "White Headphones"
    assert record["bbox_xyxy"] == [2.0, 3.0, 20.0, 18.0]
    assert record["score"] == 0.8
    assert record["frame_index"] == 24
    assert record["timestamp_seconds"] == 1.0
    assert record["model_revision"] == "a" * 40
    assert detector.processor.last_prompt == "white headphones."
    assert detector.processor.postprocess_kwargs == {
        "threshold": 0.4,
        "text_threshold": 0.3,
        "target_sizes": [(24, 32)],
    }


def test_backend_allows_per_call_relaxed_threshold_without_reloading_the_model(tmp_path, monkeypatch):
    module = backend_module()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    expected_sha = sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForZeroShotObjectDetection=FakeModel,
    ))
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "4.52.2")
    detector = module.GroundingDinoTargetDetector(
        asset_root=tmp_path,
        revision="a" * 40,
        expected_checkpoint_sha256=expected_sha,
    )
    model = detector.model

    detector.detect(
        SimpleNamespace(size=(32, 24)), video_id="clip-1", video_sha256="b" * 64,
        target_description="backpack", frame_index=151, timestamp_seconds=5.035,
        box_threshold=0.2, text_threshold=0.2,
    )

    assert detector.model is model
    assert detector.processor.postprocess_kwargs == {
        "threshold": 0.2,
        "text_threshold": 0.2,
        "target_sizes": [(24, 32)],
    }


def test_backend_treats_empty_boxes_as_no_target_detections(tmp_path, monkeypatch):
    module = backend_module()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"empty candidate fixture")
    expected_sha = sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForZeroShotObjectDetection=FakeModel,
    ))
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "4.52.2")
    detector = module.GroundingDinoTargetDetector(
        asset_root=tmp_path,
        revision="a" * 40,
        expected_checkpoint_sha256=expected_sha,
    )
    detector.processor.post_process_grounded_object_detection = lambda *args, **kwargs: [{
        "boxes": [], "scores": [], "text_labels": ["backpack"],
    }]
    assert detector.detect(
        SimpleNamespace(size=(32, 24)), video_id="clip-1", video_sha256="b" * 64,
        target_description="backpack", frame_index=3, timestamp_seconds=1.5,
    ) == []


def test_backend_rejects_missing_or_mismatched_checkpoint_before_loading(tmp_path, monkeypatch):
    module = backend_module()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        module.GroundingDinoTargetDetector(
            asset_root=tmp_path,
            revision="a" * 40,
            expected_checkpoint_sha256="a" * 64,
        )

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        module.GroundingDinoTargetDetector(
            asset_root=tmp_path,
            revision="a" * 40,
            expected_checkpoint_sha256="a" * 64,
        )
