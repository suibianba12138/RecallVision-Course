from __future__ import annotations

import gc
import math
import os
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATHS = {
    "qwen": Path(os.environ.get(
        "RECALLVISION_QWEN_MODEL_PATH",
        "~/.cache/recallvision-v3/models/qwen3.5-2b",
    )).expanduser(),
    "timelens": Path(os.environ.get(
        "RECALLVISION_TIMELENS_MODEL_PATH",
        "~/.cache/recallvision-v3/models/timelens2-2b",
    )).expanduser(),
}

_MODEL: Any = None
_PROCESSOR: Any = None
_MODEL_KIND: str | None = None
_MODEL_LOCK = threading.RLock()


def _load_model(kind: str) -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR, _MODEL_KIND
    if kind not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"Unknown local VLM route: {kind}")
    model_path = DEFAULT_MODEL_PATHS[kind]
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"{kind} model is not installed at {model_path}. See docs/LOCAL_VIDEO_MEMORY.md."
        )

    with _MODEL_LOCK:
        if _MODEL_KIND == kind and _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR

        if _MODEL is not None:
            del _MODEL
            _MODEL = None
            _PROCESSOR = None
            _MODEL_KIND = None
            gc.collect()

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        thread_count = max(1, min(8, int(os.environ.get("RECALLVISION_VLM_THREADS", "8"))))
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(min(4, thread_count))
        except RuntimeError:
            pass

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.eval()
        model._recallvision_load_seconds = time.perf_counter() - started
        _MODEL, _PROCESSOR, _MODEL_KIND = model, processor, kind
        return _MODEL, _PROCESSOR


def _video_inputs(processor: Any, video_path: Path, prompt: str, fps: float) -> Any:
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path.resolve().as_uri(),
                "fps": fps,
                "min_pixels": 32 * 32,
                "max_pixels": 480 * 480,
                "total_pixels": 128000 * 32 * 32,
            },
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos:
        decoded_videos, metadata = zip(*videos)
        decoded_videos, metadata = list(decoded_videos), list(metadata)
    else:
        decoded_videos, metadata = None, None
    inputs = processor(
        text=[text],
        images=images,
        videos=decoded_videos,
        video_metadata=metadata,
        do_resize=False,
        cap_pixels_per_frame=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs.to("cpu"), text


def ask_video(
    video_path: str | Path,
    question: str,
    *,
    route: str = "qwen",
    fps: float = 1.0,
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    """Run a direct video question or temporal-grounding request on CPU."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not question.strip():
        raise ValueError("Question must not be empty")
    if route not in {"qwen", "timelens"}:
        raise ValueError(f"Unknown local VLM route: {route}")

    if route == "qwen":
        prompt = (
            "Answer the user's question using only visual evidence in this video. "
            "The video contains timestamps. Do not infer intent, identity, or causation "
            "that is not visibly supported. If several similar objects could be the target, "
            "say so and abstain from choosing one. For movement, distinguish object motion "
            "from camera motion. Return one JSON object with exactly these keys: "
            '"answer" (short natural-language answer), '
            '"evidence_seconds" (array of seconds or [start,end] ranges), '
            '"confidence" ("high", "medium", or "low"), '
            '"abstain" (boolean), and "uncertainty" (short string or empty). '
            "Cite only times that you can locate in the sampled video; use a coarse range "
            "when exact onset is unclear. If there is no direct support, set abstain=true.\n"
            f"Question: {question.strip()}"
        )
    else:
        prompt = (
            f'Given the query: "{question.strip()}", return all time spans (in seconds) '
            "where the described visual evidence is relevant. Only include intervals visibly "
            "supported by the video; return [] if no matching evidence is visible. "
            "Output format MUST be a JSON array of [start, end] pairs."
        )

    with _MODEL_LOCK:
        model, processor = _load_model(route)
        inputs, input_text = _video_inputs(processor, video_path, prompt, fps)
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        response = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "route": route,
            "model": str(DEFAULT_MODEL_PATHS[route]),
            "question": question,
            "raw_response": response.strip(),
            "generation_seconds": time.perf_counter() - started,
            "model_load_seconds": getattr(model, "_recallvision_load_seconds", None),
            "fps": fps,
            "prompt_characters": len(input_text),
        }


def analyze_video_memory(
    video_path: str | Path,
    *,
    fps: float = 1.0,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Generate a compact, timestamped scene/entity/event memory in one local Qwen pass."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    from recallvision.video_ai.persistent_memory import build_memory_prompt

    prompt = build_memory_prompt()
    with _MODEL_LOCK:
        model, processor = _load_model("qwen")
        inputs, input_text = _video_inputs(processor, video_path, prompt, fps)
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        response = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "route": "automatic_memory_generation",
            "model": str(DEFAULT_MODEL_PATHS["qwen"]),
            "model_fingerprint": "qwen3.5-2b:aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
            "raw_response": response.strip(),
            "generation_seconds": time.perf_counter() - started,
            "model_load_seconds": getattr(model, "_recallvision_load_seconds", None),
            "fps": fps,
            "prompt_characters": len(input_text),
        }


def ask_frames(
    frames: list[tuple[Any, float]],
    question: str,
    *,
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    """Answer a final-state question from ordered source frames with timestamps."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if not frames:
        raise ValueError("At least one source frame is required")

    latest_timestamp = frames[-1][1]
    prompt = (
        "These are time-ordered source frames from the final seconds of one video. The "
        f"latest frame is at {latest_timestamp:.3f} seconds. Follow visual continuity "
        "across the frames, including partially occluded objects. Answer the requested "
        "visible location at the latest frame and include the object's state: carried or "
        "held by a person, resting on a surface, or not visible. If it is being carried, "
        "say so explicitly; do not describe it as resting on the floor. If it is not visible "
        "the latest frame, state that current location is unsupported and give only the "
        "last observed state. Return both a concise scene location and the object's "
        "physical state in separate fields; the answer sentence must include both, for "
        "example: 'In the hallway, still carried by a person.' If either part is unknown, "
        "say unknown for that part. Cite only supplied source-frame times that support "
        "your answer. Return one JSON object with exactly keys "
        '"answer", "location", "object_state", "evidence_seconds" (an array of supplied timestamps), '
        '"confidence" ("high", "medium", or "low"), "abstain" (boolean), and '
        '"uncertainty" (short string or empty).\n'
        f"Question: {question.strip()}"
    )
    content: list[dict[str, Any]] = []
    for image, timestamp in frames:
        content.extend([
            {"type": "text", "text": f"Source frame timestamp: {timestamp:.3f} seconds."},
            {"type": "image", "image": image},
        ])
    content.append({"type": "text", "text": prompt})
    messages = [{
        "role": "user",
        "content": content,
    }]

    from qwen_vl_utils import process_vision_info

    with _MODEL_LOCK:
        model, processor = _load_model("qwen")
        input_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, _videos, _video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
        )
        inputs = processor(text=[input_text], images=images, return_tensors="pt").to("cpu")
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        response = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "route": "qwen_final_frame",
            "model": str(DEFAULT_MODEL_PATHS["qwen"]),
            "question": question,
            "raw_response": response.strip(),
            "generation_seconds": time.perf_counter() - started,
            "model_load_seconds": getattr(model, "_recallvision_load_seconds", None),
            "source_timestamps": [timestamp for _, timestamp in frames],
            "prompt_characters": len(input_text),
        }


def _ask_timestamped_frames(
    frames: list[tuple[Any, float]], question: str, *, route_name: str, prompt: str,
    max_new_tokens: int, frame_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty")
    if not frames:
        raise ValueError("At least one source frame is required")
    if frame_ids is not None and (len(frame_ids) != len(frames) or any(not value for value in frame_ids)):
        raise ValueError("frame_ids must contain one nonempty identifier per frame")
    content: list[dict[str, Any]] = []
    for index, (image, timestamp) in enumerate(frames):
        identifier = f" Candidate crop ID: {frame_ids[index]}." if frame_ids is not None else ""
        content.extend([
            {"type": "text", "text": f"Source frame timestamp: {timestamp:.3f} seconds.{identifier}"},
            {"type": "image", "image": image},
        ])
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    from qwen_vl_utils import process_vision_info

    with _MODEL_LOCK:
        model, processor = _load_model("qwen")
        input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _videos, _video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True
        )
        inputs = processor(text=[input_text], images=images, return_tensors="pt").to("cpu")
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "route": route_name,
            "model": str(DEFAULT_MODEL_PATHS["qwen"]),
            "question": question,
            "raw_response": response.strip(),
            "generation_seconds": time.perf_counter() - started,
            "model_load_seconds": getattr(model, "_recallvision_load_seconds", None),
            "source_timestamps": [timestamp for _, timestamp in frames],
            "source_frame_ids": list(frame_ids) if frame_ids is not None else None,
            "prompt_characters": len(input_text),
        }


def analyze_memory_frames(
    frames: list[tuple[Any, float]], *, max_new_tokens: int = 1024
) -> dict[str, Any]:
    """Build memory from source frames whose timestamps are explicitly shown to Qwen."""
    from recallvision.video_ai.persistent_memory import build_memory_prompt

    result = _ask_timestamped_frames(
        frames,
        "Build a persistent visual memory from the supplied source frames.",
        route_name="automatic_memory_generation",
        prompt=build_memory_prompt(),
        max_new_tokens=max_new_tokens,
    )
    result["model_fingerprint"] = "qwen3.5-2b:aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
    return result


def ask_visibility_frames(
    frames: list[tuple[Any, float]], target: str, *, max_new_tokens: int = 512
) -> dict[str, Any]:
    """Check target presence at each supplied source timestamp independently."""
    target = target.strip()
    if not target:
        raise ValueError("Target description must not be empty")
    prompt = (
        f'The target is "{target}". Inspect each timestamped source image independently. '
        "Report whether this specific target is physically visible in each image. Use "
        "present only when visual details support the target; use absent when the target "
        "is plainly not in the image; use uncertain for occlusion, tiny/blurred appearance, "
        "or identity ambiguity. Do not infer where an absent object went or whether it "
        "still exists outside the camera view. For present frames give a short visible "
        "scene location (such as on the desk or held by a person). Return one JSON object "
        'with key "observations", an array of {"seconds": supplied timestamp, '
        '"visibility": "present"|"absent"|"uncertain", "location": string, '
        '"confidence": "high"|"medium"|"low"}. Include one observation for every supplied frame. '
        "Use only the exact supplied timestamps. Keep each observation compact so every "
        "frame is included in the response."
    )
    return _ask_timestamped_frames(
        frames, f"Is {target} visible at each supplied timestamp?", route_name="visibility_recheck",
        prompt=prompt, max_new_tokens=max_new_tokens,
    )


def ask_target_candidate_crops(
    frames: list[tuple[Any, float]], target: str, *, candidate_ids: list[str] | None = None,
    reference_frame: tuple[Any, float] | None = None, max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Review crops proposed by a detector; reject lookalike items and preserve uncertainty."""
    target = target.strip()
    if not target:
        raise ValueError("Target description must not be empty")
    if reference_frame is None:
        review_frames = frames
        review_ids = candidate_ids
        prompt = (
            f'The requested target is "{target}". Each supplied image is a crop around a detector-proposed region, '
            "not a confirmed target. Inspect every crop independently. Do not accept the detector label just because "
            "it is printed in the prompt. Decide whether the requested object itself is visibly present in this crop; "
            "a different object from the same broad category is not a match. If the crop is partial, confusing, or too "
            "small to distinguish identity, use uncertain. A crop-level decision applies only to its exact source time "
            "and cannot establish identity continuity across frames. Return one JSON object with key observations, one "
            'entry per crop: {"candidate_id": exact supplied ID, "seconds": exact supplied time, '
            '"visibility": "present"|"absent"|"uncertain", '
            '"location": short visible description or empty, "distinguishing_cues": short string, '
            '"confidence": "high"|"medium"|"low"}. Never infer that a detector miss means absence.'
        )
    else:
        if candidate_ids is None or len(candidate_ids) != len(frames):
            raise ValueError("candidate_ids are required when comparing crops to a reference frame")
        review_frames = [reference_frame, *frames]
        review_ids = ["verified_reference", *candidate_ids]
        prompt = (
            f'The requested object is "{target}". The first image has ID verified_reference and shows an earlier '
            "frame-local crop that was already supported as this target. Every later image is a new detector crop. "
            "For each later crop, compare distinctive appearance details with the reference and decide whether it is "
            "the same physical object. Similar category, color, or nearby location alone is not enough. Use same only "
            "when visible details match; use different when the crop shows another item; use uncertain for partial or "
            "insufficient evidence. Return one JSON object with key observations and one entry for each later crop, "
            'excluding the reference: {"candidate_id": exact candidate ID, "seconds": exact candidate time, '
            '"reference_match": "same"|"different"|"uncertain", '
            '"visibility": "present" for same, "absent" for different, or "uncertain", '
            '"distinguishing_cues": short string, "confidence": "high"|"medium"|"low"}. '
            "This decision applies only to the exact supplied crop and does not establish continuous visibility."
        )
    return _ask_timestamped_frames(
        review_frames,
        (f"Compare each candidate crop with the earlier supported reference for {target}."
         if reference_frame is not None else f"Check whether each candidate crop contains the requested target: {target}"),
        route_name="target_crop_review", prompt=prompt, max_new_tokens=max_new_tokens,
        frame_ids=review_ids,
    )


def ask_event_recheck_frames(
    frames: list[tuple[Any, float]], question: str, *, max_new_tokens: int = 384
) -> dict[str, Any]:
    """Recheck a disputed event against timestamped frames and scene-relative motion."""
    prompt = (
        "Recheck the user's claimed event using only these ordered timestamped source frames. "
        "Compare the target object's position relative to nearby stable scene landmarks. "
        "Account for a moving camera: if the object and background shift together, that is "
        "camera motion, not evidence the object moved. Ignore small detector-like framing "
        "jitter. Use moved only when the object's relation to its support surface or nearby "
        "landmarks visibly changes; use stationary when it remains in the same relative place; "
        "otherwise use uncertain. Do not infer who caused a change. Return one JSON object "
        'with keys "event_status" ("moved"|"stationary"|"uncertain"), '
        '"camera_motion" (true|false|null), "evidence_seconds" (at least two supplied times '
        'that support the comparison), "confidence" ("high"|"medium"|"low"), '
        '"abstain" (boolean), and "reason" (short string). If these frames cannot distinguish '
        "object motion from camera motion, choose uncertain and abstain.\n"
        f"User question: {question.strip()}"
    )
    return _ask_timestamped_frames(
        frames, question, route_name="event_recheck", prompt=prompt, max_new_tokens=max_new_tokens
    )
