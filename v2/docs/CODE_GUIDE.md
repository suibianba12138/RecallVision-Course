# RecallVision V2 — Code Guide

This document provides a navigation map for the independently packaged V2 course source. It describes the archived implementation rather than an idealized or redesigned architecture.

## Entry points

- `recallvision/apps/identity_review.py` — Streamlit UI, including Analyze, Explore, Ask and identity review. This is a large, legacy integration module (~1,900 lines); it is not a small reusable API.
- `recallvision/apps/run_manager.py` — bounded video import, metadata probing, controlled backend process invocation, and provenance checks when reopening completed runs.
- `recallvision/cli/pipeline.py` — fixed CLI entry for video processing.
- `recallvision/cli/identity_overlay.py` — CLI entry for identity-overlay operations and evidence regeneration.
- `recallvision/pipeline/runner.py` — orchestrates the processing stages and stores run manifests.

## Processing modules

| Directory | Responsibility |
| --- | --- |
| `perception/` | Video metadata, detector and ByteTrack integration |
| `memory/` | Observations, tracklets, persistent entities, identity overlays, trust review and JSONL stores |
| `events/` | Zone and motion-event proposals, movement evidence |
| `pipeline/` | End-to-end orchestration and trusted-memory rebuild |
| `query/` | Typed question parsing, target resolution, supported WHERE/WHEN/BEFORE retrieval |
| `visualization/` | Source-frame evidence rendering |
| `evaluation/` | Development evaluation routines and metrics |
| `pose/`, `vlm_f1/`, `vlm_f2/`, `vlm_f3/` | Retained research/experimental modules from the immutable V2 snapshot. Not required by the primary D2 course workflow. |

## Configuration and state

- Pipeline parameters: `configs/`.
- Streamlit configuration: `.streamlit/config.toml`.
- Model location: `RECALLVISION_MODEL_PATH` (defaults to `models/yolo11n.pt` under `v2/`).
- Backend interpreter: `RECALLVISION_BACKEND_PYTHON` (defaults to the interpreter running Streamlit).
- Run outputs: `outputs/` (ignored by Git).
- Original videos, generated evidence, model weights, and review databases are **not version controlled**.

## Testing

`tests/` contains the archived V2 tests. Some integration tests rely on local frozen fixtures omitted from this public source. GitHub Actions runs a small, clean-runner subset plus syntax and structural checks; this is not equivalent to the historical complete regression count.

## Design limitations and audit notes

- The UI is relatively monolithic; direct refactoring without end-to-end tests risks changing review-state behavior. Preserve its boundaries until a verified UI regression suite exists.
- The application is intended for a trusted local operator; it is **not** a hardened multi-user web service.
- Input filename sanitization, bounded uploads, fixed subprocess argv, and provenance hashes offer protection against specific failure modes. They do not eliminate untrusted video decoder risks, malicious local files, or denial-of-service by a permitted local user.
- In some development modules there are path- and fixture-specific assumptions; the public course workflow is confined to the documented `v2/` entry points.
