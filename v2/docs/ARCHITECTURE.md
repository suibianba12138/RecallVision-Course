# RecallVision 2.0 — System Architecture

## Processing graph

```text
Local input video
  → YOLO11n detection + ByteTrack
  → frame-level observations / tracklets
  → persistent entity hypotheses (P2A-R1)
  → object states / zones (P2B, P2C)
  → motion-event candidates (P2D, P3-B1)
  → reversible identity and evidence review (D1)
  → trusted memory reconstruction
  → WHERE / WHEN / WHERE-BEFORE + exact source evidence
```

## Application boundary

`recallvision/apps/identity_review.py` is the local Streamlit interface. `run_manager.py` accepts bounded local videos and executes the fixed `recallvision.cli.pipeline` entry point with controlled arguments. Inference may run in the same Python environment or a separate environment via `RECALLVISION_BACKEND_PYTHON`.

Pipeline configurations reside in `configs/`. Each run records video/model/config provenance and writes generated outputs under `outputs/`, excluded from version control. A reopened run is checked against the original inputs and output hashes.

## Evidence and trust contract

- **Perception produces candidate observations**; an observation is not automatically marked human-supported.
- **Identity operations** can combine or separate candidate tracklets and are independently recorded and reversible.
- **Observation review** applies only to a particular source frame (`SUPPORTED`, `UNCERTAIN`, `MISSING`). No synthetic observations are inserted across gaps.
- **Movement review** is separate from automatic movement proposal generation. The deterministic trusted query paths require appropriate reviewed evidence.
- **WHERE** reads the latest supported state, **WHEN** reads reviewed movement transitions, and **BEFORE** reads the reviewed pre-event state. Unsupported queries return explicit uncertainty, not extrapolated facts.
- The coordinates in reviewed location queries are image-plane positions or zone descriptions, **not** inferred 3D world coordinates.

## Scope of this release

The current architecture deliberately prioritizes inspectable object memory and reviewability. It is not a free-form VLM video QA system, does not offer general automatic WHO attribution, and has no claim of robust long-video generalization. Recorded development runs should not be confused with independent task success.

See [Evaluation](EVALUATION.md) for observed behavior and limitations.
