# RecallVision-Course

RecallVision builds structured visual memory from video observations and uses that memory for retrospective question answering. It is a computer-vision course project focused on tracking objects over time, recording state and event evidence, and retrieving answers with supporting timestamps and frames.

> **License and release scope:** the project owner's original RecallVision source is licensed under AGPL-3.0-only; see [`LICENSE`](LICENSE) and [`COPYRIGHT`](COPYRIGHT). This source-only candidate excludes model weights, third-party package binaries, raw videos, and evidence images.

## Project overview

The system separates video processing from question answering. An offline pipeline converts video into observations, persistent entities, state timelines, and supported events. At query time, deterministic handlers retrieve a matching memory record and its evidence. The project does not claim that arbitrary new videos complete this pipeline successfully.

“Long-Form” remains in the project title, but long-form performance is **NOT VALIDATED**.

## System architecture

```text
OFFLINE MEMORY CONSTRUCTION
Video
  -> P1B perception: YOLO11n via Ultralytics + ByteTrack tracking config
  -> P2A-R1 persistent-entity consolidation
  -> P2B object-state timelines
  -> P2C/P2D event memory
  -> timestamped, evidence-linked artifacts

QUERY-TIME RETRIEVAL
Question
  -> query type and target resolution
  -> structured state/event/authorized-attribution lookup
  -> answer status + event/time reference + supporting frame(s)
```

The detector and tracker produce short-term observations; a tracker ID is not treated as persistent identity. The four query handlers operate over recorded memory, not raw pixels at query time. `WHO_MOVED_OBJECT` only reads pre-authorized attributed memory and fails closed when authorization is absent.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for stage boundaries and interfaces.

## Main features

- Video observations with frame index, timestamp, class, bounding box, confidence, tracker ID, and spatial region.
- Persistent object entities consolidated across short-term tracklets.
- Per-entity state timelines and evidence-bounded movement events.
- Four typed query handlers with explicit unsupported, missing, and ambiguous-target statuses.
- A public, text-only summary of the recorded C1 demo; it does not ship evidence images or claim a public-checkout replay.

## Technical pipeline

The frozen V1 path is `P1B -> P2A-R1 -> P2B -> P2C/P2D -> query`. P1B uses the optional Ultralytics adapter with `yolo11n.pt` and the Ultralytics `bytetrack.yaml` tracker configuration. P2A-R1 consolidates observations into persistent entities; P2B builds state timelines; P2C/P2D derive supported events. Query handlers retrieve from these artifacts and preserve ambiguity rather than selecting an unsupported target.

No model weights, raw videos, training data, or cached model files are included. Perception dependencies are optional; weights and their applicable terms must be handled separately. See [`requirements.txt`](requirements.txt) and [`configs/`](configs/).

## Supported queries

| Query | Retrieval contract |
|---|---|
| `WHERE_OBJECT` | Return the target’s last-observed state/location, with evidence. |
| `WHEN_OBJECT_MOVED` | Return only supported movement event intervals. |
| `WHERE_OBJECT_BEFORE` | Return an observed state before a uniquely identified supported movement event. |
| `WHO_MOVED_OBJECT` | Return only a person identity from pre-authorized attributed memory; otherwise fail closed. |

These are implemented query contracts, not a guarantee that each query will be answerable for every video.

## Installation

Python 3.10 or newer is required. Core schemas and the public text summary do not run a detector or require model weights. The full internal C1 replay is not included in this public candidate.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

For video perception only, install the optional extra in an environment with an appropriate PyTorch build:

```bash
python -m pip install -e ".[perception]"
```

This installs the declared Ultralytics/OpenCV dependencies, not a RecallVision checkpoint or dataset. The optional Ultralytics dependency is AGPL-3.0; upstream states its AGPL route is for projects whose full source is open under AGPL-3.0. Ultralytics model weights are not included; see [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) for the source-only distribution boundary.

## Public demo summary

The public page is a static summary of recorded course results. It is **not** the C1 replay program and does not validate artifact hashes or render evidence frames. The original C1 replay passed in the internal course package; its three Pixabay-derived evidence images are deliberately omitted from this public candidate, so a full replay is not reproducible from this checkout.

Serve the summary from the repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/public-demo/`. No model, source video, or image evidence is loaded. See [`public-demo/README.md`](public-demo/README.md) for the demonstration order and limitations.

Do not describe this public summary as fresh inference or a successful full artifact replay.

## Validation results

- **C1 — RECORDED PASS (internal artifact replay):** four-query development-artifact replay; not reproducible in this asset-restricted public checkout.
- **C2 — AMBIGUOUS_TARGET:** independent cake video; target resolution blocked at P2B.
- **C2-R — AMBIGUOUS_TARGET:** independent bottle video; target resolution blocked at P2B. P2C/P2D and downstream positive queries were not reached.

Fresh-video end-to-end QA success is **not established**. Long-form generalization is **NOT VALIDATED**; cross-video robustness and automatic VLM attribution are **NOT ESTABLISHED**. Details are in [`docs/VALIDATION_RESULTS.md`](docs/VALIDATION_RESULTS.md).

## Tests

Run the included lightweight source tests:

```bash
python -m pytest -q
```

The public summary does not validate fixture hashes, run query handlers, or display supporting frames. In the lightweight test environment, OpenCV-dependent tests skip unless the optional perception extra is installed. These software tests are not video inference or a reproduction of the C1 replay.

## Current limitations

- Independent-video tests did not resolve a unique target, so their event/query stages are blocked or unrun.
- Long-form performance, cross-video robustness, and open-vocabulary generalization are not validated or established.
- Person attribution is not automatic; `WHO_MOVED_OBJECT` requires pre-authorized attributed memory.
- The recorded internal C1 demo was an artifact replay, not fresh inference; the public summary is not an executable demo.
- No dataset-level end-to-end accuracy has been established.
- Model weights and raw videos are intentionally excluded.

## Acknowledgments

The optional perception adapter depends on Ultralytics and its ByteTrack tracking implementation/configuration; see [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md). The C1 clip was sourced from [Pixabay video 82261](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/), source SHA256 `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730`. No video or evidence image is included in the public Git tree. The images remain only in the separate local internal fixture pending rights review.

## License

Copyright (C) 2026 suibianba12138. The original RecallVision-Course source, configuration, tests, and documentation are licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [`LICENSE`](LICENSE). Third-party components and media are governed by their own notices and are not relicensed by this statement. The repository does not distribute platform wheels/binaries, model weights, raw videos, or evidence images.
