# RecallVision 2.0 — Course Submission

**Evidence-Aware Visual Memory for Retrospective Video Question Answering**

This directory contains the archived **RecallVision V2 application source**. Version 1 remains at the repository root. V2 is a local, review-oriented video-memory prototype; it is not the separately developed V3 video-language research pipeline.

## 1. Capabilities and workflow

| Stage | Implementation |
| --- | --- |
| **Analyze** | Import a local video or reopen a previous run, perform YOLO11n detection and ByteTrack tracking, and construct object/identity and motion candidates. |
| **Explore** | Inspect entity histories and exact source-frame evidence. Review or revoke candidate identity relations, observation trust, and movement events. |
| **Ask** | Query WHERE, WHEN, and WHERE-BEFORE using reviewed memory. Return timestamps/source evidence where supported; otherwise report uncertainty or the need for review. |

**Important:** An automatic candidate is *not* a certified event. The trusted query layer intentionally refuses unsupported claims; a user must inspect and approve relevant evidence. V2 does not implement general automatic WHO attribution.

## 2. Quick start

Requirements: **Python 3.12 on Linux** (the application uses the POSIX `fcntl` file-lock API), a working FFmpeg/OpenCV video decoder, and sufficient RAM and storage for local CPU inference. macOS has not been acceptance-tested; native Windows is not supported by the current Streamlit application.

From the repository root:

```bash
cd v2
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[app,perception]"
```

Obtain a compatible **YOLO11n** checkpoint from the [Ultralytics documentation](https://docs.ultralytics.com/models/yolo11/). Do not commit model files. Place the checkpoint at `v2/models/yolo11n.pt`, or specify its location:

```bash
export RECALLVISION_MODEL_PATH="$PWD/models/yolo11n.pt"
# Optional: only needed when the Streamlit UI uses a separate Python environment.
export RECALLVISION_BACKEND_PYTHON="$(command -v python)"
python -m streamlit run recallvision/apps/identity_review.py
```

The app uses the current Python interpreter for its backend when `RECALLVISION_BACKEND_PYTHON` is unset. On Windows, use a suitable Linux environment (such as WSL2); native Windows execution is not supported by the current code.

The app supports videos up to **250 MiB and 120 seconds**, subject to available resources. Original source videos, trained weights, and generated analysis outputs are not included in this repository. No API keys or hosted services are required.

## 3. Using the application

1. Open **Analyze**, upload a supported video, inspect metadata, and run analysis. Alternatively reopen a completed local run.
2. Select the target entity candidate. Inspect identity fragments and exact observed frames.
3. Open **Explore** to review identity, observation and event evidence. Unreviewed data remains `UNCERTAIN`; missing observations are not filled in.
4. Open **Ask** to issue typed WHERE, WHEN, or WHERE-BEFORE requests. With insufficient review evidence, the app returns an explicit status rather than an unsupported answer.
5. Reopen a saved run to reuse its original analysis. Run reuse validates video/model/configuration and output hashes.

See [Application Guide](docs/LOCAL_APP.md) and [Architecture](docs/ARCHITECTURE.md) for further details.

## 4. Repository structure

```text
v2/
├── recallvision/
│   ├── apps/                 # Streamlit app, import and run management
│   ├── cli/                  # Controlled pipeline entry points
│   ├── perception/           # Detector/tracker adapter
│   ├── memory/               # Objects, identity overlay and review
│   ├── events/               # Candidate events and evidence
│   ├── pipeline/             # Integration and trusted rebuild
│   └── query/                # Evidence-gated historical queries
├── configs/                  # Version 2 pipeline configurations
├── tests/                    # Source tests (some require archived local fixtures)
├── docs/                     # Usage, architecture and evaluation
├── pyproject.toml
└── .streamlit/config.toml
```

The entire V2 code tree is retained under `v2/recallvision` so internal imports resolve independently of the root V1 package. **Always change directory to `v2/` before launching or testing.** No V3 modules, private raw media, checkpoints or generated experiment runs are copied here.

## 5. Tests

```bash
cd v2
python -m pip install -e ".[test]"
python -m pytest -q tests/test_d2_application_services.py tests/test_memory_store.py tests/test_imports.py
```

These are representative source checks, not a claim of complete clean-clone parity. Some historical integration tests reference ignored V1/V2 development fixtures and may skip or require separately supplied artifacts; see [Evaluation and Limitations](docs/EVALUATION.md). The archived engineering worktree reported **416 passed, 7 skipped**, which is *not* a new clean-clone result.

## 6. Evidence, limitations and course demonstration

- [Project overview and version selection](https://suibianba12138.github.io/RecallVision-Course/)
- [V2 workflow and archived run results](https://suibianba12138.github.io/RecallVision-Course/v2/)
- [V1 real C1 annotated video](https://suibianba12138.github.io/RecallVision-Course/v1/)
- [Evaluation and limitations](docs/EVALUATION.md)

The published website is a static demonstration of recorded outcomes; it does **not** execute V2 inference. Archived V2 applications completed C1 and coffee-cup processing, but independent E1 evaluation resolved **0/5** annotated targets automatically. The implementation is therefore an inspectable course prototype, **not** a validated general-purpose video understanding system.

## 7. License and third-party materials

This V2 source is released as part of RecallVision-Course under **AGPL-3.0-only**; see [repository LICENSE](../LICENSE) and [COPYRIGHT](../COPYRIGHT). Dependencies, checkpoint weights, dataset videos and other third-party materials retain their own terms; see [Third-Party Notices](docs/THIRD_PARTY_NOTICES.md). No raw video, weights, or private research data are distributed with this release.

**Source provenance:** copied from the immutable private V2 archive commit `ae0a65a21e02366453a9612cb620649f3a0bac3f`; only release-safe portability, documentation and packaging changes are applied to this public course copy. The frozen private source and ongoing V3 research remain separate.
