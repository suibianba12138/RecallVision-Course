# RecallVision

**Object-Centric Visual Memory for Retrospective Video Understanding**

RecallVision is a computer vision course project that studies how video observations can be represented as persistent object histories and evidence-linked event records for subsequent retrieval. The project includes two distinct implementations.

**[Project Website](https://suibianba12138.github.io/RecallVision-Course/)** · **[V2 Demonstration](https://suibianba12138.github.io/RecallVision-Course/v2/)** · **[V1 Demonstration](https://suibianba12138.github.io/RecallVision-Course/v1/)**

## Versions

| Version | Location | Implementation | Status |
| --- | --- | --- | --- |
| **2.0 — Course submission** | **[`v2/`](v2/README.md)** | Local Streamlit application; video import; reviewable object and event histories; evidence-gated WHERE, WHEN, and BEFORE queries | Source published; selected clean-runner tests passed. Full model and browser acceptance remain outstanding. |
| **1.0 — Original implementation** | **[`recallvision/`](recallvision/) and [`course_demo/`](course_demo/)** | Object-centric detection, tracking, temporal state and event records; four frozen C1 query routes | Source and recorded C1 demonstration preserved. |

The V1 package remains at repository root to preserve existing course artifacts. **V2 is an independently packaged Python project**; enter `v2/` before running its commands. Ongoing V3 research is not part of this submission.

## RecallVision 2.0

The V2 system extends the original object-memory pipeline with a local application and explicit review of uncertain evidence.

```text
Local video
    ↓
YOLO11n detection + ByteTrack
    ↓
Object/identity candidates → State and event candidates
    ↓
Human evidence review (identity, observation, movement)
    ↓
Evidence-aware memory reconstruction
    ↓
WHERE / WHEN / BEFORE + source frame and timestamp
```

| Interface stage | Purpose |
| --- | --- |
| **Analyze** | Import a bounded local video, execute the processing pipeline, or reopen an existing run. |
| **Explore** | Inspect object histories and movement candidates; review or revoke evidence decisions. |
| **Ask** | Retrieve historical information from reviewed observations and events, or return an explicit insufficient-evidence status. |

The system does **not** treat unreviewed movement candidates as confirmed events, and it does not provide general automatic person attribution.

### Installation and course assessment

The complete V2 source, installation procedure and usage guide are available in **[v2/README.md](v2/README.md)**. The project is intended for **local Linux/Python 3.12 operation** with a separately obtained YOLO11n checkpoint.

Useful references:

- [V2 architecture](v2/docs/ARCHITECTURE.md)
- [V2 code organization](v2/docs/CODE_GUIDE.md)
- [V2 evaluation and limitations](v2/docs/EVALUATION.md)
- [Third-party dependencies and licenses](v2/docs/THIRD_PARTY_NOTICES.md)

The [public V2 website](https://suibianba12138.github.io/RecallVision-Course/v2/) presents **recorded** results and a read-only workflow illustration; it does not run the inference system remotely.

### Evaluation status

The frozen V2 development environment reported **416 tests passed, 7 skipped**, with some tests using external archived fixtures. In the published repository, the GitHub Actions course-source checks have separately passed **12 dependency-light tests**. These are not equivalent to a complete end-to-end clean installation.

Archived D2 application runs completed on the C1 bottle and coffee-cup sequences. Independent E1 evaluation, however, resolved **0/5 annotated targets automatically**. The prototype demonstrates an auditable integration workflow, **not validated general-purpose video understanding**. For detailed results and limitations, consult the [evaluation report](v2/docs/EVALUATION.md).

## RecallVision 1.0

V1 preserves the original detection-to-memory pipeline and the recorded 7.91-second C1 bottle sequence. The [V1 interactive demonstration](https://suibianba12138.github.io/RecallVision-Course/case-studies/) replays authentic recorded observations and four saved query outputs (WHERE, WHEN, BEFORE, and an authorized WHO attribution). It does **not** execute fresh inference.

- [V1 interactive C1 results](https://suibianba12138.github.io/RecallVision-Course/case-studies/)
- [C1 recorded query fixtures](course_demo/)
- [V1 source code](recallvision/)
- [V1 method and evaluation documents](docs/)

The V1 application should be executed from the repository root; V2 commands should be executed from `v2/`.

## Repository structure

```text
RecallVision-Course/
├── README.md                  # Version overview and course access
├── v2/                        # Version 2.0: independent application + docs + tests
│   ├── recallvision/          # V2 Python implementation
│   ├── configs/               # V2 inference and event configurations
│   ├── tests/                 # V2 source and integration tests
│   ├── docs/                  # Technical guides and evaluations
│   └── pyproject.toml
├── recallvision/              # Preserved V1 Python implementation
├── course_demo/               # Original V1 C1 recorded-query fixtures
├── configs/                   # V1 configuration
├── tests/                     # V1 tests
└── docs/                      # V1 documentation and project material
```

The website is maintained separately on the `gh-pages` branch.

## Security, reproducibility and licensing

- The course repository intentionally excludes raw third-party videos, model weights, uploaded files, generated outputs, and local review databases.
- V2 uses bounded upload sizes, controlled subprocess commands and source-hash checks for reopening analyses. These measures **do not** make Streamlit suitable for unauthenticated public deployment.
- The main application has a Linux/POSIX file-lock dependency. Full fresh-video inference and real-browser acceptance are **not yet verified** for the public release.
- Project source code is licensed under [AGPL-3.0-only](LICENSE); third-party media and model licenses are independent. See [COPYRIGHT](COPYRIGHT) and [V2 third-party notices](v2/docs/THIRD_PARTY_NOTICES.md).

**Repository scope:** final V2 course source and preserved V1 artifacts. V3 research is developed separately.
