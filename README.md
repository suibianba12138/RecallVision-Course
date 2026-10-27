<div align="center">

# RecallVision

### Persistent Visual Memory for Video Question Answering

**A local video-memory system that records what was observed, connects answers to source frames, and reuses evidence across questions.**

<br>

[![Python V3](https://img.shields.io/badge/Python%20V3-3.12-3776AB?logo=python&logoColor=white)](v3/README.md)
[![License](https://img.shields.io/badge/License-AGPL--3.0--only-2B8C8A)](LICENSE)
[![V3 Source Checks](https://github.com/suibianba12138/RecallVision-Course/actions/workflows/v3-course-checks.yml/badge.svg?branch=main)](https://github.com/suibianba12138/RecallVision-Course/actions/workflows/v3-course-checks.yml)
[![V3 Source](https://img.shields.io/badge/V3%20Source-Public-286A68)](v3/)
[![V1 Archive](https://img.shields.io/badge/V1%20Demo-Available-526A95)](course_demo/)

<br>

**[Project Website](https://suibianba12138.github.io/RecallVision-Course/)** &nbsp;·&nbsp;
**[Version 3.0 Demo](https://suibianba12138.github.io/RecallVision-Course/v3/)** &nbsp;·&nbsp;
**[Version 2.0 Demo](https://suibianba12138.github.io/RecallVision-Course/v2/)** &nbsp;·&nbsp;
**[Version 1.0 Demo](https://suibianba12138.github.io/RecallVision-Course/v1/)** &nbsp;·&nbsp;
**[V3 Source & Setup](v3/README.md)**

</div>

---

## Overview

RecallVision is a computer vision course project investigating how video observations can be converted into **object histories and event records** for retrospective retrieval. Instead of treating each question as an isolated video-processing task, the system stores structured observations and links its outputs to source frames and timestamps.

The repository preserves three distinct development milestones:

| | **RecallVision 3.0 — Final course version** | **RecallVision 2.0 — Reviewable baseline** | **RecallVision 1.0 — Recorded baseline** |
| :-- | :-- | :-- | :-- |
| **Focus** | Persistent semantic memory and natural-language Ask | Reviewable object memory and evidence-gated retrieval | Offline perception and recorded queries |
| **Interface** | **Analyze → Explore → Ask → Reopen** | Local Streamlit evidence review | Archived C1 interactive replay |
| **Queries** | Memory-first natural-language retrieval and source-linked evidence | Trusted **WHERE / WHEN / BEFORE** | Four recorded C1 routes, including authorized **WHO** |
| **Implementation** | [`v3/`](v3/README.md) | [`v2/`](v2/README.md) | [`recallvision/`](recallvision/) and [`course_demo/`](course_demo/) |
| **Demonstration** | [View V3](https://suibianba12138.github.io/RecallVision-Course/v3/) | [View V2](https://suibianba12138.github.io/RecallVision-Course/v2/) | [View V1](https://suibianba12138.github.io/RecallVision-Course/v1/) |

**V3 is the current course release candidate.** V3 is a separately packaged short-video memory application, with real saved-result demonstrations; first-pass inference and browser visual QA are not fully independently accepted. V2 remains the prior course milestone. The original V1 source remains at the repository root so that its code and recorded evidence are preserved without breaking existing paths. Ongoing V3 research is not included in this repository.

## RecallVision 3.0 · Persistent Visual Memory

V3 turns video analysis into a reusable visual record. The local application generates timestamped scene, object, and event descriptions, stores them in SQLite, and retrieves them for later natural-language questions. The **Analyze → Explore → Ask → Reopen** workflow keeps observations connected to the original video.

**[Explore V3 and video cases](https://suibianba12138.github.io/RecallVision-Course/v3/)** · **[Watch original clips](https://suibianba12138.github.io/RecallVision-Course/v3/#videos)** · **[Run locally](v3/README.md)** · **[Restore course demonstrations](v3/demo/README.md)**

| Recorded course case | Stored memories / evidence frames | Recorded outcome |
| :-- | :-- | :-- |
| Backpack / moving camera | 24 / 11 | Reference-matched candidate at **4.963 s**; later visibility **UNKNOWN** |
| Tablet and mug | 11 / 6 | Model-reported object candidates, **UNVERIFIED** |
| Plant interaction | 11 / 8 | Model-reported interaction, **UNVERIFIED** |

The demo importer restores authentic stored records for the **exact SHA-256-matched** Mixkit clips obtained from their original source pages; it does not rerun inference, and the repository does not redistribute videos, weights, JPEGs, or databases. Recorded restore times were **0.659–2.160 s** across the three clips. First-time CPU model inference on new uploads still took **202–272 s**.

RC1 verification reported **61 focused tests passed**; the broader suite reported **530 passed, 5 skipped, 2 deselected, 3 failed** (three legacy V2 UI environment-dependency failures). Browser visual acceptance remains outstanding. These are application/recorded-demo results, not claims of general video QA accuracy.

## RecallVision 2.0

### System workflow

V2 extends the original object-memory pipeline with **human-reviewable identity and event evidence** and an integrated local interface.

```text
                       LOCAL VIDEO
                            │
                   YOLO11n + ByteTrack
                            │
                Observations and tracklets
                            │
                  Persistent object IDs
                            │
                  State / event records
                            │
              Identity & evidence review
                            │
                 Trusted memory rebuild
                            │
                 WHERE / WHEN / BEFORE
                            │
                  Answer + source evidence
```

| Stage | Capabilities |
| :-- | :-- |
| **01 · Analyze** | Import a supported local video, execute the processing pipeline, inspect object candidates, or reopen a completed analysis. |
| **02 · Explore** | Examine identity hypotheses and observed histories; explicitly review or revoke frame observations, entity associations, and motion-event candidates. |
| **03 · Ask** | Retrieve historical states and reviewed event transitions with frame/time evidence. Return explicit uncertainty when the required evidence has not been approved. |

Automatic detections and movement candidates are **not treated as confirmed facts**. Review applies to specific observations or events, not to unobserved intervals. V2 does not claim general automatic person attribution.

<div align="center">

**[Explore the V2 Workflow and Recorded Results →](https://suibianba12138.github.io/RecallVision-Course/v2/)**

</div>

### Recorded application results

The archived V2 application completed local processing for the C1 bottle and P1-B coffee-cup clips:

<div align="center">

| **C1 · Bottle** | **P1-B · Coffee cup** |
| :--: | :--: |
| **577** total observations | **3,268** total observations |
| **50** bottle observations | **752** cup observations |
| **1** persistent bottle candidate | **7** cup candidates |
| **45.97 s** pipeline runtime | **90.65 s** pipeline runtime |

</div>

These numbers describe **completed application runs**, not confirmed object-level accuracy. Without explicit human review, the trusted query paths correctly report insufficient evidence or require event review.

**Evaluation boundary:** the independent E1 evaluation resolved **0/5 annotated targets automatically** and found motion-event false positives. The prototype demonstrates system integration and inspectable evidence, **not validated general-purpose video understanding**.

**Verification:** the original V2 engineering worktree reported **416 passed, 7 skipped**, including tests relying on external frozen fixtures. The published repository's course-source CI separately passed **14 selected tests** at the latest verified commit; this does not establish full fresh-video inference or browser acceptance.

For complete results and limitations, see [V2 Evaluation](v2/docs/EVALUATION.md).

### Run V2 locally

**Requirements:** Linux, Python 3.12, a compatible locally obtained YOLO11n checkpoint, and the additional perception dependencies. Native Windows is not supported by the archived Streamlit application.

```bash
git clone https://github.com/suibianba12138/RecallVision-Course.git
cd RecallVision-Course/v2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,perception]"

# Download the compatible YOLO11n weights separately.
export RECALLVISION_MODEL_PATH="$PWD/models/yolo11n.pt"
python -m streamlit run recallvision/apps/identity_review.py
```

The checkpoint is **not** included in Git. See the [V2 setup guide](v2/README.md) for environment configuration, supported video inputs, saved runs, and the complete application workflow.

> **Local application only.** The public GitHub Pages demonstration replays recorded evidence and interface states; it does **not** run YOLO or the V2 Streamlit backend. V2 is intended for a trusted local operator and is not configured as an authenticated public service.

## RecallVision 1.0

V1 is the original **perception → object memory → historical retrieval** pipeline. The published C1 demonstration uses a real 7.91-second bottle sequence and replays saved observations, a movement event, and four typed query results.

<div align="center">

| **Perception** | **Persistent memory** | **Historical retrieval** |
| :--: | :--: | :--: |
| **48** bottle observations | **7 → 1** track IDs to object identity | **4** recorded query routes |
| YOLO11n + ByteTrack | State timeline + `MOVE_OBJECT` | Query result + timestamp + frame |

**[Watch the Annotated C1 Video and Query Replay →](https://suibianba12138.github.io/RecallVision-Course/case-studies/)**

</div>

**Example recorded query**

> **Where was the bottle before it moved?**  
> **Left region**, supported by **frame 92 (3.0697 s)**. The saved movement evidence spans frames **92–99 (3.0697–3.3033 s)**.

| Query | Recorded C1 result |
| :-- | :-- |
| `WHERE_OBJECT` | Center region, frame 152 at 5.0717 s |
| `WHEN_OBJECT_MOVED` | Movement interval, frames 92–99 |
| `WHERE_OBJECT_BEFORE` | Left region, frame 92 |
| `WHO_MOVED_OBJECT` | `obj:person:0001` from a **pre-authorized** attribution record |

The C1 video is an **annotated analysis export**, not an online inference stream. Its query answers are frozen development artifacts; the WHO route does not establish automatic person identification.

Resources: [V1 source](recallvision/) · [Recorded fixtures](course_demo/fixture/) · [Reproduction guide](docs/COURSE_REPRODUCTION.md).

## Repository layout

```text
RecallVision-Course/
├── README.md                 # Project overview and version entry points
├── v3/                       # V3 final course application and recorded demo
│   ├── recallvision/         # V3 video memory and evidence logic
│   ├── demo/                 # Hash-bound, saved-result demo package
│   ├── scripts/              # Safe offline import utility
│   └── docs/                 # V3 setup, guide, and RC1 report
├── v2/                       # V2 reviewable course application
│   ├── recallvision/         # App, perception, memory, events, query, pipeline
│   ├── configs/              # Pipeline configurations
│   ├── tests/                # Source and integration tests
│   ├── docs/                 # Architecture, code guide, evaluation and usage
│   └── pyproject.toml
├── recallvision/             # Preserved V1 implementation
├── course_demo/              # V1 C1 recorded-query artifacts
├── configs/                  # V1 configurations
├── tests/                    # V1 tests
└── docs/                     # V1 documentation and course resources
```

**Run each version from its own project root.** The separately maintained `gh-pages` branch contains the public project website.

## Documentation & reproducibility

| Document | Purpose |
| :-- | :-- |
| [V3 README](v3/README.md) | Final course application setup, upload/analysis and local demo | 
| [V3 Demo Guide](v3/demo/README.md) | Original clips, SHA-256 checks, memory-pack restoration | 
| [V3 User Guide](v3/docs/RECALLVISION_V3_USER_GUIDE.md) | Analyze, Explore, Ask and Reopen | 
| [V2 README](v2/README.md) | Previous reviewable application and setup |
| [V2 Architecture](v2/docs/ARCHITECTURE.md) | Processing stages and trust semantics |
| [V2 Code Guide](v2/docs/CODE_GUIDE.md) | Source entry points, module responsibilities, and reading order |
| [V2 Evaluation](v2/docs/EVALUATION.md) | Recorded results, testing boundaries, and limitations |
| [V1 Reproduction](docs/COURSE_REPRODUCTION.md) | Archived C1 evidence and source provenance |

**Distribution scope:** this repository includes source code, documentation, configurations and selected recorded C1 fixtures. It excludes private raw videos, trained model weights, uploaded content, generated run outputs and local review databases. Some archived tests require fixtures that are not in the public repository.

**Security:** bounded uploads, fixed subprocess arguments, local-only service binding and provenance checks address specific local failure modes; they do not constitute a comprehensive security audit or make the application suitable for public deployment.

**License:** project source is released under [AGPL-3.0-only](LICENSE). Third-party dependencies, models and media retain their own license terms. See [COPYRIGHT](COPYRIGHT) and [V2 Third-Party Notices](v2/docs/THIRD_PARTY_NOTICES.md).

---

<div align="center">

**RecallVision · Computer Vision Course Project**

[Project Website](https://suibianba12138.github.io/RecallVision-Course/) ·
[V3 Code](v3/) ·
[V3 Demonstration](https://suibianba12138.github.io/RecallVision-Course/v3/) ·
[V2 Demonstration](https://suibianba12138.github.io/RecallVision-Course/v2/) ·
[V1 Demonstration](https://suibianba12138.github.io/RecallVision-Course/v1/)

</div>
