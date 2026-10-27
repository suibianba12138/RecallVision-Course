# RecallVision V3 — Course Release Candidate 1

RecallVision V3 is a local visual-memory application for short videos. Upload a clip and the app builds persistent scene, object, event, and time-stamped observation memories. Explore their source frames, ask questions in natural language, and reopen the saved video and memory after restarting the app.

The app keeps model-generated claims separate from frame-level checks. `CORROBORATED` applies to the specific sampled frame that was checked; it does not prove identity continuity. `UNKNOWN` remains unknown. Model confidence is not calibrated.

## Product flow

```text
Upload → Analyze → Explore → Ask → Reopen
```

- **Analyze** stores the source video and creates a timestamped memory from sampled frames.
- **Explore** shows object and event claims, evidence status, exact source times, and frames.
- **Ask** retrieves saved memory first, reuses cached answers, and calls local models only when the question needs additional analysis.
- **Reopen** restores videos, memories, evidence frames, and saved answers from the local persistent store.

The app is designed for short clips (up to 180 seconds and 250 MiB). Inference is local; model weights, uploaded clips, SQLite data, and caches are not part of this repository.

## Install

Use Python 3.12. A course-demo-only environment can open restored memories without installing model weights:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[app]'
```

For new-video analysis and model-backed questions, install PyTorch from the [official PyTorch selector](https://pytorch.org/get-started/locally/) and the pinned V3 dependencies:

```bash
# In the same activated environment, after installing a compatible PyTorch build:
python -m pip install -e '.[app,test]'
python -m pip install -r requirements-v3-video-ai.txt
```

Download the pinned local model snapshots and configure their paths as described in [`docs/LOCAL_VIDEO_MEMORY.md`](docs/LOCAL_VIDEO_MEMORY.md). No API key or paid service is used.

## Run

```bash
export RECALLVISION_MEMORY_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/recallvision-v3"
export RECALLVISION_VIDEO_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/recallvision-v3/video-cache"
python -m streamlit run recallvision/apps/video_memory.py --server.address 127.0.0.1 --server.port 8504
```

Open <http://127.0.0.1:8504>. For the repeatable course demo, first follow [`demo/README.md`](demo/README.md). It restores real saved results for the three course clips without running inference. The regular **Choose a video** upload path continues to analyze new local videos.

See the [user guide](docs/RECALLVISION_V3_USER_GUIDE.md) for the Analyze, Explore, Ask, and Reopen workflow.

## Test

Install the test extra in the active environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -m 'not historical_artifact' -q
```

The `historical_artifact` tests refer to frozen pre-V3 outputs that are not included in this product checkout. Excluding that marker does not alter the tests or their assertions.

## Architecture and boundaries

Qwen3.5 builds source-timed semantic memory from sampled frames. SQLite stores video metadata, memory claims, verification state, evidence references, and cached answers. Source video frames are decoded on demand and cached locally. TimeLens2 and Grounding DINO support selected temporal and target-visibility questions; they are not required for every query. A question that lacks supporting evidence should remain uncertain.

RecallVision V2 remains frozen in its release notes and historical artifacts; this README describes the V3 application. This standalone V3 course package is located under `RecallVision-Course/v3/`; V1 and V2 remain independent. This public release contains the runtime, reproducible recorded demo pack, tests, and selected user documentation, not the full private research workspace.

## Known limits

- The default runtime is CPU-based; first-time video memory generation takes minutes on this host.
- Sparse samples do not establish continuous visibility or exact event boundaries.
- Similar objects, camera movement, missed detections, and occlusion can leave identity or visibility `UNKNOWN`.
- Many scene/object/event memories are unverified model claims. Open the linked source frames before relying on them.
- General video understanding and person attribution are not guaranteed.
