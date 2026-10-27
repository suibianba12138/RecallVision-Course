# Local Video Memory Setup

RecallVision V3 runs video analysis and question-specific checks locally. Uploaded media, SQLite memory, evidence JPEGs, and analysis caches are stored on the machine. The app does not call a paid API.

## Environment

Use Python 3.12. Install a PyTorch build that matches the operating system and CPU/GPU from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch using the official selector before this step.
python -m pip install -e '.[app,test]'
python -m pip install -r requirements-v3-video-ai.txt
```

`requirements-v3-video-ai.txt` pins the Transformers source revision and Qwen video utilities used by the current V3 runtime. The `app` extra installs Streamlit and OpenCV. Model weights are downloaded separately and are not included in Git.

## Download the pinned models

The Qwen model is needed for automatic memory generation and general questions. TimeLens2 and Grounding DINO are used by selected event and target-visibility questions. Installing all three enables the complete local feature set. The checked-in revisions are:

| Role | Model | Revision |
|---|---|---|
| Video memory and visual QA | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Temporal grounding | `MCG-NJU/TimeLens2-2B` | `10a779b6ef6fc8ad41367db2d1eb6779ca84995a` |
| Target candidate detection | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |

Store model snapshots outside the repository. This command uses the user cache directory rather than a machine-specific absolute path:

```bash
export RECALLVISION_MODEL_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/recallvision-v3/models"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["RECALLVISION_MODEL_ROOT"]).expanduser()
models = [
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "qwen3.5-2b"),
    ("MCG-NJU/TimeLens2-2B", "10a779b6ef6fc8ad41367db2d1eb6779ca84995a", "timelens2-2b"),
    ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71", "grounding-dino-tiny-a2bb814dd30d776dcf7e30523b00659f4f141c71"),
]
for repo_id, revision, directory in models:
    snapshot_download(repo_id, revision=revision, local_dir=root / directory)
PY
```

Point the app at those local snapshots:

```bash
export RECALLVISION_QWEN_MODEL_PATH="$RECALLVISION_MODEL_ROOT/qwen3.5-2b"
export RECALLVISION_TIMELENS_MODEL_PATH="$RECALLVISION_MODEL_ROOT/timelens2-2b"
export RECALLVISION_GROUNDING_DINO_PATH="$RECALLVISION_MODEL_ROOT/grounding-dino-tiny-a2bb814dd30d776dcf7e30523b00659f4f141c71"
```

These are also the default model paths. No model downloads occur automatically during inference; missing local files produce an actionable error. Grounding DINO checks the pinned checkpoint checksum before use.

## Start the app

```bash
export RECALLVISION_MEMORY_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/recallvision-v3"
export RECALLVISION_VIDEO_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/recallvision-v3/video-cache"
python -m streamlit run recallvision/apps/video_memory.py --server.address 127.0.0.1 --server.port 8504
```

The app accepts local clips up to 250 MiB and 180 seconds. Use [`RECALLVISION_V3_USER_GUIDE.md`](RECALLVISION_V3_USER_GUIDE.md) for product steps and [`../demo/README.md`](../demo/README.md) for restoring the course demo pack.

## Local data locations

- `$RECALLVISION_MEMORY_DIR` — SQLite store, persistent copies of uploaded videos, and source evidence frames.
- `$RECALLVISION_VIDEO_CACHE_DIR` — decoded source frames, video metadata, and reusable model-route outputs.
- `$RECALLVISION_MODEL_ROOT` — local model snapshots.

Keep these local directories out of version control and backups shared outside the machine unless the media and model licenses allow it.

## Model and runtime provenance

The current runtime loads models in CPU mode with Transformers and local-only model loading. The validated host used Python 3.12, CPU PyTorch, and no available CUDA/ROCm device. Model confidence is displayed as uncalibrated.

The three course clips took approximately 202 s (tablet/mug), 272 s (backpack), and 254 s (plant) for first memory generation in the recorded CPU run. Restoring the saved course demo in a fresh local store took 0.6–1.5 s per clip and did not run inference. These are this-machine observations, not runtime guarantees. Cached natural-language answers are read from SQLite; frame decoding and rendering still take local time.

The app separates semantic claims from visual checks. A corroborated check applies only to its exact frame. Sparse frame samples, detector misses, and camera movement do not establish continuous visibility, exact disappearance, or object motion.
