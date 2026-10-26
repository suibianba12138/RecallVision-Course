# RecallVision V2 — Local Application Guide

## Installation

From the public repository:

```bash
cd v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,perception]"
```

Place a licensed YOLO11n checkpoint at `models/yolo11n.pt` (excluded from Git) or configure `RECALLVISION_MODEL_PATH`. The default backend uses the same interpreter as the UI. To use a separate perception environment, set `RECALLVISION_BACKEND_PYTHON` to its Python executable.

Start with `python -m streamlit run recallvision/apps/identity_review.py`.

## Analyze / Explore / Ask

1. **Analyze:** select a locally supported video, review metadata and start processing. Supported size ≤250 MiB, duration ≤120 seconds. Alternatively reopen an earlier completed run.
2. **Explore:** select an entity hypothesis; inspect original observations and identity fragments. Review exact frames (`SUPPORTED`, `UNCERTAIN`, `MISSING`) and movement candidates. Decisions are append-only and reversible.
3. **Ask:** use the supported WHERE, WHEN and WHERE-BEFORE queries. Unreviewed or ambiguous evidence produces explicit refusal/uncertainty rather than a fabricated answer.

For consistent behavior, operate from `v2/` so the correct package and configs are selected.

## Local network exposure

The committed Streamlit configuration binds the service to `127.0.0.1`. Open it from the same machine (the default Streamlit port is 8501). Do **not** override the bind address to a public interface or expose this service to untrusted networks: the application has no account-based authentication or multi-user isolation, and video parsing uses native decoding libraries. This configuration does not replace OS permissions or sandboxing.

## Storage and security

- `outputs/`, `models/`, local uploads, secrets and raw media are excluded from Git.
- Input paths and subprocess calls are handled by the controlled application manager; do not expose the development Streamlit server directly to the public Internet.
- A review action does not certify a whole interval. Only inspected exact frames and explicitly reviewed events may support trusted queries.
- The static project website shows recorded results but is not a live inference endpoint.

See [architecture](ARCHITECTURE.md) and [evaluation](EVALUATION.md).
