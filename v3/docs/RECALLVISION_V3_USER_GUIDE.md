# RecallVision V3 User Guide

## Start

Follow the repository [installation instructions](../README.md#install), configure the model paths in [Local video memory setup](LOCAL_VIDEO_MEMORY.md), and start the app:

```bash
python -m streamlit run recallvision/apps/video_memory.py --server.address 127.0.0.1 --server.port 8504
```

Open <http://127.0.0.1:8504> in a browser on the same computer. The database, original videos, evidence frames, and caches stay in the local data directories.

## Upload and analyze

1. Select **Choose a video** and upload a short local clip. The app accepts MP4, MOV, M4V, AVI, or WEBM files up to 250 MiB and 180 seconds.
2. Watch the Analyze status while frames are decoded, model analysis runs, and memory is saved. A first Qwen pass can take several minutes on CPU.
3. Check the video duration, dimensions, memory count, and completion state. Failed or partial generation is labeled and can be retried.

An uploaded video is copied into the persistent memory store. Re-uploading the same bytes reuses their content hash and saved memory version.

## Explore

Open **Explore** to browse object observations, scene claims, event candidates, and saved source frames. Each record distinguishes its semantic claim from the evidence status:

- **UNVERIFIED** — a model-generated claim with no confirming visual check.
- **CORROBORATED** — a targeted crop review supports presence in that exact sampled frame; it does not prove that later frames show the same object.
- **CONFLICTED** — visual checks disagree or identity is ambiguous.
- **UNKNOWN** — evidence did not settle visibility. A detector miss is not proof of absence.

Use **Play** or **Play from** next to a timestamp to restart the source video near that frame. Timestamps refer to sampled source frames, not exact event boundaries.

## Ask

Open **Ask**, enter a natural-language question, and select **Analyze question**. The app checks persistent memory and saved answers first. It calls a local model only when the current evidence path needs it. The answer panel shows its source path, verification state, uncertainty, and linked source frames.

Questions with cached demo answers are listed in [`demo/README.md`](../demo/README.md). For a new question, use the linked frames to confirm whether the memory supports the answer. A last-seen answer reports the latest supported sampled view and keeps later unknown intervals separate from a confirmed disappearance.

Repeated questions with the same video, model, and evidence version reuse saved results. A model or evidence version change can invalidate that cache so stale answers are not silently reused.

## Reopen

After restarting the app, choose a video under **Or reopen a saved video**. The app restores its SQLite memories, saved answers, source clip, and evidence frames without regenerating completed memory. Course-demo entries are marked **Course demo** and state that restoration did not run models.

The default persistent directory is `$XDG_DATA_HOME/recallvision-v3`, or `~/.local/share/recallvision-v3` when `XDG_DATA_HOME` is unset. Set `RECALLVISION_MEMORY_DIR` to select a different directory. The video-analysis cache defaults to the system temporary directory and can be moved with `RECALLVISION_VIDEO_CACHE_DIR`.

## Course demo versus a new upload

The course demo restores a frozen set of real analysis results for exact source-video hashes. It is separate from the normal upload-and-analyze workflow, and the import command never calls a model. See [`demo/README.md`](../demo/README.md) for the source links and restore steps.

For a new clip, use **Choose a video**. A completed course demo is shown in the saved-video selector and can be reopened without rerunning analysis.

## Known limits

Sparse frames cannot establish exact motion boundaries or continuous visibility. An object leaving the camera view is not proof that it was moved or no longer exists. The CPU route may be slow on first analysis, and model claims may be incomplete or wrong. When evidence is inconclusive, the app should preserve uncertainty rather than infer a fact.
