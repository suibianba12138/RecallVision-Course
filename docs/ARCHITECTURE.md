# Architecture and stage boundaries

## Offline memory construction

1. **P1B perception** reads video frames and emits observations. The optional adapter calls Ultralytics YOLO with the `yolo11n.pt` model identifier and the `bytetrack.yaml` tracking configuration. Detection/tracking IDs are short-term identifiers.
2. **P2A-R1 entity consolidation** groups compatible tracklets into persistent object entities within a video. A persistent ID is a project-level association result, not a raw ByteTrack ID.
3. **P2B state memory** converts observations and persistent entities into time-ordered object-state segments.
4. **P2C/P2D event memory** emits only supported zone-transition or move-object events from recorded state/evidence. These stages do not rerun perception.
5. **Evidence references** retain frame indices, timestamps, and source segment references so answers can be inspected.

## Query-time retrieval

The handlers in `recallvision/query/` consume typed state timelines, event records, and (for WHO) authorized attributed-event memory. They do not invoke a detector or infer identity from raw video. `dispatch_query` routes the four frozen query types. Ambiguous targets and missing/unsupported evidence produce explicit fail-closed statuses.

The original internal `course_demo` replay adapter validates hash-bound C1 artifacts and renders static HTML; it is deliberately not an inference entry point. Its adapter source and evidence images are retained only in the local internal course copy, not this public candidate. Some non-image C1 JSON records remain as regression-test inputs, but without the images they do not form a runnable full replay. The `public-demo/` page is a text-only summary of recorded results, not a replay or evidence renderer.

## Included scope

This candidate includes the frozen V1 source modules for perception, persistent entities, timelines, events, four query handlers, evidence rendering, and the associated stage CLIs/configurations. It excludes pose/E1, VLM F1/F2/F3 generation pipelines, experimental annotations/results, S4A, Grounding DINO, SAM2, OSTrack, weights, raw videos, and generated research outputs.
