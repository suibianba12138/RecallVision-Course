# RecallVision C1 analysis video

This directory publishes one annotated C1 course demonstration as H.264/yuv420p MP4. The video combines the identified source footage with time-indexed detector, tracker, persistent-entity and event annotations from recorded artifacts. Boxes and identity labels occur only on frames with matching observations; no model inference, interpolation or new tracking was performed for this export.

## Provenance and observed result

- **Input:** [Pixabay bottle video 82261](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/)
- **Input SHA-256:** `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730`
- **Recorded C1 observations:** 48 canonical bottle observations across 48 frames; source observation-artifact SHA-256: `8db7412fd859d09ac821548710c7ea68ecf65712ff7778ad5426da980c42bdc6`
- **Identity:** seven recorded short-term tracker IDs mapped to `obj:bottle:0001`.
- **Memory:** recorded `MOVE_OBJECT` evidence connects left and center state segments at frames 92 and 99, over 3.069733–3.303300 s. This is an observation-bounded interval, **not** an exact movement onset.
- **Output:** [c1-analysis.mp4](./c1-analysis.mp4), 1280×720 H.264/yuv420p, 237 frames, silent; output SHA-256 `2d38e240c4f26e044cfb6ccc8869253fb08148f056dad7a84a7120446eb565a3`.
- **Query support:** recorded artifact replay, not fresh video inference; consult `main/course_demo/fixture/queries/` for the four supported query records.

See [manifest.json](./manifest.json) for complete hash and frame metadata.

## Source and license

Pixabay's [content-license summary](https://pixabay.com/service/license-summary/) permits use in adaptations but restricts standalone redistribution. This published file is an annotated analytical composite rather than the unmodified source MP4. This description is a technical record, not legal advice; anyone reusing source media must confirm the applicable rights independently.
