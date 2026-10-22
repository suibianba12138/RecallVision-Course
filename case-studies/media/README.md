# RecallVision case-study analysis videos

These three silent H.264 composites are served by the case-study page. Each combines its identified source video with recorded, time-indexed detector/tracker/persistent-entity observations and a case-specific diagnostic panel. The bounding boxes and labels are rendered only at the exact recorded `frame_index`; no boxes are interpolated, predicted, or added to frames without a matching observation. These are visualizations of frozen artifacts, not new model runs.

The repository publishes these analysis composites, not the original source MP4 files or the non-public observation JSONL artifacts. The source links and SHA-256 values below make the experimental inputs traceable. Provider terms were checked on 2026-10-22; anyone reusing the underlying clips should consult the current provider terms and any third-party rights that may apply.

## Source and export provenance

| Case | Source | Source SHA-256 | Recorded data | Outcome shown |
|---|---|---|---|---|
| C1 | [Pixabay video 82261](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/) | `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730` | P2A-R1 canonical observations; 48 bottle observations on 48 frames; artifact SHA-256 `8db7412fd859d09ac821548710c7ea68ecf65712ff7778ad5426da980c42bdc6` | One persistent bottle ID from seven observed track IDs. A recorded `MOVE_OBJECT` links left to center using evidence frames 92 and 99 (3.069733–3.303300 s); this is an evidence interval, not an exact physical onset. C1 remains a development artifact replay, not fresh inference. |
| C2 | [Mixkit video 5117](https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/) | `c9524bc8dd28b775572b74e9f6d44865c9158d0ce8cdee315462b9b2e2397255` | P2A-R1 observations with persistent IDs; 579 cake observations; artifact SHA-256 `ed951f3b958a0fac6c6fabb0b937f4b7726eadf3599aac171817343752563f76` | Fourteen short-term track IDs produced 12 candidate identities. Tracks 17+57 join as `obj:cake:0006`, tracks 36+39 join as `obj:cake:0007`, and ten candidates each retain one track. These are not 12 verified physical cakes. P2B is `AMBIGUOUS_TARGET`; no positive downstream query or event is claimed. |
| C2-R | [Pexels video 6768198](https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/) | `1b3a3efd32bc2c57bc775f8b9d68b66c25d3c8de264650601aeec22a5cbbf2c8` | P2A-R1 observations with persistent IDs; 625 bottle observations; artifact SHA-256 `191dabce0af1ae635c279ee4a459b8d54c2826a011d34f0dad6b01a0cc45324f` | Track 11 maps to `obj:bottle:0001` (621 observations); track 14 maps to `obj:bottle:0002` (4). No relation provenance is recorded between the candidates. P2B is `AMBIGUOUS_TARGET`; no downstream event or successful QA is claimed. |

Each output is 1280×720, H.264, `yuv420p`, silent, and retains the complete decoded source frame count. Exact output hashes, frame rates, durations, and sizes are in [`manifest.json`](./manifest.json). The C2-R input is 4096×2160; it is proportionally scaled and padded in the composite, not cropped to hide content.

## Provider terms and attribution

- Pixabay's [Content License Summary](https://pixabay.com/service/license-summary/) permits modifying or adapting content into new works and prohibits standalone redistribution. The C1 file here is a composite with recorded visual annotations and structured analysis, not the unaltered MP4.
- Mixkit's [Stock Video Free License](https://mixkit.co/license/modal/videoFree/) lists educational use and permits modifying and distributing items, subject to its terms.
- Pexels' [Terms of Service](https://www.pexels.com/terms-of-service/) permit modifying/adapting licensed content but prohibit standalone redistribution; the terms describe combinations with text, other media, and editing that create a new work as non-standalone use. The C2-R file is an evidence-linked analysis composite, not the original MP4 or a mere crop/filter.

The provider/source links above are included for traceability and credit. Their presence does not imply endorsement by any provider or person depicted. Provider terms can change; consult the linked current terms before reusing these assets.
