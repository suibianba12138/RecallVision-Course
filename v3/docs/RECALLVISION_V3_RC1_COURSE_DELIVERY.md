# RecallVision V3-RC1 Course Delivery Check

Date: 2026-10-27
Branch: `research/recallvision-v3`

## Demo package

`demo/course_memory_pack_v1.json` was exported from the three completed local V3 runs. It contains the stored memory records, evidence frame indices/timestamps, and one previously saved answer per clip. It includes no original video, SQLite database, evidence image, or model weight. `scripts/import_course_demo.py` checks the source SHA-256, restores the records and answer cache, recreates evidence frames from the supplied source video, and does not load or run a model.

The exact Mixkit item pages currently label the clips under the Stock Video Free License. The pack links to the item pages and license; the source MP4s are downloaded directly by the presenter rather than included in the repository:

- [Panning Shot of College Study Desk (101309)](https://mixkit.co/free-stock-video/panning-shot-of-college-study-desk-101309/)
- [Tablet clip (100072)](https://mixkit.co/free-stock-video/the-hands-of-a-young-individual-take-the-tablet-from-100072/)
- [Gardener changing a pot plant (43699)](https://mixkit.co/free-stock-video/gardener-changing-a-pot-plant-43699/)
- [Mixkit Stock Video Free License](https://mixkit.co/license/modal/videoFree/) · [Mixkit User Terms](https://mixkit.co/terms/)

## Real-video results

| Demo | Stored records | Evidence frames | Restored answer | Evidence state |
|---|---:|---:|---|---|
| Backpack — Moving Camera | 24 | 11 | “Where did I leave my backpack?” | Reference-matched frame candidate at 4.963 s; later sampled checks remain `UNKNOWN`. This is not an exact disappearance time or world location. |
| Tablet & Mug — Fixed Camera | 11 | 6 | “What objects appeared in the video?” | Tablet and mug are model-generated `UNVERIFIED` candidates. |
| Plant Interaction | 11 | 8 | “What happened to the plant?” | Plant-pot interaction is an `UNVERIFIED` event candidate; timestamps are sampled evidence times. |

The 3 source files restored into a fresh isolated store in 2.160 s (backpack), 0.659 s (tablet/mug), and 0.713 s (plant). A second store instance reopened all records, evidence references, and saved answer text. The documented CLI was also run successfully for the backpack source and reported 24 memories, 11 source-aligned evidence frames, one cached answer, and no inference.

First memory generation in the recorded CPU runs took 202.1 s (tablet/mug), 271.7 s (backpack), and 253.8 s (plant). The demo pack removes that first-run wait for these exact source hashes; first-time analysis of a new upload still takes minutes on this host.

## App and tests

- Streamlit started locally on port 19487 and the app workflow was exercised with AppTest for all three restored videos: Analyze state, course-demo provenance, Explore records, saved-answer Ask path, and reopened video data had no app exceptions.
- The backpack AppTest clicked **Play from 4.96 s** and confirmed the app set the player start time to 4.963 s.
- Focused V3 suite: **61 passed** (product logic, persistent memory, memory store, demo restore/CLI, and test-package resolution).
- Full suite: **530 passed, 5 skipped, 2 `historical_artifact` tests deselected, 3 failed**. The three failures are legacy V2 Streamlit AppTests that load the separate frozen V2 checkout and fail when PyArrow requires NumPy ≥2.0 but this shared environment provides NumPy 1.26.4. V2 code and historical test assertions were not changed.
- Browser visual QA and screenshots: **not available**. The browser runtime returned no browser; its troubleshooting procedure also listed an empty browser set. AppTest is reported separately and is not browser acceptance.

## Running

The root README covers Python 3.12 installation, a minimal app-only environment for restoring/browsing the demo, optional pinned local model setup for new-video analysis, and the Streamlit command. The user guide covers Analyze → Explore → Ask → Reopen. `demo/README.md` lists exact media hashes and restore commands.

## Remaining limits

- The three demo answers are real saved outputs, but the tablet/mug inventory and plant event remain unverified. Demonstrate them as candidates, not confirmed facts.
- The backpack result supports a sampled view at 4.963 s; later checks are inconclusive. The system does not establish the precise time it left the camera view.
- CPU first-run analysis is slow. No model inference runs during demo restoration or when the saved demonstration questions hit their stored answer cache.
- Real browser playback/rendering was not visually inspected because the session has no browser. Repeat the visual check in an environment with an available browser before a live course presentation.
- A recipient must download the exact licensed clip variant. The restore command rejects a different SHA-256 so timestamps cannot silently attach to another encoding.
