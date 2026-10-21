# Course validation results

These are the recorded course outcomes; they are not recomputed by the public text-only demo page. C1 is an internal development-artifact replay result. The public Git tree omits its three Pixabay-derived supporting images, so the full C1 replay cannot be reproduced from this checkout.

| Validation | Recorded result | Boundary |
|---|---|---|
| C1 | `PASS` — four-query artifact replay | Uses existing development artifacts and their supporting frames; not fresh-video inference. |
| C2 | `AMBIGUOUS_TARGET` at P2B | Independent cake video; multiple persistent entities matched the target, so downstream event/query success was not established. |
| C2-R | `AMBIGUOUS_TARGET` at P2B | Independent Pexels bottle video; two persistent bottle entities remained. P2C/P2D were not run and no movement event was generated. |

The internal C1 fixture's recorded answers are:

- `WHERE_OBJECT`: last observed in `center_region`, about 5.0717 s, frame 152.
- `WHEN_OBJECT_MOVED`: supported interval about 3.0697–3.3033 s, frames 92–99.
- `WHERE_OBJECT_BEFORE`: observed in `left_region` at about 3.0697 s, frame 92.
- `WHO_MOVED_OBJECT`: `obj:person:0001`, from an existing pre-authorized event-local attribution record. This is not automatic VLM attribution.

Overall claim limits:

- Fresh-video end-to-end QA success: **NOT ESTABLISHED**.
- Long-form performance/generalization: **NOT VALIDATED**.
- Cross-video robustness: **NOT ESTABLISHED**.
- Automatic VLM attribution: **NOT ESTABLISHED**.
- Open-vocabulary generalization: **NOT ESTABLISHED**.
- Reliable dataset-level end-to-end accuracy: **NOT ESTABLISHED**.

The independent-video failures remain visible; they are not replaced with C1 results.

## Candidate software checks

The sanitized public Git tree, exported from the final candidate commit, passed `158 passed, 4 skipped`. Two OpenCV tests skip because optional `cv2` is not installed in the lightweight validation runtime; one P2D CLI integration test requires generated local P1B/P2A/P2B/P2C outputs excluded from this candidate; one P3C CLI integration test requires development F3/P2A artifacts. No inference or training was run. Separately, the full internal P0 fixture replay returned `artifact_integrity_status=PASS` and all four expected query types. That internal result is not a claim about replaying from the asset-restricted public Git tree; the public static summary is not a replay.
