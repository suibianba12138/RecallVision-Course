# RecallVision course reproduction and video evidence

The repository contains the frozen V1 source, tests, and C1 **precomputed** query and memory JSON. It does **not** ship the three original third-party MP4 inputs or the internal annotated C1 evidence JPGs. Software tests passing is not a successful end-to-end rerun.

## 1. Obtain the EXACT original MP4s

Download each video from its original provider for local course replication. Providers may offer multiple resolutions/re-encodings: a different download can have a different SHA-256, even if it looks identical. The frozen recorded outputs correspond to the exact hash listed below.

| Case | Original source | Duration | Exact recorded SHA-256 | Recorded outcome |
| --- | --- | ---: | --- | --- |
| C1 | [Pixabay 82261 — bottle](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/) | 7.9079 s | `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730` | Four-query development-artifact replay PASS |
| C2 | [Mixkit 5117 — cake](https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/) | 10.041667 s | `c9524bc8dd28b775572b74e9f6d44865c9158d0ce8cdee315462b9b2e2397255` | AMBIGUOUS_TARGET (12 cake candidates) |
| C2-R | [Pexels 6768198 — bottle](https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/) | 29.268333 s | `1b3a3efd32bc2c57bc775f8b9d68b66c25d3c8de264650601aeec22a5cbbf2c8` | AMBIGUOUS_TARGET (621 + 4 bottle observations) |

If the current provider download's SHA-256 differs, report `EXACT_INPUT_UNAVAILABLE`: do not claim a successful identical reproduction. You can still inspect the source and recorded output, but any fresh run with the new encoding is a separate experiment.

### Import into an ignored LOCAL directory

```bash
git clone https://github.com/suibianba12138/RecallVision-Course.git
cd RecallVision-Course
python scripts/import_course_videos.py --help
python scripts/import_course_videos.py --c1 ~/Downloads/your-exact-original-c1.mp4
# Optional, independent:
python scripts/import_course_videos.py --c2 ~/Downloads/your-exact-original-c2.mp4
python scripts/import_course_videos.py --c2r ~/Downloads/your-exact-original-c2r.mp4
```

Only matching videos are copied to `data/local/course_videos/`. Any one case may be imported separately. The script **never** stages, commits, or pushes files and has no `--push` flag. The original experimental MP4 files are available on the project maintainer's computer, but are not in the public checkout.

## 2. Test the source

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

Optional detector/tracker dependencies: `python -m pip install -e '.[perception]'`, with separately obtained Ultralytics model weights and an appropriate PyTorch installation. The public source tests do not recreate the full C1 evidence replay.

## 3. Inspect existing real V1 output

The published C1 records are found in:

- `course_demo/fixture/artifacts/persistent_entities.json` — P2A-R1 associations;
- `course_demo/fixture/artifacts/object_state_timelines.json` — P2B state segments;
- `course_demo/fixture/artifacts/move_object_events.json` — P2D supported movement;
- `course_demo/fixture/queries/` — four stored query results;
- `course_demo/fixture/manifest.json` — source and evidence integrity reference.

Watch the [C1 analysis video](https://github.com/suibianba12138/RecallVision-Course/blob/gh-pages/case-studies/media/c1-analysis.mp4) for an annotated visualization, and inspect the committed `course_demo/fixture/queries/` JSON to see the four precomputed answers. The published video does not run new Python model inference. The annotated supporting JPGs from the full internal C1 replay are absent from the public source-only checkout.

## 4. Honest reproducibility status

| Scope | Current result |
| --- | --- |
| Clone source + run lightweight source tests | Supported (latest maintainer record: 158 passed, 4 skipped) |
| Inspect recorded C1 memory and four answers | Supported using committed JSON |
| Obtain and SHA-verify original input clips | Script supported if provider still offers identical encoding |
| Reproduce full internal C1 evidence/integrity replay from clone alone | NOT SUPPORTED (annotated JPG evidence not included) |
| Fresh end-to-end video QA from clone alone | NOT ESTABLISHED |
| General long-form video QA | NOT VALIDATED |
| Reliable independent-video QA | NOT ESTABLISHED |

## 5. Public display and source provenance

Video source files remain governed by their upstream provider terms; the public repository does not separately redistribute unaltered Pixabay/Pexels clips. A website may show analytical demo recordings created from real perception data with clearly identified origin and transformations where the terms permit it. Merely adding superficial text or a box is not assumed to resolve license restrictions. See `docs/THIRD_PARTY_NOTICES.md`.

A contributor who has permission to publish a particular asset may submit a documented media manifest and a verified demo export as a separate change. Do not silently replace frozen input videos or rewrite recorded measurements.
