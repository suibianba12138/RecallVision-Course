# Classroom video assets and reproduction

A public source checkout alone does **not** recreate the original demonstrations unless the input videos and evidence data are present. The original published tree contained C1 structured-memory JSON but no MP4.

## The three video inputs

| Case | Expected path after import | Source | Recorded result |
| --- | --- | --- | --- |
| C1 | `course_demo/videos/c1_bottle.mp4` | [Pixabay video 82261](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/), 7.9079 seconds | Frozen four-query artifact replay **PASS**. |
| C2 | `course_demo/videos/c2_cake.mp4` | [Mixkit 5117](https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/), ~10 seconds | **AMBIGUOUS_TARGET**, 12 cake candidates. |
| C2-R | `course_demo/videos/c2r_bottle.mp4` | [Pexels 6768198](https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/), ~29.268 seconds | **AMBIGUOUS_TARGET**, bottle candidates 621 and 4 observations. |

The original C1 source is expected at `/home/nsk/RecallVision/data/local/p1b_pixabay_bottle.mp4`. Its SHA-256 must equal `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730`. Do not substitute a different resolution, trim, or clip for C1 while claiming the frozen results.

To publish the local originals into this course repository:

```bash
cd /home/nsk/RecallVision-Course
mkdir -p course_demo/videos
cp /home/nsk/RecallVision/data/local/p1b_pixabay_bottle.mp4 course_demo/videos/c1_bottle.mp4
sha256sum course_demo/videos/c1_bottle.mp4
# Also copy the exact Mixkit/Pexels input videos used for C2 and C2-R:
# cp /path/to/original/cake.mp4 course_demo/videos/c2_cake.mp4
# cp /path/to/original/pexels.mp4 course_demo/videos/c2r_bottle.mp4
git add course_demo/videos
git diff --cached --stat
git commit -m "media: add three original classroom demo clips"
git push origin main
```

The `.gitignore` whitelist admits only the named course videos; research outputs and other video formats remain excluded. The clips should be in normal Git, not Git LFS, so that a normal `git clone` downloads them. Individual media files must be below GitHub's 100 MB limit.

## What is already reproducible without the clips?

Clone and run source tests:

```bash
git clone https://github.com/suibianba12138/RecallVision-Course.git
cd RecallVision-Course
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

For the optional model-dependent perception adapter, follow `README.md` instructions for `.[perception]` and appropriate model weights.

The C1 `course_demo/fixture/queries/*.json` files hold genuine recorded answers, and `course_demo/fixture/artifacts/*.json` stores P2A/P2B/P2D information. The original annotated evidence JPGs remain absent, so **the full original artifact-integrity replay is not yet reproducible**. The public project page visualizes frozen C1 results rather than running fresh inference. C2 and C2-R demonstrate documented ambiguous identity resolution, not successful new-video QA.

Video publication and a complete clean-clone end-to-end rerun are distinct acceptance checks; neither should be marked PASS solely because the software tests pass.
