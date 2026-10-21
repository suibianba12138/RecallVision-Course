# Source provenance

- Frozen V1 method source: commit `bc9ca6370962df108bd44d4a50cbdf2132234e8a` (`fix: route V1 where query through state memory`).
- Frozen V1 runtime modules, stage CLIs, configurations, and selected unit tests in this candidate were exported from that commit without algorithm edits.
- The internal C1 replay adapter (`recallvision/course_demo.py`, `recallvision/cli/course_demo.py`) and fixture were copied from the completed course repository's C1 artifact fixture. The adapter performs hash validation and HTML rendering only; it does not alter V1 outputs. The adapter source and evidence images are preserved in the local internal course copy and excluded from this public candidate. Non-image C1 JSON records remain as regression-test inputs; they are not a complete replay fixture.
- The public `public-demo/` page is a text-only summary transcribed from recorded C1/C2/C2-R results. It does not run the adapter, validate fixture integrity, or display supporting frames.
- C1 fixture mode: `EXISTING_ARTIFACT_REPLAY`; source video SHA256: `24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730`.
- No original Git history, Git remote, model checkpoint, raw video, generated research output, or S4A artifact is included.
- The candidate is initialized as a separate repository. The frozen source repository is not modified by this preparation. The three Pixabay-derived C1 evidence images remain unchanged in the local internal copy but are excluded from the public Git tree.
