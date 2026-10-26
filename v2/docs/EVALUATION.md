# RecallVision 2.0 — Evaluation and Limitations

**Source:** frozen V2 archive at `ae0a65a21e02366453a9612cb620649f3a0bac3f`. This document summarizes the existing release notes and D1/D2/E1 reports; no new inference or independent accuracy benchmark is implied.

## Implementation and regression

The original V2 development worktree recorded **416 passed / 7 skipped / 0 failed** regression results. That run used externally preserved development fixtures and temporary read-only fixture links. It is **not** proof that `pytest` in a fresh public clone produces the same count.

D2 integrated the local Streamlit application with safe video import, re-openable analyses, candidate review, and trusted WHERE / WHEN / BEFORE query routes. Human acceptance of visual decisions and complete browser UI validation were pending.

## Application runs

| Source | Recorded completion | Relevant observations | Query outcome |
| --- | --- | --- | --- |
| C1 bottle (7.91 s) | 45.97 s | 577 total; 50 bottle; 1 persistent bottle candidate; 1 event candidate | WHERE insufficient trusted evidence; WHEN and BEFORE require event review |
| P1-B coffee cup (15.43 s) | 90.65 s | 3,268 total; 752 cup; 40 persistent entities; 7 cup candidates; 2 event candidates across the run | Selected cup ambiguous/unreviewed; trusted queries withheld |

These are successful **pipeline executions**, not measurements of automatic understanding accuracy.

## Independent E1 diagnosis

Four separate videos with five annotated targets were evaluated. The archived automatic pipeline resolved **0/5** targets; movement evidence contained **four visually confirmed false-positive event candidates**, and **0/3** tested same-zone movement cases were accepted correctly. Important contributing factors included limited COCO-class coverage, identity fragmentation, unstable localization and false movement triggers.

## Known limitations

- Vocabulary limited to supported COCO categories; target instances may be missed or split.
- Identity suggestions and motion-event candidates require review and may be wrong.
- Long gaps and camera movement make continuity and movement evidence unreliable.
- Saved query statuses should not be treated as independent-video correctness metrics.
- Browser visual QA and fully independent clean-clone regression were not established in the source archive.
- Original ignored run outputs, video files, model weights, and local review databases are not included in the public source.

The [V2 website](https://suibianba12138.github.io/RecallVision-Course/v2/) shows archived metrics and a read-only workflow illustration. Its source video panel uses a previously published V1 annotated composite from the same C1 source. It does not depict new V2 model inference.
