# Public V3 RC1 course release boundaries

This folder is a standalone course delivery copy of the V3 video-memory application from private source commit `dbb5a1022b669df520349c2fd38ad5088d39a962`. It preserves the V3 runtime, persisted demonstration importer, real recorded memory pack, relevant tests, and user documentation. Legacy experiments, annotations, private run outputs, and earlier V2 research worktrees are not copied.

## Verification scope

The private RC1 handoff reported 61 focused tests passed and AppTest for three restored clips. The source private full test result was 530 passed, 5 skipped, 2 deselected, 3 failed (legacy frozen V2 UI tests with a NumPy/PyArrow environment incompatibility). **Those historical counts do not establish fresh-public-clone parity.** The V3 course CI tests source imports, packaging, syntax, and isolated non-inference checks only. Full model initialization, new-video inference, and browser visual review remain unverified in the released copy.

The published demo is reproducible only after the operator downloads the exact source-licensed Mixkit clip variant. The importer refuses mismatched hashes and does not generate new model conclusions. Frame-local corroboration and UNKNOWN must not be presented as ground-truth disappearance or identity certainty.

For install and usage see [README](../README.md); for demonstration evidence see [demo instructions](../demo/README.md).
