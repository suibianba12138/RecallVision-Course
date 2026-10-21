# Third-party components and media

This file records components used or declared by the source project. It is not a complete resolved SBOM: the optional perception requirements are not fully pinned and no platform-specific dependency lock has been generated. No third-party package binaries or standalone third-party implementation source are bundled in this repository.

## Software

- **Ultralytics `8.4.171` (AGPL-3.0):** optional runtime dependency used by the perception adapter. Its tagged source declares AGPL-3.0 and the bundled ByteTrack implementation/configuration carries the Ultralytics AGPL notice. The package metadata declares dependencies including NumPy, Matplotlib, OpenCV, Pillow, PyYAML, Requests, PyTorch, TorchVision, psutil, Polars, NVIDIA ML Python bindings, and Ultralytics THOP. See the [tagged source license](https://github.com/ultralytics/ultralytics/blob/v8.4.171/LICENSE), [tagged dependency metadata](https://github.com/ultralytics/ultralytics/blob/v8.4.171/pyproject.toml), and [Ultralytics license guidance](https://www.ultralytics.com/license). No package source or model weights are vendored by RecallVision.
- **ByteTrack:** RecallVision selects Ultralytics' `bytetrack.yaml`; it does not copy standalone ByteTrack source. The upstream standalone project publishes an [MIT license](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE). Ultralytics' bundled tracker implementation is covered by the Ultralytics license header above.
- **OpenCV:** optional dependency `opencv-python>=4.11`. The [Python wrapper project](https://pypi.org/project/opencv-python/) is MIT-licensed; OpenCV itself is Apache-2.0 for current 4.x releases. Its upstream documentation says wheels include FFmpeg under LGPL-2.1, and non-headless Linux/macOS wheels also include Qt 5 under LGPL-3.0; other bundled components have platform-specific notices. See the package's [third-party license inventory](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt). RecallVision does not redistribute these wheels, and the exact installed wheel is not pinned here.
- **Pytest:** optional test dependency `pytest>=7`, MIT; see the [upstream license](https://github.com/pytest-dev/pytest/blob/main/LICENSE).
- **Setuptools:** build requirement `setuptools>=61`; dependency only, not bundled. See [Setuptools licensing](https://github.com/pypa/setuptools/blob/main/LICENSE).
- **PyTorch/TorchVision:** transitive requirements of the optional Ultralytics package, not direct pinned RecallVision dependencies. The official PyTorch repository publishes a BSD-style license; exact versions and the surrounding dependency graph are not locked here, and TorchVision notices should be checked for each supported resolved environment.
- **Transitive dependencies:** Ultralytics' dependency ranges are not frozen into a lockfile, and selected versions/platform wheels may add further dependencies and notices. The current lightweight validation environment does not have Ultralytics, OpenCV, or PyTorch installed. A complete resolved dependency license inventory remains a pre-publication task; this note must not be read as one.
- **Original project source:** no outbound project license is assigned. The course involved five team members, and the candidate contains no record authorizing all contributors' work to be relicensed under AGPL-3.0. See [`LICENSE_TODO.md`](../LICENSE_TODO.md).

## C1 media

The C1 development clip is Pixabay video 82261: <https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/>. The raw clip and all three extracted evidence JPEGs are excluded from the public Git tree. The original local fixture and its images are preserved unchanged for the internal course demo. The public candidate does not claim a full artifact replay without those images. Pixabay's [license summary](https://pixabay.com/service/license-summary/) permits adaptations but prohibits standalone distribution and warns that additional rights may apply; no distribution permission for these particular frames is asserted here.

Visual inspection of the three frames found a bottle and a hand, but no visible face or text identifier. This does not resolve copyright, privacy, or publicity-rights questions.

The independent C2-R Pexels video and the C2 Mixkit video are not included. This notice does not grant rights to any media.
