# Third-Party Notices — RecallVision V2

The project code is published under [AGPL-3.0-only](../../LICENSE). Third-party code, models and media are **not** relicensed by this notice.

| Component | Role | Distribution in this repository |
| --- | --- | --- |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | Detection and ByteTrack integration | Dependency only; model weights are not included. Review upstream AGPL/commercial terms. |
| [PyTorch](https://pytorch.org/) | Model runtime | External dependency; not vendored. |
| [OpenCV](https://opencv.org/) | Video decoding and frame processing | External dependency; not vendored. |
| [Streamlit](https://streamlit.io/) | Local review UI | External dependency; not vendored. |
| [Pixabay bottle video](https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/) | Archived C1 case | This `v2/` folder contains no video; the previously published analysis composite is served by the separate Pages branch. |
| [Mixkit coffee-cup video](https://mixkit.co/free-stock-video/) | D2 development case | Video excluded; license/source reference is documented separately in the source archive. |

Users are responsible for complying with applicable upstream licenses before redistributing model weights or third-party media. Generated user videos, frame outputs, and review databases should be kept private unless separately authorized.

No API credentials or paid services are required for the V2 local application.
