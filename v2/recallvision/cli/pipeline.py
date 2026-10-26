"""Run the unified RecallVision offline video pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recallvision.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one local video with the configured COCO detector and existing "
            "RecallVision memory, event, and query stages."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="input video file")
    parser.add_argument("--output-root", type=Path, required=True, help="root for unique run directories")
    parser.add_argument("--target-class", help="COCO class to resolve for a query")
    parser.add_argument("--target-entity-id", help="explicit persistent entity ID from this run")
    parser.add_argument(
        "--question",
        dest="questions",
        action="append",
        default=None,
        help="supported WHERE/WHEN/BEFORE question; repeat to run multiple queries",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="path to an existing local model checkpoint")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-url", default=None)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.questions and len(arguments.questions) > 1:
        query_arguments = {"questions": arguments.questions}
    else:
        query_arguments = {"question": arguments.questions[0] if arguments.questions else None}
    result = run_pipeline(
        video_path=arguments.video,
        output_root=arguments.output_root,
        target_class=arguments.target_class,
        target_entity_id=arguments.target_entity_id,
        **query_arguments,
        model_path=arguments.model,
        tracker=arguments.tracker,
        confidence_threshold=arguments.conf,
        imgsz=arguments.imgsz,
        device=arguments.device,
        source_url=arguments.source_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"COMPLETED": 0, "FAILED": 2, "PARTIAL": 3}.get(result.get("run_status"), 2)


if __name__ == "__main__":
    raise SystemExit(main())
