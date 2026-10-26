"""Review, confirm, revoke, and rebuild persistent identity relations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from recallvision.memory.identity_overlay import IdentityOverlay, OverlayError
from recallvision.memory.trust_review import TrustReviewError, TrustReviewOverlay
from recallvision.memory.schema import BoundingBox, Observation
from recallvision.pipeline.identity_overlay_rebuild import IdentityRebuildError, rebuild_visual_memory
from recallvision.visualization.evidence import render_evidence_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("candidates", "inspect", "evidence", "confirm", "revoke", "history", "mapping", "rebuild", "trust-evidence"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--ledger", type=Path, required=True)
        command.add_argument("--source-hash-manifest", type=Path, help="Optional sha256sum manifest for source artifacts not pinned by the run manifest.")
        if name == "candidates":
            command.add_argument("--class-name")
        elif name == "inspect":
            command.add_argument("--entity-a", required=True)
            command.add_argument("--entity-b", required=True)
        elif name == "evidence":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--entity-id", help="Render one persistent entity for the candidate gallery.")
            selection.add_argument("--entity-a", help="First entity in a reviewed pair.")
            command.add_argument("--entity-b", help="Second entity in a reviewed pair; required with --entity-a.")
            command.add_argument("--output-dir", type=Path, required=True)
        elif name == "confirm":
            command.add_argument("--entity-a", required=True)
            command.add_argument("--entity-b", required=True)
            command.add_argument("--relation", choices=("SAME_INSTANCE", "DIFFERENT_INSTANCE"), required=True)
            command.add_argument("--evidence-frame", type=int, action="append", required=True)
            command.add_argument("--evidence-path", type=Path, action="append", default=[])
            command.add_argument("--expected-revision", type=int, required=True)
        elif name == "revoke":
            command.add_argument("--operation-id", required=True)
            command.add_argument("--expected-revision", type=int, required=True)
        elif name == "rebuild":
            command.add_argument("--output-root", type=Path, required=True)
            command.add_argument("--question", action="append", default=[], help="Supported question; repeat for multiple queries.")
            command.add_argument("--target-class")
            command.add_argument("--target-entity-id")
            command.add_argument("--trust-ledger", type=Path, help="Optional append-only trusted-observation/event review ledger.")
        elif name == "trust-evidence":
            command.add_argument("--trust-ledger", type=Path, required=True)
            command.add_argument("--observation-id", required=True)
            command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overlay = IdentityOverlay(args.run_dir, args.ledger, args.source_hash_manifest)
        if args.command == "candidates":
            snapshot = overlay.snapshot()
            result: Any = {
                "overlay_revision": snapshot["overlay_revision"],
                "entities": overlay.list_entities(args.class_name, snapshot=snapshot),
            }
        elif args.command == "inspect":
            snapshot = overlay.snapshot()
            result = overlay.inspect_pair(args.entity_a, args.entity_b, snapshot=snapshot)
        elif args.command == "evidence":
            if args.entity_id:
                result = {"status": "RENDERED", "mode": "ENTITY", "evidence": overlay.render_entity_evidence(args.entity_id, args.output_dir)}
            elif args.entity_a and args.entity_b:
                result = {"status": "RENDERED", "mode": "PAIR", "evidence": overlay.render_pair_evidence(args.entity_a, args.entity_b, args.output_dir)}
            else:
                raise OverlayError("INVALID_EVIDENCE_REQUEST", "provide --entity-id or both --entity-a and --entity-b")
        elif args.command == "confirm":
            if args.evidence_path and len(args.evidence_path) not in {1, len(args.evidence_frame)}:
                raise OverlayError("INVALID_EVIDENCE", "provide one evidence path or one path per frame")
            refs = []
            for index, frame in enumerate(args.evidence_frame):
                evidence_path = None
                if args.evidence_path:
                    evidence_path = args.evidence_path[0 if len(args.evidence_path) == 1 else index]
                    if not evidence_path.is_file():
                        raise OverlayError("INVALID_EVIDENCE", "evidence image does not exist", {"evidence_path": str(evidence_path)})
                refs.append({"video_id": overlay.video_id, "frame_index": frame, "evidence_path": str(evidence_path) if evidence_path else None})
            result = overlay.record_relation(
                args.entity_a, args.entity_b, args.relation,
                evidence_refs=refs, expected_revision=args.expected_revision,
            )
        elif args.command == "revoke":
            result = overlay.revoke(args.operation_id, expected_revision=args.expected_revision)
        elif args.command == "history":
            result = overlay.history()
        elif args.command == "mapping":
            snapshot = overlay.snapshot()
            result = {"overlay_revision": snapshot["overlay_revision"], "mapping": snapshot["effective_mapping"]}
        elif args.command == "rebuild":
            result = rebuild_visual_memory(
                run_dir=args.run_dir,
                ledger_path=args.ledger,
                output_root=args.output_root,
                source_hash_manifest=args.source_hash_manifest,
                questions=args.question,
                target_class=args.target_class,
                target_entity_id=args.target_entity_id,
                trust_ledger_path=args.trust_ledger,
            )
        elif args.command == "trust-evidence":
            trust = TrustReviewOverlay(
                args.run_dir, args.trust_ledger,
                identity_ledger_path=args.ledger,
                source_hash_manifest=args.source_hash_manifest,
            )
            try:
                args.output.resolve().relative_to(trust.run_dir)
            except ValueError:
                pass
            else:
                raise TrustReviewError("OUTPUT_IN_SOURCE_RUN", "evidence image must be written outside the frozen source run")
            if args.output.exists():
                result = {"status": "CACHED", "evidence_path": str(args.output)}
            else:
                candidate = trust.observation_by_id.get(args.observation_id)
                if candidate is None:
                    raise TrustReviewError("OBSERVATION_NOT_FOUND", "observation reference is not present in the frozen run")
                raw = candidate["raw_observation"]
                box = raw["bbox"]
                observation = Observation(
                    video_id=raw["video_id"], frame_index=int(raw["frame_index"]),
                    timestamp=float(raw["timestamp"]), track_id=raw["track_id"],
                    persistent_id=raw.get("persistent_id"), entity_type=raw.get("entity_type", "object"),
                    class_name=raw["class_name"],
                    bbox=BoundingBox(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                    confidence=raw.get("confidence"), zone=raw.get("zone"),
                    source_track_ids=tuple(raw.get("source_track_ids", [raw["track_id"]])),
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                rendered = render_evidence_frame(trust.identity.video_path, observation, args.output)
                result = {"status": "RENDERED", "evidence_path": str(rendered)}
        else:  # pragma: no cover - argparse constrains this branch
            raise OverlayError("INVALID_COMMAND", f"unsupported command: {args.command}")
    except (OverlayError, TrustReviewError, IdentityRebuildError) as error:
        status = getattr(error, "status", "FAILED")
        details = getattr(error, "details", {})
        print(json.dumps({"status": status, "error": str(error), "details": details}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
