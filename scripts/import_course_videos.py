#!/usr/bin/env python3
"""Verify and import the frozen course video inputs LOCALLY.

Examples:
  python scripts/import_course_videos.py --c1 /path/to/pixabay.mp4
  python scripts/import_course_videos.py --c2 /path/to/mixkit.mp4 --c2r /path/to/pexels.mp4

This tool NEVER commits or pushes raw third-party videos.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "local" / "course_videos"
CASES = {
    "c1": {
        "name": "c1_bottle.mp4",
        "sha256": "24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730",
        "source": "https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/",
        "duration_seconds": 7.9079,
    },
    "c2": {
        "name": "c2_cake.mp4",
        "sha256": "c9524bc8dd28b775572b74e9f6d44865c9158d0ce8cdee315462b9b2e2397255",
        "source": "https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/",
        "duration_seconds": 10.041667,
    },
    "c2r": {
        "name": "c2r_bottle.mp4",
        "sha256": "1b3a3efd32bc2c57bc775f8b9d68b66c25d3c8de264650601aeec22a5cbbf2c8",
        "source": "https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/",
        "duration_seconds": 29.268333,
    },
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for case in CASES:
        parser.add_argument("--" + case, metavar="PATH",
                            help="Local original MP4 obtained from the listed source")
    args = parser.parse_args()
    selected = [(name, Path(getattr(args,name)).expanduser()) for name in CASES
                if getattr(args,name)]
    if not selected:
        parser.print_help()
        print("\nExpected original video sources and hashes:")
        for name, item in CASES.items():
            print(f"  {name}: {item['source']} sha256={item['sha256']}")
        return 0
    resolved = []
    for name, path in selected:
        if not path.is_file():
            print(f"ERROR: {name}: input not found: {path}", file=sys.stderr)
            return 1
        actual = sha256(path)
        expected = CASES[name]["sha256"]
        if actual != expected:
            print(f"ERROR: {name}: hash mismatch: {actual}; expected {expected}",file=sys.stderr)
            print("Do not use a different resolution, crop or encoding to reproduce the frozen run.",
                  file=sys.stderr)
            return 1
        resolved.append((name, path.resolve(), actual))
    DEST.mkdir(parents=True,exist_ok=True)
    manifest_file = DEST / "manifest.json"
    manifest = {"schema":"recallvision_course_original_inputs_v1","clips":{}}
    if manifest_file.exists():
        try:
            prior = json.loads(manifest_file.read_text(encoding="utf-8"))
            if prior.get("schema") == manifest["schema"]:
                manifest["clips"].update(prior.get("clips",{}))
        except (OSError,ValueError):
            pass
    for name, source, digest in resolved:
        target = DEST / CASES[name]["name"]
        if source != target.resolve():
            shutil.copy2(source,target)
        manifest["clips"][name] = {
            "path":str(target.relative_to(ROOT)),
            "sha256":digest,
            "source_page":CASES[name]["source"],
            "recorded_duration_seconds":CASES[name]["duration_seconds"],
        }
        print(f"VERIFIED: {name} -> {target.relative_to(ROOT)} [{digest}]")
    manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("LOCAL_IMPORT=PASS; no Git commit, push, video transformation or inference performed.")
    print("Original videos are in data/local/ (gitignored); share official source links for reproduction.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
