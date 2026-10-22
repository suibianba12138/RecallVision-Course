#!/usr/bin/env python3
"""Import exact course videos from the maintainer's local machine.

Run: python scripts/import_course_videos.py --push
Or: python scripts/import_course_videos.py --c1 /path/a.mp4 --c2 /path/b.mp4 --c2r /path/c.mp4 --push
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
C1_HASH = "24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730"
FILES = {"c1": "c1_bottle.mp4", "c2": "c2_cake.mp4", "c2r": "c2r_bottle.mp4"}
URLS = {
    "c1": "https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/",
    "c2": "https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/",
    "c2r": "https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/",
}
RANGES = {"c1": (7.7, 8.2), "c2": (8., 12.), "c2r": (28.5, 30.1)}

def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def probe_duration(p):
    if not shutil.which("ffprobe"):
        return None
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(p)],capture_output=True,text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None

def locate(key, override):
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"{key}: missing file: {p}")
        return p
    if key == "c1":
        p = HOME / "RecallVision/data/local/p1b_pixabay_bottle.mp4"
        if p.is_file() and sha256(p) == C1_HASH:
            return p
    roots = [HOME/"RecallVision/data",HOME/"RecallVision/outputs",HOME/"recallvision-course-final",HOME/"Downloads",HOME/"下载"]
    matches = []
    for root in roots:
        if root.is_dir():
            matches += [p for p in root.rglob("*.mp4") if
                        (("pixabay","bottle") if key=="c1" else ("mixkit","cake") if key=="c2" else ("pexels","bottle"))[0] in p.name.lower() or
                        (("pixabay","bottle") if key=="c1" else ("mixkit","cake") if key=="c2" else ("pexels","bottle"))[1] in p.name.lower()]
    matches = sorted(set(matches))
    if key == "c1":
        matches = [p for p in matches if sha256(p) == C1_HASH]
    else:
        valid = [p for p in matches if (d:=probe_duration(p)) is not None and RANGES[key][0] <= d <= RANGES[key][1]]
        if valid: matches = valid
    if len(matches) != 1:
        raise RuntimeError(f"{key}: unable to select unique source MP4. Use --{key} /absolute/path/file.mp4; candidates: {matches[:12]}")
    return matches[0]

def main():
    pa = argparse.ArgumentParser(description=__doc__)
    for key in FILES: pa.add_argument("--"+key)
    pa.add_argument("--push",action="store_true")
    args = pa.parse_args()
    if ROOT.name != "RecallVision-Course":
        pa.error("run from a checked-out RecallVision-Course repository")
    if args.push:
        branch = subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
        if branch != "main": pa.error("refusing to push outside main")
        if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
            pa.error("working tree must be clean before --push")
    dest = ROOT / "course_demo/videos"
    dest.mkdir(parents=True,exist_ok=True)
    manifest = {"schema":"recallvision_course_media_v1","clips":{}}
    for key, name in FILES.items():
        p = locate(key,getattr(args,key))
        digest = sha256(p)
        if key == "c1" and digest != C1_HASH:
            raise RuntimeError("C1 SHA256 mismatch: not frozen experimental input")
        duration = probe_duration(p)
        if duration is not None and not (RANGES[key][0] <= duration <= RANGES[key][1]):
            raise RuntimeError(f"{key}: unexpected video duration {duration:.2f} s")
        if p.stat().st_size >= 100_000_000:
            raise RuntimeError(f"{key}: file >= 100MB; GitHub won't accept direct upload")
        target = dest/name
        if p.resolve() != target.resolve():
            shutil.copy2(p,target)
        manifest["clips"][key] = {
            "file":"course_demo/videos/"+name,
            "sha256":digest,
            "size_bytes":target.stat().st_size,
            "duration_seconds":duration,
            "source_page":URLS[key]
        }
        print(f"{key}: {target} ({target.stat().st_size} bytes); sha256={digest}")
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    if args.push:
        for command in [
            ["git","add","course_demo/videos"],
            ["git","diff","--cached","--check"],
            ["git","commit","-m","media: add three course video inputs"],
            ["git","push","origin","main"]
        ]: subprocess.run(command,cwd=ROOT,check=True)
        print("SUCCESS: video inputs published to main")
    else: print("LOCAL_ONLY: to publish, re-run with --push")

if __name__=="__main__":
    main()
