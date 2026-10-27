#!/usr/bin/env python3
"""Build actual source-frame stills for the RecallVision V3 static showcase.

Uses only Mixkit source media whose bytes match the frozen V3 course demo pack.
Images are extracted by their stored frame indices, not generated or redrawn.
"""
from __future__ import annotations
import hashlib, json, subprocess, tempfile
from pathlib import Path
from urllib.request import urlopen,Request
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"v3"/"media"
MANIFEST={
 "backpack":("025bb4c46ec15ccfa44dbff8375911d4450ff2a85708ff9cef8e894f3bbc1dd1","https://assets.mixkit.co/njpruhts4frcepd3xnlpi1h7xor0",{75:"backpack-3_128.webp",119:"backpack-4_963.webp",127:"backpack-5_297.webp"}),
 "tablet":("38b6f9fee3c0e22fb7547dfeab92a391ca01476a1da6f1e3d6ed4f807e2a1e38","https://assets.mixkit.co/active_storage/video_items/100072/1719941812/100072-video-720.mp4",{0:"tablet-0_000.webp",127:"tablet-5_297.webp"}),
 "plant":("a6b0fcd0eb43be64ee5525ed5dc2ad90ec3707bacd6318472d122d5f7021e10d","https://assets.mixkit.co/videos/43699/43699-720.mp4",{120:"plant-5_005.webp",200:"plant-8_342.webp"}),
}
def download(url,path):
    h=hashlib.sha256()
    with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=70) as res, open(path,"wb") as out:
        assert res.headers.get_content_type()=="video/mp4",(res.headers.get_content_type(),url)
        while chunk:=res.read(1024*1024):
            out.write(chunk);h.update(chunk)
    return h.hexdigest()
def still(video,index,out):
    # Frame index is the one stored by V3, extracted from the validated original media.
    expression=f"select=eq(n\\,{index})"
    result=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-loglevel","error","-i",str(video),"-vf",expression,"-vsync","0","-frames:v","1","-y",str(out)],capture_output=True,text=True)
    if result.returncode or not out.exists():raise RuntimeError(f"ffmpeg frame {index} failed: {result.stderr[:300]}")
    with Image.open(out) as im:
        im.convert("RGB").save(out.with_suffix(".webp"),"WEBP",quality=86,method=5)
    out.unlink()
def main():
    pack=json.loads(urlopen("https://raw.githubusercontent.com/suibianba12138/RecallVision-Course/dee4f02e314ee845337b271fc613fc8ec80d9e17/v3/demo/course_memory_pack_v1.json",timeout=40).read().decode("utf-8"))
    expected={x["sha256"] for x in pack["videos"]}
    assert expected=={x[0] for x in MANIFEST.values()}
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for key,(sha,url,frames) in MANIFEST.items():
            mp4=Path(temp)/(key+".mp4")
            actual=download(url,mp4)
            if actual!=sha:raise RuntimeError(f"{key} source sha mismatch {actual}")
            print(key,"verified sha256",sha,flush=True)
            for idx,name in frames.items():
                raw=OUT/(name.removesuffix(".webp")+".png")
                still(mp4,idx,raw)
                print("wrote",name,(OUT/name).stat().st_size,flush=True)
    # Two closeups of the actual source frame at index 119. The original saved
    # detector proposals have locations x=0..47 (reference match) and
    # x=1005..1279 (nonmatching object). Crop extra context on each side.
    with Image.open(OUT/"backpack-4_963.webp") as src:
        image=src.convert("RGB")
        if image.size != (1280,720):raise RuntimeError(f"Unexpected dimensions: {image.size}")
        crops={
            "backpack-reference-match.webp":(0,120,260,600),
            "backpack-other-candidate.webp":(900,230,1280,720),
        }
        for name,rect in crops.items():
            clip=image.crop(rect)
            clip.thumbnail((700,540),Image.Resampling.LANCZOS)
            clip.save(OUT/name,"WEBP",quality=89,method=5)
            print("wrote crop",name,flush=True)
    print("Complete: original recorded source frames, not synthetic output")
if __name__=="__main__":main()
