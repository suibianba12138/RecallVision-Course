#!/usr/bin/env python3
"""Check official Mixkit source pages and 720p media candidates (read-only)."""
import html
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SOURCES = {
    "backpack": ("101309", "https://mixkit.co/free-stock-video/panning-shot-of-college-study-desk-101309/"),
    "tablet": ("100072", "https://mixkit.co/free-stock-video/the-hands-of-a-young-individual-take-the-tablet-from-100072/"),
    "plant": ("43699", "https://mixkit.co/free-stock-video/gardener-changing-a-pot-plant-43699/"),
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RecallVision-Course/1.0)", "Accept": "text/html,application/xhtml+xml,*/*"}

def fetch(url, *, method="GET"):
    req = Request(url, headers=HEADERS, method=method)
    with urlopen(req, timeout=22) as res:
        return res.status, res.geturl(), res.headers, res.read(2_000_000) if method == "GET" else b""

for key, (vid, page) in SOURCES.items():
    print(f"\n=== {key} / Mixkit item {vid} ===", flush=True)
    try:
        status, final_url, _, raw = fetch(page)
        source = html.unescape(raw.decode("utf-8", errors="replace")).replace("\\/", "/")
        print("page status", status, "chars", len(source), "url", final_url)
        urls = sorted(set(re.findall(r'(?:https?:)?//[^\s"<>]+(?:\.mp4|\.webm)(?:\?[^\s"<>]*)?', source)))
        urls += sorted(set(re.findall(r'https?://assets\.mixkit\.co/[^"\s<>]+', source)))[:20]
        print("media references", len(urls), *(x[:220] for x in urls[:20]), sep="\n   ")
        for term in ["video-720", "video_items", "download", "video_url", "videoUrl", "cloudinary", "assets.mixkit.co"]:
            at = source.find(term)
            if at >= 0: print("context",term,repr(source[max(0,at-95):at+195]))
    except Exception as exc:
        print("page fetch failed",type(exc).__name__,str(exc)[:240])

    for term in ['og:video', '<video', 'contentUrl', 'embedUrl', 'data-controller="video', 'data-video', 'video-720', 'video-360', 'active_storage', 'download-link', 'data-action="click->download']:
        at = source.find(term)
        if at >= 0: print("special", term, repr(source[max(0,at-290):at+600]))
    video_mp4 = sorted(set(re.findall(r'https?://[^"\\s<>]+?\\.mp4', source)))
    for url in video_mp4[:4]:
        try:
            status, resolved, headers, _ = fetch(url, method="HEAD")
            print("page-video-head", status, headers.get("Content-Type"), headers.get("Content-Length"), resolved)
            if "720.mp4" in url and status == 200:
                import hashlib
                req = Request(url, headers=HEADERS)
                sha = hashlib.sha256(); total = 0
                with urlopen(req, timeout=50) as res:
                    while data := res.read(1_048_576):
                        sha.update(data); total += len(data)
                print("PAGE VIDEO SHA256", key, total, sha.hexdigest())
        except Exception as exc:
            print("page-video-failed",type(exc).__name__,str(exc)[:150])

    for field in ("contentUrl","embedUrl"):
        match = re.search(r'"'+field+r'":"([^"]+)"',source)
        if not match: print("structured",field,"MISSING"); continue
        url = match.group(1)
        print("structured",field,url)
        try:
            status,resolved,headers,_ = fetch(url,method="HEAD")
            print("structured-head",field,status,headers.get("Content-Type"),headers.get("Content-Length"),resolved)
            if field == "contentUrl" and status == 200:
                import hashlib
                h=hashlib.sha256(); n=0
                with urlopen(Request(url,headers=HEADERS),timeout=90) as resp:
                    while chunk:=resp.read(2_097_152):
                        h.update(chunk);n+=len(chunk)
                print("structured-sha256",key,n,h.hexdigest())
        except Exception as ex: print("structured-error",field,type(ex).__name__,str(ex)[:180])
    candidates = [
        f"https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4",
        f"https://assets.mixkit.co/videos/preview/mixkit-{page.rstrip('/').split('/')[-1]}-large.mp4",
    ]
    for url in candidates:
        try:
            status, resolved, headers, _ = fetch(url, method="HEAD")
            print("candidate",status,headers.get("Content-Type"),headers.get("Content-Length"),resolved)
        except Exception as exc:
            print("candidate failed", type(exc).__name__,str(exc)[:190],url)
