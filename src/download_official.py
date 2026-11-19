#!/usr/bin/env python3
"""Download official OpenSLR file links from the unified catalog.

No crawling GitHub leads or commercial pages. Only OpenSLR official archive links.
License approval is required separately; download files only after reviewing rights.
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ARCHIVES = (".tar.gz", ".tgz", ".tar", ".zip", ".flac", ".wav", ".mp3", ".opus", ".ogg")
HOSTS = {"openslr.org", "www.openslr.org", "us.openslr.org", "eu.openslr.org", "cn.openslr.org", "openslr.elda.org"}
HEADERS = {"User-Agent": "duplexget-self/1.0 (public corpus downloader)"}

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.urls.append(href)

def allowed(url):
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname in HOSTS and not parsed.username and not parsed.password

def get_page(url):
    if not allowed(url):
        raise ValueError("Only approved OpenSLR HTTPS hosts may be queried")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as response:
        final = response.geturl()
        if not allowed(final):
            raise ValueError("Page redirected outside allowed hosts")
        content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise ValueError("Page too large")
        return content.decode("utf-8", errors="replace")

def file_links(url):
    parser = Links()
    parser.feed(get_page(url))
    found = []
    for href in parser.urls:
        target = urllib.parse.urljoin(url, href)
        if not allowed(target):
            continue
        if urllib.parse.urlsplit(target).path.lower().endswith(ARCHIVES):
            found.append(target)
    return sorted(set(found))

def download(url, dest):
    if not allowed(url):
        raise ValueError("Not an OpenSLR HTTPS URL")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".part")
    start = temp.stat().st_size if temp.exists() else 0
    headers = dict(HEADERS)
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        if not allowed(response.geturl()):
            raise ValueError("Redirected outside OpenSLR")
        code = response.status
        if start and code != 206:
            start = 0
            mode = "wb"
        else:
            mode = "ab" if start else "wb"
        with temp.open(mode) as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
    temp.replace(dest)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="data/unified_catalog_500.json")
    p.add_argument("--out", default="raw/official")
    p.add_argument("--source", choices=["openslr", "all"], default="openslr")
    p.add_argument("--id", action="append", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--acknowledge-rights", action="store_true")
    p.add_argument("--max-files", type=int, default=100)
    args = p.parse_args()
    if not args.dry_run and not args.acknowledge_rights:
        p.error("Review original corpus license, then pass --acknowledge-rights")
    if args.max_files < 1:
        p.error("--max-files must be positive")
    data = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    rows = [r for r in data["datasets"] if r.get("source") == "openslr"
            and (not args.id or int(r["id"]) in args.id)]
    if args.source == "all":
        other = [r for r in data["datasets"] if r.get("source") in ("official", "github_lead")
                 and (not args.id or int(r["id"]) in args.id)]
        for row in other:
            print(f"[{int(row['id']):03d}] {row['name']} | manual_review: no verified direct download URL")
    count = 0
    for row in rows:
        rid = int(row["id"])
        page = row["url"]
        if not allowed(page):
            print(f"[{rid:03d}] skipped: unsupported official host")
            continue
        try:
            links = file_links(page)
        except Exception as exc:
            print(f"[{rid:03d}] error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"[{rid:03d}] {row['name']} | {len(links)} direct archive links", flush=True)
        for url in links:
            if count >= args.max_files:
                print(f"Stopped at --max-files={args.max_files}")
                return
            count += 1
            print(f"    {url}", flush=True)
            if args.dry_run:
                continue
            filename = Path(urllib.parse.urlsplit(url).path).name
            dest = Path(args.out) / f"{rid:03d}" / filename
            if dest.exists():
                print(f"    exists: {dest}")
                continue
            try:
                download(url, dest)
                print(f"    downloaded: {dest}")
            except Exception as exc:
                print(f"    error: {exc}", file=sys.stderr)
    print(f"Finished. {count} direct archive links {'listed' if args.dry_run else 'attempted'}.")
    print("Archives are not extracted automatically. Stereo inspection requires unpacked audio.")

if __name__ == "__main__":
    main()
