#!/usr/bin/env python3
"""Merge HF, GitHub leads, and official corpus links without inflating verified counts.

Only one canonical row per source URL; a mirror of a named corpus is flagged for
manual deduplication. This is a candidate registry, not a download authorization.
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}

def canonical(url):
    parsed = urlsplit(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/").lower()
    return host + path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", default="data/researched_datasets_500.json")
    p.add_argument("--github", default="data/github_research_leads.json")
    p.add_argument("--official", default="data/official_corpus_sources.json")
    p.add_argument("--out", default="data/unified_catalog_500")
    p.add_argument("--target", type=int, default=500)
    args = p.parse_args()
    items, seen = [], set()
    def add(row):
        key = canonical(row["url"])
        if not key or key in seen:
            return
        seen.add(key)
        row["id"] = len(items) + 1
        items.append(row)
    for d in read(args.hf).get("datasets", []):
        add({"name": d["name"], "url": d["url"], "source": "huggingface",
             "lang": d.get("lang", "und"), "license": d.get("license"),
             "access": d.get("access", "unknown"),
             "audio_evidence": d.get("audio_evidence", "unverified"),
             "dialogue_verified": False, "download_verified": False,
             "channel_status": "not_inspected",
             "inspect_status": "pending_download",
             "duplicate_review": "pending", "download_method": "hf",
             "notes": d.get("notes", ""), "hf_quality_tier": d.get("quality_tier")})
    for d in read(args.official).get("datasets", []):
        add({"name": d["name"], "url": d["url"], "source": d["source"],
             "lang": d.get("lang", "und"), "license": d.get("license"),
             "access": d.get("access", "unknown"),
             "audio_evidence": d.get("audio_evidence", "official_description"),
             "dialogue_verified": d.get("dialogue_verified", False),
             "download_verified": False,
             "channel_status": "not_inspected", "inspect_status": "pending_download",
             "duplicate_review": "pending", "download_method": d.get("download_method", "manual"),
             "notes": d.get("notes", ""), "hf_quality_tier": None})
    # GitHub search returns many models, scripts, forks, sales listings, and indexes.
    # Keep these as research leads, NEVER promote to downloadable corpora.
    for d in read(args.github).get("leads", []):
        add({"name": d["name"], "url": d["github_url"], "source": "github_lead",
             "lang": "und", "license": None, "access": "unknown",
             "audio_evidence": "unverified", "dialogue_verified": False,
             "download_verified": False, "channel_status": "not_inspected",
             "inspect_status": "not_applicable_until_audio_found",
             "duplicate_review": "pending", "download_method": "manual_review",
             "notes": d.get("note", ""), "hf_quality_tier": None})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target, "total_candidates": len(items),
        "source_counts": dict(Counter(x["source"] for x in items)),
        "verified_dialogue_count": sum(bool(x["dialogue_verified"]) for x in items),
        "verified_download_count": sum(bool(x["download_verified"]) for x in items),
        "warning": "500 is a research target, not 500 unique confirmed dialogue recordings. "
                   "Mirror/content deduplication, rights and audio inspection pending."
    }, "datasets": items}
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["id", "name", "url", "source", "lang", "license", "access", "audio_evidence",
               "dialogue_verified", "download_verified", "channel_status", "inspect_status",
               "duplicate_review", "download_method", "notes", "hf_quality_tier"]
    with out.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    print(f"Unified candidates: {len(items)}; target: {args.target}")
    print("By source:", dict(Counter(x["source"] for x in items)))
    print("Dialogue/download/channel status must be confirmed after actual file inspection.")
    print(out.with_suffix(".csv"))

if __name__ == "__main__":
    main()
