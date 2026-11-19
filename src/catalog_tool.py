#!/usr/bin/env python3
"""Catalog audit and opt-in Hugging Face dataset downloader."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

CATALOG = Path(__file__).resolve().parents[1] / "data" / "catalog.json"
PATTERNS = ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus", "*.m4a", "*.aac", "*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.parquet", "*.arrow", "*.json", "*.jsonl", "*.csv", "*.txt", "*.tsv", "*.md"]

def records(catalog=CATALOG):
    obj = json.loads(Path(catalog).read_text(encoding="utf-8"))
    return obj["datasets"] if isinstance(obj, dict) else obj

def repo_id(url):
    u = urlparse(url)
    if u.hostname not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        return f"{parts[1]}/{parts[2]}"
    return None

def select(rows, ids, languages, public_only):
    for row in rows:
        if ids and int(row["id"]) not in ids:
            continue
        codes = {x.strip().lower() for x in (row.get("lang") or "").split(",")}
        if languages and not languages.intersection(codes):
            continue
        if public_only and row.get("access") != "공개":
            continue
        yield row

def inspect(row):
    from huggingface_hub import HfApi
    rid = repo_id(row["url"])
    if not rid:
        return {"id": row["id"], "repo": None, "status": "not_hf"}
    try:
        info = HfApi().dataset_info(rid, files_metadata=True, token=os.getenv("HF_TOKEN"))
        card = info.card_data
        license_id = getattr(card, "license", None) if card else None
        if isinstance(card, dict):
            license_id = card.get("license")
        files = [x.rfilename for x in info.siblings or []]
        media = [x for x in files if x.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".tar", ".zip", ".parquet", ".arrow"))]
        return {"id": row["id"], "repo": rid, "status": "accessible",
                "gated": bool(info.gated), "private": bool(info.private),
                "license": license_id, "file_count": len(files),
                "media_container_files": len(media), "sample_files": media[:8]}
    except Exception as exc:
        return {"id": row["id"], "repo": rid, "status": "unavailable", "error": str(exc)[:300]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(CATALOG))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "audit", "download"):
        p = sub.add_parser(command)
        p.add_argument("--id", action="append", type=int)
        p.add_argument("--lang", action="append")
        p.add_argument("--public-only", action="store_true")
        if command != "list":
            p.add_argument("--all", action="store_true")
            p.add_argument("--out", default="reports/hf_audit.jsonl" if command == "audit" else "raw")
        if command == "download":
            p.add_argument("--acknowledge-rights", action="store_true")
            p.add_argument("--include-restricted", action="store_true")
            p.add_argument("--allow-unknown-license", action="store_true")
            p.add_argument("--all-files", action="store_true")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = list(select(records(args.catalog), set(args.id or []),
                       set(x.lower() for x in args.lang or []), args.public_only))
    if args.command == "list":
        for row in rows:
            print(f"{int(row['id']):03d} | {row.get('lang', '?'):20} | {row.get('access', '?'):5} | {row['name']} | {row['url']}")
        return
    if not args.id and not args.all:
        parser.error("Specify --id N (repeatable) or --all")
    if args.command == "audit":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as log:
            for row in rows:
                result = inspect(row)
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(row["id"], result["status"], result.get("license", ""))
        return
    if not args.acknowledge_rights:
        parser.error("Review original-source rights and pass --acknowledge-rights")
    from huggingface_hub import snapshot_download
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "download_results.jsonl").open("a", encoding="utf-8") as log:
        for row in rows:
            rid = repo_id(row["url"])
            result = {"id": row["id"], "repo": rid, "status": "skipped"}
            if not rid:
                result["reason"] = "not_hf"
            elif row.get("access") != "공개" and not args.include_restricted:
                result["reason"] = "restricted_or_unverified"
            else:
                audit = inspect(row)
                result["audit"] = audit
                if audit["status"] != "accessible":
                    result["reason"] = "unavailable"
                elif (audit.get("gated") or audit.get("private")) and not args.include_restricted:
                    result["reason"] = "gated_or_private"
                elif not audit.get("license") and not args.allow_unknown_license:
                    result["reason"] = "unknown_license"
                else:
                    destination = out / f"{int(row['id']):03d}_{rid.replace('/', '__')}"
                    try:
                        files = snapshot_download(repo_id=rid, repo_type="dataset",
                                                  local_dir=destination,
                                                  allow_patterns=None if args.all_files else PATTERNS,
                                                  max_workers=args.workers, dry_run=args.dry_run,
                                                  token=os.getenv("HF_TOKEN"))
                        result["status"] = "dry_run" if args.dry_run else "downloaded"
                        result["destination"] = str(destination)
                        if args.dry_run:
                            result["files"] = len(files)
                            result["estimated_bytes"] = sum((getattr(f, "file_size", 0) or 0) for f in files)
                    except Exception as exc:
                        result.update(status="error", reason=str(exc)[:500])
            print(f"[{int(row['id']):03d}] {result['status']}: {result.get('reason', result.get('destination', ''))}")
            log.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            log.flush()

if __name__ == "__main__":
    main()
