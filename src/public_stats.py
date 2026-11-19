#!/usr/bin/env python3
"""Summarize catalog-marked public datasets by language without inventing allocation.

Hours are announced catalog metadata, not measured downloadable audio.
Multi-language hours are unallocated; duplicated datasets may double-count.
Optional HF audit JSONL reduces counts to remotely accessible, non-gated
repos but does not grant training/commercial rights.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="data/catalog.json")
    p.add_argument("--audit", default="")
    p.add_argument("--out", default="reports/public_language_stats.csv")
    args = p.parse_args()
    data = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    rows = data["datasets"] if isinstance(data, dict) else data
    audit = {}
    if args.audit and Path(args.audit).exists():
        for line in Path(args.audit).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                audit[int(record["id"])] = record

    totals = defaultdict(lambda: dict(datasets=0, single_language_hours=0.0,
                                      multi_language_datasets=0, unquantified_datasets=0,
                                      accessible=0, unknown_access=0))
    public = [r for r in rows if r.get("access") == "공개"]
    multi_hours = []
    for d in public:
        langs = [x.strip().lower() for x in d.get("lang", "").split(",") if x.strip()]
        hours = d.get("hours")
        if len(langs) > 1 and isinstance(hours, (int, float)):
            multi_hours.append((d["name"], langs, hours))
        state = audit.get(int(d["id"]))
        checked = state.get("audit", state) if state else None
        accessible = bool(checked and checked.get("status") == "accessible" and
                          not checked.get("gated") and not checked.get("private") and
                          state.get("status") in ("accessible", "dry_run", "downloaded"))
        # A dry-run records a snapshot of actual repository accessibility, not a download.
        # Do not imply a public repository is available if it was skipped as gated/private.
        for lang in langs:
            item = totals[lang]
            item["datasets"] += 1
            if len(langs) > 1:
                item["multi_language_datasets"] += 1
            elif isinstance(hours, (int, float)):
                item["single_language_hours"] += hours
            else:
                item["unquantified_datasets"] += 1
            if accessible:
                item["accessible"] += 1
            elif state is None:
                item["unknown_access"] += 1
    result = [{"language": lang, **v} for lang, v in totals.items()]
    result.sort(key=lambda x: (-x["datasets"], x["language"]))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["language", "datasets",
                                "single_language_hours", "multi_language_datasets",
                                "unquantified_datasets", "accessible", "unknown_access"])
        writer.writeheader()
        writer.writerows(result)
    print(f"Catalog-marked public entries: {len(public)} / {len(rows)}")
    print(f"Languages represented: {len(result)}")
    print("Language | dataset entries | single-language published hours | multilingual entries | HF access confirmed")
    for d in result:
        print(f'{d["language"]:>8} | {d["datasets"]:>3} | {d["single_language_hours"]:>12,.2f}h | {d["multi_language_datasets"]:>3} | {d["accessible"]:>3}')
    print("Multilingual global hours (NOT allocated to individual languages):")
    for name, langs, hours in multi_hours:
        print(f'  {name}: {hours:,.2f}h total for {",".join(langs)}')
    print(f"Saved: {path}")
    if not audit:
        print("Access status: NOT CHECKED (no audit file found).")
    else:
        print("HF access confirmed from supplied audit/dry-run results; not a training-license check.")
    print("Actual disk storage (GB) cannot be inferred reliably from audio hours; inspect HF file sizes/dry-run.")
    print("Do NOT add all hours without deduplication or interpret this as confirmed audio download.")
if __name__ == "__main__":
    main()
