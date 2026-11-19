#!/usr/bin/env python3
"""Summarize already downloaded audio inspected by inspect_audio.py.

Audio length and sample rate are measured from directly readable files only.
TAR, ZIP, Parquet or Arrow files are counted, NOT unpacked or credited as audio.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ARCHIVES = (".tar", ".tar.gz", ".tgz", ".zip", ".parquet", ".arrow")
COLS = [
    "dataset_id", "dataset", "languages", "audio_files", "mono_files",
    "stereo_files", "multichannel_files", "error_files", "other_files",
    "total_audio_hours", "stereo_hours", "sample_rates",
    "archive_files", "archive_size_gib"
]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--inspection", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/audio_by_dataset.csv"))
    args = ap.parse_args()
    root = args.input.resolve()
    if not root.is_dir():
        ap.error(f"Input directory missing: {root}")
    if not args.inspection.is_file():
        ap.error(f"Inspection report missing: {args.inspection}")

    catalog = {}
    if args.catalog.is_file():
        data = json.loads(args.catalog.read_text(encoding="utf-8"))
        for record in data.get("datasets", []):
            catalog[int(record["id"])] = record

    def group(path):
        try:
            rel = path.resolve().relative_to(root)
            folder = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        except ValueError:
            folder = "(outside root)"
        match = re.match(r"^(\d{3})[_-]", folder)
        if not match:
            match = re.match(r"^test_(\d{3})$", folder)
        code = int(match.group(1)) if match else None
        record = catalog.get(code) if code is not None else None
        return code, (record or {}).get("name", folder), (record or {}).get("lang", "und")

    rows = defaultdict(lambda: {
        "audio_files": 0, "mono_files": 0, "stereo_files": 0,
        "multichannel_files": 0, "error_files": 0, "other_files": 0,
        "total_audio_hours": 0.0, "stereo_hours": 0.0,
        "sample_rates": defaultdict(int), "archive_files": 0,
        "archive_size_gib": 0.0,
    })

    for line in args.inspection.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = group(Path(item["path"]))
        v = rows[key]
        v["audio_files"] += 1
        status = item.get("status")
        ch = item.get("channels")
        length = max(0.0, float(item.get("duration_sec") or 0.0))
        if status == "error":
            v["error_files"] += 1
            continue
        sr = str(item.get("sample_rate") or "?")
        v["sample_rates"][sr] += 1
        v["total_audio_hours"] += length / 3600
        if ch == 1:
            v["mono_files"] += 1
        elif ch == 2:
            v["stereo_files"] += 1
            v["stereo_hours"] += length / 3600
        elif isinstance(ch, int) and ch > 2:
            v["multichannel_files"] += 1
        else:
            v["other_files"] += 1

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".cache" in path.parts:
            continue
        if not path.name.lower().endswith(ARCHIVES):
            continue
        v = rows[group(path)]
        v["archive_files"] += 1
        try:
            v["archive_size_gib"] += path.stat().st_size / (1024 ** 3)
        except OSError:
            pass

    result = []
    for key, val in sorted(rows.items(), key=lambda x: str(x[0])):
        did, name, lang = key
        result.append({
            "dataset_id": f"{did:03d}" if did is not None else "",
            "dataset": name, "languages": lang,
            **{k: val[k] for k in (
                "audio_files", "mono_files", "stereo_files", "multichannel_files",
                "error_files", "other_files", "archive_files")},
            "total_audio_hours": round(val["total_audio_hours"], 3),
            "stereo_hours": round(val["stereo_hours"], 3),
            "sample_rates": ";".join(
                f"{rate}:{n}" for rate, n in sorted(val["sample_rates"].items())),
            "archive_size_gib": round(val["archive_size_gib"], 3)
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(result)
    print("Dataset | audio | stereo | stereo hours | archives")
    for row in result:
        print(f"[{row['dataset_id'] or '---'}] {row['dataset']}: "
              f"{row['audio_files']} audio, {row['stereo_files']} stereo, "
              f"{row['stereo_hours']:.2f}h stereo, {row['archive_files']} archive files")
    print(f"REPORT: {args.out}")
    if not result:
        print("No directly readable audio or archive files were found.")
    print("Archive/Parquet contents are NOT included in audio hours; stereo is not proof of separate speakers.")

if __name__ == "__main__":
    main()
