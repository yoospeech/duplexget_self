#!/usr/bin/env python3
"""Inspect audio before Moshi preprocessing. No automatic speaker-separation claims.

scan: ffprobe every directly accessible audio file, sample-decode stereo audio
      and flag possibly duplicated channels. Outputs JSONL + CSV.
approve: manually register a reviewed, speaker-separated two-channel file,
         with its channel roles and fingerprint in a separate JSONL allowlist.

WARNING: channel correlation and RMS activity do NOT prove speaker identity.
Requires ffmpeg, ffprobe, numpy and soundfile-independent sample decoding.
"""
import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
COLUMNS = ["path", "status", "codec", "channels", "sample_rate", "duration_sec",
           "size_bytes", "sampled_sec", "left_rms_dbfs", "right_rms_dbfs",
           "correlation", "active_overlap_ratio", "notes"]

def locate(root):
    p = Path(root).expanduser().resolve()
    if p.is_file():
        return [p] if p.suffix.lower() in SUFFIXES else []
    if not p.is_dir():
        raise FileNotFoundError(root)
    return sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in SUFFIXES)

def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate",
        "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    stream = info["streams"][0]
    return {"codec": stream.get("codec_name", ""), "channels": int(stream["channels"]),
            "sample_rate": int(stream.get("sample_rate") or 0),
            "duration_sec": float(info["format"].get("duration") or 0)}

def sample(path, start_sec, duration=15, rate=16000):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start_sec),
           "-i", str(path), "-t", str(duration), "-map", "0:a:0",
           "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    vals = np.frombuffer(result.stdout, dtype="<f4")
    if vals.size < rate * 2:
        return None
    return vals[:vals.size // 2 * 2].reshape(-1, 2)

def metrics(path, duration, sample_count, sample_sec):
    start_times = np.linspace(0, max(0, duration - sample_sec), sample_count)
    outputs = []
    for start in sorted(set(round(float(x), 2) for x in start_times)):
        block = sample(path, start, min(sample_sec, max(.1, duration - start)))
        if block is None:
            continue
        rms = np.sqrt(np.mean(block * block, axis=0) + 1e-12)
        db = 20 * np.log10(rms)
        corr = float(np.corrcoef(block[:, 0], block[:, 1])[0, 1]) if all(rms > 1e-4) else None
        if corr is not None and not math.isfinite(corr):
            corr = None
        frame = 1600   # 100ms energy-based activity, NOT VAD
        full = block[:len(block) // frame * frame]
        if len(full) == 0:
            continue
        activity = (20 * np.log10(
            np.sqrt(np.mean(full.reshape(-1, frame, 2) ** 2, axis=1) + 1e-12)
        ) > -42)
        overlap = float(np.mean(activity[:, 0] & activity[:, 1]))
        outputs.append((db, corr, overlap, len(block) / 16000))
    if not outputs:
        return {}
    corrs = [c for _, c, _, _ in outputs if c is not None]
    rms_left = np.median([z[0][0] for z in outputs])
    rms_right = np.median([z[0][1] for z in outputs])
    return {"sampled_sec": round(sum(z[3] for z in outputs), 2),
            "left_rms_dbfs": round(float(rms_left), 2),
            "right_rms_dbfs": round(float(rms_right), 2),
            "correlation": round(float(np.median(corrs)), 4) if corrs else None,
            "active_overlap_ratio": round(float(np.mean([z[2] for z in outputs])), 4)}

def inspect(path, sample_count=3, sample_sec=12):
    out = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        out.update(probe(path))
        count = out["channels"]
        if count == 1:
            out.update(status="mono", notes="Retain for diarization/source separation; not suitable as separated stereo.")
        elif count > 2:
            out.update(status="multichannel_review", notes="Must identify channels/speakers before remapping.")
        elif count == 2:
            out.update(metrics(path, out["duration_sec"], sample_count, sample_sec))
            corr = out.get("correlation")
            if corr is not None and corr > .985:
                out.update(status="stereo_near_duplicate",
                           notes="High positive sample correlation; channels may contain duplicated/mixed audio. Human review needed.")
            else:
                out.update(status="stereo_review",
                           notes="Two audio channels exist. Speaker isolation and left/right role still UNVERIFIED.")
        else:
            out.update(status="invalid_channels", notes="Unsupported channel count")
    except Exception as exc:
        out.update(status="error", notes=str(exc)[:300])
    return out

def atomic_jsonl_append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("path") != record["path"]]
    rows.append(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="reports/audio_inspection.jsonl")
    s.add_argument("--sample-count", type=int, default=3)
    s.add_argument("--sample-sec", type=float, default=12)
    a = sub.add_parser("approve")
    a.add_argument("--file", action="append", required=True)
    a.add_argument("--out", default="reports/approved_stereo.jsonl")
    a.add_argument("--roles", choices=["left-assistant", "right-assistant"], required=True)
    a.add_argument("--reviewed-speakers", action="store_true", required=True,
                   help="A human has verified each channel contains one intended speaker.")
    a.add_argument("--allow-similar-channels", action="store_true")
    args = p.parse_args()
    if args.command == "scan":
        if args.sample_count < 1 or args.sample_sec <= 0:
            p.error("Positive sample count and duration required")
        files = locate(args.input)
        results = [inspect(x, args.sample_count, args.sample_sec) for x in files]
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Inspected {len(results)} audio files. Report: {output}")
        print("Statuses:", dict(Counter(r["status"] for r in results)))
        print("No stereo file is automatically approved. Review channel roles before approve.")
    else:
        for filename in args.file:
            path = Path(filename).expanduser().resolve()
            if not path.is_file():
                p.error(f"Missing file: {path}")
            result = inspect(path)
            if result.get("channels") != 2:
                p.error(f"Only stereo can be approved: {path}")
            if result["status"] == "error":
                p.error(f"Cannot inspect: {path}: {result['notes']}")
            if result["status"] == "stereo_near_duplicate" and not args.allow_similar_channels:
                p.error(f"Potential duplicated channels: {path}. Review, then optionally --allow-similar-channels")
            stat = path.stat()
            record = {"path": str(path), "size_bytes": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "role_order": args.roles,
                      "human_verified": True}
            atomic_jsonl_append(Path(args.out), record)
            print("Approved:", path, args.roles)

if __name__ == "__main__":
    main()
