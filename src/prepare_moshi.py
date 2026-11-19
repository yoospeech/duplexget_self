#!/usr/bin/env python3
"""Prepare Moshi input: 24 kHz 2-channel WAV (left assistant, right user).

Stereo does not imply speaker-separated audio. Verify channel roles manually.
Word-level aligned transcription must be produced by official Moshi annotate.py.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}

def probe(path):
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-show_entries", "format=duration",
         "-of", "json", str(path)], text=True))
    return int(info["streams"][0]["channels"]), float(info["format"]["duration"])

def ffmpeg_convert(src, dest, start, duration, swap, mono_policy):
    channels, _ = probe(src)
    if channels < 2 and mono_policy == "reject":
        raise ValueError("mono source cannot provide independent dialogue channels")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start), "-i", str(src), "-t", str(duration)]
    if channels >= 2:
        cmd += ["-af", "pan=stereo|c0=c1|c1=c0" if swap else "pan=stereo|c0=c0|c1=c1"]
    else:
        cmd += ["-ac", "2"]
    cmd += ["-ar", "24000", "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True)

def source_sample_rate(path):
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "json", str(path)], text=True))
    return int(info["streams"][0]["sample_rate"])


def activity_80ms(path, threshold_db=-42):
    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    frame = int(round(sr * .08))
    flags = []
    for start in range(0, len(signal), frame):
        chunk = signal[start:start + frame]
        rms = np.sqrt(np.mean(np.square(chunk), axis=0) + 1e-12)
        flags.append((20 * np.log10(rms) >= threshold_db).astype(int).tolist())
    return {
        "frame_sec": .08, "threshold_dbfs": threshold_db,
        "left_assistant_activity_bins": sum(f[0] for f in flags),
        "right_user_activity_bins": sum(f[1] for f in flags),
        "overlap_activity_bins": sum(f[0] and f[1] for f in flags),
        "total_bins": len(flags), "activity_flags_left_right": flags,
        "note": "Energy heuristic, not validated VAD or speaker diarization."
    }

def approved_sources(path):
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing approved stereo manifest: {manifest}")
    allowed = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if item.get("human_verified") is True:
            allowed[Path(item["path"]).resolve()] = item
    return allowed


def prepare(args):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe must be installed")
    source_root = Path(args.input).resolve()
    out = Path(args.out).resolve()
    wav_dir = out / "data_stereo"
    wav_dir.mkdir(parents=True, exist_ok=True)
    allowed = approved_sources(args.approved_manifest)
    sources = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    manifest, provenance, errors = [], [], []
    for index, source in enumerate(sources):
        try:
            source = source.resolve()
            review = allowed.get(source)
            if review is None:
                raise ValueError("Not in human-verified stereo allowlist")
            stat = source.stat()
            if stat.st_size != review.get("size_bytes") or stat.st_mtime_ns != review.get("mtime_ns"):
                raise ValueError("Source file changed since manual approval; inspect and approve again")
            if review.get("role_order") not in ("left-assistant", "right-assistant"):
                raise ValueError("Approved source has invalid channel roles")
            channels, total = probe(source)
            if channels != 2:
                raise ValueError("Approved file must remain exactly stereo")
            if channels < 2 and args.mono_policy == "reject":
                raise ValueError("mono source skipped; two speakers cannot be recovered by duplication")
            fingerprint = hashlib.sha1(str(source.relative_to(source_root)).encode()).hexdigest()[:12]
            original_sr = source_sample_rate(source)
            processed_source = source
            sidon_applied = False
            if original_sr < 24000 and args.lowrate_method == "sidon":
                restored = out / "sidon_restored" / f"{index:05d}_{fingerprint}.wav"
                restored.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite or not restored.exists():
                    subprocess.run([sys.executable, str(Path(__file__).with_name("restore_sidon.py")),
                                    "--input", str(source), "--output", str(restored),
                                    "--device", args.sidon_device], check=True)
                processed_source = restored
                sidon_applied = True
            elif original_sr < 24000 and args.lowrate_method == "reject":
                raise ValueError(f"Source {original_sr} Hz <24kHz; choose --lowrate-method sidon or resample")
            part = 0
            start = 0.0
            while start < total:
                duration = min(args.chunk_sec, total - start)
                if duration < args.min_sec:
                    break
                dest = wav_dir / f"{index:05d}_{fingerprint}_{part:04d}.wav"
                if args.overwrite or not dest.exists():
                    ffmpeg_convert(processed_source, dest, start, duration, review["role_order"] == "right-assistant", "reject")
                data, sr = sf.read(dest, dtype="float32", always_2d=True)
                if sr != 24000 or data.shape[1] != 2:
                    raise ValueError("failed 24 kHz stereo validation")
                rel = dest.relative_to(out).as_posix()
                actual = len(data) / sr
                manifest.append({"path": rel, "duration": actual})
                provenance.append({"path": rel, "source": str(source),
                                   "source_channels": channels, "original_sample_rate": original_sr,
                                   "sidon_applied": sidon_applied, "original_start_sec": start,
                                   "duration": actual, "mono_duplicated": channels < 2,
                                   "swapped": review["role_order"] == "right-assistant"})
                if not args.skip_activity:
                    dest.with_suffix(".activity.json").write_text(
                        json.dumps(activity_80ms(dest, args.energy_threshold_dbfs),
                                   ensure_ascii=False), encoding="utf-8")
                part += 1
                start += args.chunk_sec
        except Exception as exc:
            errors.append({"source": str(source), "error": str(exc)})
            print("SKIP", source, str(exc)[:160])
    for name, rows in (("train.jsonl", manifest), ("provenance.jsonl", provenance)):
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} clips; skipped {len(errors)} sources.")
    print("Next: run src/annotate.py for aligned transcription.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="processed/moshi")
    p.add_argument("--chunk-sec", type=float, default=100)
    p.add_argument("--lowrate-method", choices=["reject", "resample", "sidon"], default="reject")
    p.add_argument("--sidon-device", default="cuda:0")
    p.add_argument("--min-sec", type=float, default=3)
    p.add_argument("--approved-manifest", required=True, help="Explicitly reviewed stereo file allowlist")
    p.add_argument("--swap-channels", action="store_true", help="Deprecated; roles from approval manifest")
    p.add_argument("--mono-policy", choices=["reject", "duplicate"], default="reject")
    p.add_argument("--energy-threshold-dbfs", type=float, default=-42)
    p.add_argument("--skip-activity", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.chunk_sec <= 0 or args.min_sec <= 0:
        p.error("Duration parameters must be positive.")
    if args.swap_channels:
        p.error("--swap-channels deprecated; use inspect_audio.py approve --roles right-assistant")
    prepare(args)

if __name__ == "__main__":
    main()
