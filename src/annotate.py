#!/usr/bin/env python3
"""Whisper-timestamped aligned transcription for Kyutai Moshi stereo clips.

Adapted from the user-provided annotate.py. By default only channel 0
(assistant, SPEAKER_MAIN) is transcribed to clip.json as in upstream.
With --channels both, channel 1 is additionally saved as clip.user.json.
A separate user file is NOT consumed by upstream Moshi automatically.
"""
import argparse
import gc
import gzip
import importlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import sphn
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper

SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)

@contextmanager
def write_and_rename(path, mode="w"):
    temp = Path(str(path) + f".tmp.{os.getpid()}")
    try:
        with temp.open(mode, encoding="utf-8") as f:
            yield f
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def load_audio_paths(egs_path):
    opening = gzip.open if str(egs_path).endswith(".gz") else open
    with opening(egs_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                path = Path(record["path"])
                # The prepare_moshi manifest stores relative paths from its own parent.
                if not path.is_absolute():
                    path = egs_path.parent / path
                yield path

def annotate_one(audio_path, output_path, model, language, channel,
                 keep_silence=0.2, overwrite=False):
    if output_path.exists() and not overwrite:
        return "skipped"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    audio, sr = sphn.read(audio_path)
    if audio.ndim == 1:
        audio = audio[None, :]
    if channel >= audio.shape[0]:
        raise ValueError(f"Audio has {audio.shape[0]} channel(s); requested {channel}")
    vocals = torch.from_numpy(np.asarray(audio[channel]).copy()).float()[None]
    vocals = F.resample(vocals, sr, SAMPLE_RATE).squeeze(0).numpy()
    duration = len(vocals) / SAMPLE_RATE
    if duration > 4 * 3600:
        raise ValueError("Input audio exceeds 4 hours")
    # Scoped patch: restore after each utterance to avoid wrappers accumulating.
    transcribe_module = importlib.import_module("whisper_timestamped.transcribe")
    original = transcribe_module.get_vad_segments
    def padded_vad(*args, **kwargs):
        segments = original(*args, **kwargs)
        result = []
        last_end = 0
        pad = int(SAMPLE_RATE * keep_silence)
        max_end = len(vocals)
        for seg in segments:
            start = max(last_end, seg["start"] - pad)
            end = min(max_end, seg["end"] + pad)
            if end > start:
                result.append({"start": start, "end": end})
                last_end = end
        return result
    try:
        if keep_silence > 0:
            transcribe_module.get_vad_segments = padded_vad
        output = whisper.transcribe(
            model, vocals, language=None if language == "auto" else language,
            vad="auditok" if duration > 10 else None, best_of=5, beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), verbose=None)
    finally:
        transcribe_module.get_vad_segments = original
    alignments = []
    for segment in output.get("segments", []):
        for word in segment.get("words", []):
            if {"text", "start", "end"} <= word.keys():
                alignments.append([word["text"], [word["start"], word["end"]], "SPEAKER_MAIN"])
    with write_and_rename(output_path) as f:
        json.dump({"alignments": alignments}, f, ensure_ascii=False)
    return "done"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("egs", type=Path, help="Moshi train.jsonl or JSONL.gz")
    p.add_argument("--lang", default="en", help="ISO language code, or auto")
    p.add_argument("--whisper_model", default="medium")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--channels", choices=["assistant", "both"], default="assistant",
                   help="Upstream-compatible assistant json; optional sidecar user json")
    p.add_argument("--keep_silence_in_segments", type=float, default=0.2)
    p.add_argument("--rerun_errors", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shards", "-S", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--local", "-l", action="store_true", help="Kept for compatibility; local is default")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.shards <= 0 or args.shard < 0 or args.shard >= args.shards:
        p.error("shard must satisfy 0 <= shard < shards")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    model = whisper.load_model(args.whisper_model, device=device)
    paths = list(load_audio_paths(args.egs))
    selected = paths[args.shard::args.shards]
    errors = 0
    for i, path in enumerate(selected, 1):
        if not path.is_file():
            logger.error("Missing audio: %s", path)
            errors += 1
            continue
        for channel in ([0, 1] if args.channels == "both" else [0]):
            dest = path.with_suffix(".json" if channel == 0 else ".user.json")
            err = Path(str(dest) + ".err")
            if err.exists() and not args.rerun_errors:
                logger.warning("Skipping previously failed %s", path)
                continue
            try:
                status = annotate_one(path, dest, model, args.lang, channel,
                                      keep_silence=args.keep_silence_in_segments,
                                      overwrite=args.overwrite)
                if status == "done":
                    err.unlink(missing_ok=True)
                logger.info("[%d/%d] channel=%s %s %s", i, len(selected), channel, status, path.name)
            except Exception:
                errors += 1
                logger.exception("Failed: %s channel %d", path, channel)
                err.touch()
                if device == "cuda" and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
    if errors:
        raise SystemExit(f"Annotation failures: {errors}")

if __name__ == "__main__":
    main()
