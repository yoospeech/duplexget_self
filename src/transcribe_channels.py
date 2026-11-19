#!/usr/bin/env python3
"""Optional ASR preview by channel. NOT the official aligned Moshi annotation."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--language", default=None)
    args = p.parse_args()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("pip install -r requirements-asr.txt") from exc
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    for path in sorted((Path(args.root) / "data_stereo").glob("*.wav")):
        dest = path.with_suffix(".asr.json")
        if dest.exists():
            continue
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        sides = []
        for channel, role in ((0, "assistant"), (1, "user")):
            segments, info = model.transcribe(
                np.ascontiguousarray(audio[:, channel]), language=args.language,
                beam_size=5, vad_filter=False)
            sides.append({"speaker_role": role, "language": info.language,
                          "segments": [{"start": x.start, "end": x.end, "text": x.text}
                                       for x in segments]})
        dest.write_text(json.dumps({
            "source": path.name, "sample_rate": sr, "channels": sides,
            "warning": "Segment-level ASR guesses; NOT word-aligned Moshi annotation"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(dest)

if __name__ == "__main__":
    main()
