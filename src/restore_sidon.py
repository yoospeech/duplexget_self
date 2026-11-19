#!/usr/bin/env python3
"""Optional Sidon v0.1 speech restoration for low-rate, speaker-isolated stereo.

Sidon is a single-speaker restoration model: each stereo channel is processed
independently to 48 kHz and saved directly at 48 kHz.
This MAY change turn timing, speaker timbre, phase and overlap behavior.
Do not use this as proof of real high-frequency information.
License: Sidon v0.1 checkpoint MIT; verify all dependencies and source data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def restore(channel, orig_sr, fe, decoder, preprocessor, device):
    import torch
    import torchaudio.functional as AF
    # Original public Sidon demo recipe; each channel handled separately.
    samples = torch.from_numpy(channel.copy()).float().view(1, -1)
    samples = samples / samples.abs().max().clamp_min(1e-6) * 0.9
    samples = AF.highpass_biquad(samples, orig_sr, 50)
    wav16 = AF.resample(samples, orig_sr, 16000)
    target48 = round(channel.shape[0] * 48000 / orig_sr)
    # Demo uses extra 1.5 s tail to compensate chunking effects.
    wav16 = torch.nn.functional.pad(wav16, (0, 24000))
    pieces, cached = [], None
    with torch.inference_mode():
        for chunk in wav16.flatten().split(16000 * 96):
            features = preprocessor(torch.nn.functional.pad(chunk, (160, 160)),
                                    return_tensors="pt")
            y = fe(features["input_features"].to(device))["last_hidden_state"]
            if cached is not None:
                y = torch.cat((cached, y), dim=1)
            piece = decoder(y.transpose(1, 2)).reshape(-1)[:-960]
            pieces.append(piece.detach().float().cpu())
            cached = y[:, -1:].detach()
    out = torch.cat(pieces).view(1, -1)
    if out.shape[-1] < target48:
        out = torch.nn.functional.pad(out, (0, target48 - out.shape[-1]))
    out = out[:, :target48]
    return out.view(-1).numpy()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source stereo WAV/FLAC")
    parser.add_argument("--output", required=True, help="Restored stereo 48 kHz WAV")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-source-rate", type=int, default=22050)
    parser.add_argument("--allow-reprocess", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from huggingface_hub import hf_hub_download
    from transformers import SeamlessM4TFeatureExtractor

    inp = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    if inp == dst:
        parser.error("input and output paths must differ; never overwrite the source")
    audio, sr = sf.read(inp, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        parser.error("Only genuine two-channel recordings are eligible")
    if sr >= 24000 and not args.allow_reprocess:
        parser.error("Source is already >=24kHz; Sidon bypass recommended")
    if sr > args.max_source_rate and not args.allow_reprocess:
        parser.error(f"Source sr {sr} exceeds configured restoration ceiling")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Public Sidon v0.1 demo checkpoints were built for CUDA; use a CUDA device")
    model_id = "sarulab-speech/sidon-v0.1"
    fe_file = hf_hub_download(repo_id=model_id, filename="feature_extractor_cuda.pt")
    decoder_file = hf_hub_download(repo_id=model_id, filename="decoder_cuda.pt")
    fe = torch.jit.load(fe_file, map_location=args.device).eval().to(args.device)
    decoder = torch.jit.load(decoder_file, map_location=args.device).eval().to(args.device)
    preprocessor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    channels = [restore(audio[:, i], sr, fe, decoder, preprocessor, args.device) for i in range(2)]
    stereo = np.stack(channels, axis=-1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, stereo, 48000, subtype="PCM_16")
    meta = {
        "source": str(inp), "output": str(dst), "original_sample_rate": sr,
        "output_sample_rate": 48000, "original_channels": 2,
        "model": model_id, "mode": "independent_stereo_channel_restoration",
        "risk": "Generated bandwidth and altered timing; validate original vs restored before training"
    }
    dst.with_suffix(".sidon.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
