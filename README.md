# Multilingual Duplex Dataset Acquisition → Moshi Preprocessing

Dataset catalog (100 entries), conservative Hugging Face downloader, and 48 kHz stereo WAV / JSONL preparation with optional Sidon restoration. **Official Moshi/Mimi still requires a separate 24 kHz input adapter**; this project does not downsample the archived 48 kHz outputs.

> **Scope:** This repository does **not** contain acquired audio, provide a verified right to train commercially, validate all 100 source links, or perform full Moshi tokenization/training. Some catalog entries are metadata-only, paid, gated, duplicates, or unverified. `audit` checks HF availability before any download.

## Environment

Python 3.10+, `ffmpeg`, `ffprobe`. On Ubuntu: `sudo apt-get install ffmpeg`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional per-channel ASR (not word-aligned Moshi training JSON)
pip install -r requirements-asr.txt
```

## Step 1 — Browse and verify the catalog

Open `data/catalog.html` locally. Data in `data/catalog.json` drives the CLI. The IDs are not a guarantee that a repository is valid or directly downloadable.

```bash
python src/catalog_tool.py list --lang id
python src/catalog_tool.py list --public-only
python src/catalog_tool.py audit --id 4 --id 5
# Audit all (100 API requests; may be slow / rate limited)
python src/catalog_tool.py audit --all
```

An `audit` writes a JSONL with `accessible` / `unavailable`, HF gated status, license metadata when available, and sample media-container filenames. **Metadata correctness, source ownership, and commercial rights still require review.** Datasets may contain audio inside `.parquet`, `.zip`, `.tar` files rather than loose WAVs.

## Step 2 — Dry run then download

```bash
# Explicit license acknowledgement is required; unknown license is blocked by default.
python src/catalog_tool.py download --id 4 --acknowledge-rights --dry-run
python src/catalog_tool.py download --id 4 --acknowledge-rights
python src/catalog_tool.py download --id 5 --acknowledge-rights

# Explicitly override for separately authorized datasets only:
python src/catalog_tool.py download --id 7 --acknowledge-rights --include-restricted --allow-unknown-license
```

`--all` is supported, **but not advised** before verifying licenses and total size. Only Hugging Face dataset repository URLs are automatically downloaded. Others are logged and skipped. Publicly viewable repositories with no declared license are skipped unless `--allow-unknown-license`; that flag is *not* permission from the owner. Restricted data still requires valid access and license acceptance on Hugging Face. `HF_TOKEN` can be set for gated datasets after independent approval. `--all-files` downloads extra repository files.

Downloads go to `raw/` and logs to `raw/download_results.jsonl`; subsequent `snapshot_download` calls are resumable through the HF cache/metadata. Audio nested in dataset containers (tar/parquet/zip) needs dataset-specific extraction before preprocessing; **never blindly extract untrusted archives**.

## Step 3 — Moshi audio preparation

Kyutai's official `moshi-finetune` input schema requires:

* stereo audio, **left=assistant / generated audio**, **right=user / input audio**;
* `train.jsonl`, one `{ "path": "...wav", "duration": seconds }` per clip;
* an aligned per-clip `.json` transcription, generated using the official annotator.

Place **verified separated two-channel dialogues** under `incoming/` (or any folder):

```bash
python src/inspect_audio.py scan --input incoming --out reports/audio_inspection.jsonl
# Human review is required: listen to both channels and verify the speaker roles.
python src/inspect_audio.py approve --file /ABS/PATH/TO/incoming/dialogue.wav \
  --roles left-assistant --reviewed-speakers
python src/prepare_moshi.py --input incoming --out processed/moshi --chunk-sec 100 \
  --approved-manifest reports/approved_stereo.jsonl
# If right channel is assistant, approve using --roles right-assistant instead.
```

This creates `processed/moshi/data_stereo/*.wav` (48 kHz 16-bit stereo), `processed/moshi/train.jsonl`, `provenance.jsonl`, optional energy-based `.activity.json` (80 ms bins), and `errors.json`.

**Attention:** stereo does not automatically mean isolated speakers. If stereo is spatial room recording or podcast mix, first verify channel provenance and perform appropriate speaker separation. Mono sources are skipped by default; `--mono-policy duplicate` only tests the pipeline and does **not** reconstruct two speakers. No arbitrary output silence/padding is added. The 100-second slicing may cut turns at clip boundaries; production quality checks should account for this.

## Step 4 — End-to-end download + Moshi annotation

Use `download_proc.sh` to run **catalog download → 48 kHz stereo conversion → word-level alignment**.

```bash
pip install -r requirements.txt
# ffmpeg + ffprobe required: sudo apt-get install ffmpeg
# For CUDA, install a mutually compatible torch/torchaudio build for your CUDA first.

# Check the candidate first
python src/catalog_tool.py audit --id 4

# Full pipeline for a public, independently authorized HF dataset
bash download_proc.sh --id 4 --lang id --acknowledge-rights --inspect-only

# Multiple datasets in one run
bash download_proc.sh --id 4 --id 5 --lang id --acknowledge-rights --inspect-only

# Audio already available locally (left=assistant, right=user)
bash download_proc.sh --inspect-only --input /path/to/stereo_audio
# Listen and approve selected stereo files before proceeding:
python src/inspect_audio.py approve --file /absolute/path/to/separated.wav --roles left-assistant --reviewed-speakers
bash download_proc.sh --prep-only --input /path/to/stereo_audio --lang ko

# Reannotate existing processed/moshi/train.jsonl
bash download_proc.sh --annotate-only --lang ko --whisper-model medium

# Transcribe both channels: clip.json for assistant; clip.user.json sidecar for user
bash download_proc.sh --prep-only --input /path/to/stereo_audio --lang ko --channels both
```

The `--lang` option filters catalog language during download (if supplied), and sets Whisper's transcription language. Use `--lang auto` for mixed-language local input (not a catalog filter). For data acquisition only, use `--download-only`. For a rights-gated preview without downloading files, use `--dry-run`.

### About `src/annotate.py`

This replaces the previous `transcribe_channels.py` preview and follows the user-provided `whisper_timestamped` approach. `src/annotate.py` accepts `train.jsonl` or `.jsonl.gz`, resamples a selected stereo channel to **16 kHz only for Whisper**, and writes each `<clip>.json` as:

```json
{"alignments": [["hello", [0.1, 0.48], "SPEAKER_MAIN"]]}
```

Default: channel 0 (assistant) only, preserving the original formatter. `--channels both` also creates `<clip>.user.json` with channel-1 alignments; the user-sidecar JSON is **not automatically consumed by upstream Moshi** and its speaker label is still `SPEAKER_MAIN`, scoped to its own channel. Verify the exact Moshi fine-tune commit's expected transcript schema before training.

### Mandatory audio inspection gate

No input audio is automatically accepted as speaker-separated. The `src/inspect_audio.py scan` stage records sample rate, channels, codec, duration, size, channel correlation and approximate simultaneous energy activity in `reports/audio_inspection.jsonl` and `.csv`. Only WAV/FLAC/MP3/etc directly present as files are inspected, not audio nested within TAR/ZIP/Parquet.

`mono` audio is preserved for future diarization/separation. `multichannel_review` and `stereo_near_duplicate` are flagged for review. Even `stereo_review` requires listening to both channels to verify speaker isolation. No automatic heuristic certifies individual speakers or true barge-in.

Once a human has verified a particular file contains speaker-separated audio, run:
```bash
python src/inspect_audio.py approve --file /absolute/path/to/recording.wav \
  --roles left-assistant --reviewed-speakers
```
The approval file `reports/approved_stereo.jsonl` stores original absolute path, size, modification timestamp and roles. Changed files must be re-approved. `prepare_moshi.py` refuses all unapproved sources, and automatically swaps channels for entries approved with `--roles right-assistant`.

If source channels are mixed, use a separate diarization/source separation process and inspect the separated result. This repository does not perform automated speaker separation.

Stereo source channels must be independent speakers; if not, perform source-specific speaker separation first. Mono inputs are rejected by default. Source data inside Parquet, TAR or ZIP is **not automatically extracted**. In that case the downloader completes but preprocessing cannot start until audio has been extracted safely and its stereo channel roles confirmed. No credentials or raw audio are pushed to GitHub.

This is a local-first orchestration script; it does not submit SLURM jobs or pretend commercial permission has been granted. `--all` can be extremely large and many catalog entries are restricted or unverified. Rights flags only acknowledge your own review; they do not provide a license. Some data catalog links and language/hours metadata remain unverified.

## Current verification status

Repository files are committed and readable through the GitHub connector. Actual large-corpus download, diarization quality and end-to-end integration have not been validated in this environment. Test locally with `python -m unittest discover -s tests` and a short manually reviewed stereo clip before large downloads.

## Optional Sidon restoration for low-rate stereo

The independent-channel `src/restore_sidon.py` adapter implements the public
[sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon) v0.1
[Hugging Face demo](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta/blob/main/app.py)
recipe. It uses the single-speaker **Sidon** checkpoint (not DialogueSidon).
Each of the two original channels is restored independently using a
16 kHz feature model and 48 kHz decoder. The generated signals are then
saved at 48 kHz as stereo, without additional downsampling.

Low-rate behavior is **opt-in** and only applied to files already in the
human-reviewed stereo allowlist. You must keep original data to audit synthetic
bandwidth, changed timbre, channel bleed and potential time shifts. This is
not literal reconstruction of the original missing 8–12 kHz information.

```bash
# CUDA PyTorch + matching torchaudio must be installed
pip install -r requirements-sidon.txt

# Inspect first
bash download_proc.sh --inspect-only --input incoming
# Manually review each file, and approve channel roles:
python src/inspect_audio.py approve --file /ABS/PATH/incoming/recording.wav \
  --roles left-assistant --reviewed-speakers

# Approved <24 kHz stereo: optional Sidon -> 48 kHz -> annotate.py
bash download_proc.sh --prep-only --input incoming --lang ko \
  --lowrate-method sidon --sidon-device cuda:0

# Alternative for low-rate data: basic resample only
bash download_proc.sh --prep-only --input incoming --lang ko \
  --lowrate-method resample

# By default: --lowrate-method reject (sources below 24 kHz are skipped).
```

For a standalone test:
```bash
python src/restore_sidon.py --input lowrate_stereo.wav \
  --output restored_stereo_48k.wav --device cuda:0
```

Sidon checkpoint **v0.1** is marked MIT. That does not grant rights to the
input recordings; check the licenses of dependencies, models and training
datasets. DialogueSidon/DialogueSidon2 checkpoints are **CC BY-NC 4.0**
and are not used here. The independent channel restoration method may alter
speaker overlap and precise alignment. Compare against original audio
before using restored samples for full-duplex timing supervision.

**Validation:** Code uploaded but Sidon inference has not yet been exercised
against a real checkpoint and GPU as part of this repository.

## GitHub publishing

Create or identify an existing GitHub repository and share its `https://github.com/OWNER/REPO` URL in ChatGPT. Upload **only code, catalog, docs and tests**; keep `raw/`, `processed/`, personal recordings, credentials, and accepted license documents private. A connected GitHub account with write access is required.

## References

- [Kyutai Moshi Fine-tune README](https://github.com/kyutai-labs/moshi-finetune#prepare-dataset)
- [HF snapshot_download docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
## Stereo sample-rate conversion policy

| Original sample rate | Nyquist frequency | Default processing | 48 kHz stereo output | Notes |
|---|---:|---|---|---|
| 8 / 16 kHz | 4 / 8 kHz | Optional **Sidon** restoration, after explicit `--lowrate-method sidon` | 48 kHz | Generated high-frequency content is model-estimated, not recovered ground truth; default `reject` |
| 22.05 kHz | 11.025 kHz | Optional **Sidon** restoration, after explicit `--lowrate-method sidon` | 48 kHz | Default `reject`; `resample` is also possible, but adds no actual bandwidth |
| 24 kHz | 12 kHz | **FFmpeg resampling ×2** | 48 kHz | Original information above 12 kHz is absent; Sidon is **not** applied by default |
| 44.1 kHz | 22.05 kHz | **FFmpeg resampling ×(160/147)** | 48 kHz | Spectral information between 22.05–24 kHz is absent; ordinary resampling is preferred |
| 48 kHz | 24 kHz | Preserve / stereo PCM conversion | 48 kHz | No sample-rate conversion necessary |
| 96 kHz | 48 kHz | FFmpeg downsampling ×0.5 | 48 kHz | Anti-alias low-pass removes frequencies above 24 kHz |

All rows assume **genuine 2-channel input**. Mono and >2-channel files are not automatically admitted. The processing stage preserves channel ordering (unless the approved channel roles explicitly request swapping) and retains the original file separately.

**FFmpeg examples** for verified stereo audio:

```bash
# 24 kHz -> 48 kHz, with separate left/right channels
ffmpeg -i input_24k.wav -af "pan=stereo|c0=c0|c1=c1" -ar 48000 -c:a pcm_s16le output_48k.wav

# 44.1 kHz -> 48 kHz, with separate left/right channels
ffmpeg -i input_44k.wav -af "pan=stereo|c0=c0|c1=c1" -ar 48000 -c:a pcm_s16le output_48k.wav
```

Resampling changes the sample grid, **not** the original high-frequency information. Sidon restores or synthesizes features and may alter timbre or fine timing. All processed data remains **48 kHz stereo**; the Whisper annotation stage resamples transiently to 16 kHz for ASR without replacing stored audio. The stock Kyutai Mimi audio encoder expects 24 kHz, so a separate training-input adaptation is still needed.

## 48 kHz retention policy (2026-11-19)

All processed two-channel audio is saved at **48,000 Hz**. Sidon-generated 48 kHz waveforms are saved directly, without any 48→24 kHz reduction. Files with other rates are converted to 48 kHz in the normalization stage. Original recordings are preserved. The annotation script resamples transiently to 16 kHz for Whisper ASR and writes timestamps relative to the original duration; it does not replace or alter the stored 48 kHz file.

**Training compatibility:** the released Kyutai Moshi/Mimi tokenizer ordinarily expects 24 kHz waveforms. The files from this pipeline are **not plug-and-play Moshi audio inputs** until a downstream training-time adapter/resampler handles 48→24 kHz, or the audio tokenizer is explicitly replaced/retrained for 48 kHz. This repository intentionally does not reduce the processed audio to 24 kHz.
