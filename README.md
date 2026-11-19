# Multilingual Duplex Dataset Acquisition → Moshi Preprocessing

Dataset catalog (100 entries), conservative Hugging Face downloader, and **Kyutai `moshi-finetune`-style** 24 kHz stereo WAV / JSONL preparation.

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

This creates `processed/moshi/data_stereo/*.wav` (24 kHz 16-bit stereo), `processed/moshi/train.jsonl`, `provenance.jsonl`, optional energy-based `.activity.json` (80 ms bins), and `errors.json`.

**Attention:** stereo does not automatically mean isolated speakers. If stereo is spatial room recording or podcast mix, first verify channel provenance and perform appropriate speaker separation. Mono sources are skipped by default; `--mono-policy duplicate` only tests the pipeline and does **not** reconstruct two speakers. No arbitrary output silence/padding is added. The 100-second slicing may cut turns at clip boundaries; production quality checks should account for this.

## Step 4 — End-to-end download + Moshi annotation

Use `download_proc.sh` to run **catalog download → 24 kHz stereo conversion → word-level alignment**.

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

## GitHub publishing

Create or identify an existing GitHub repository and share its `https://github.com/OWNER/REPO` URL in ChatGPT. Upload **only code, catalog, docs and tests**; keep `raw/`, `processed/`, personal recordings, credentials, and accepted license documents private. A connected GitHub account with write access is required.

## References

- [Kyutai Moshi Fine-tune README](https://github.com/kyutai-labs/moshi-finetune#prepare-dataset)
- [HF snapshot_download docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)