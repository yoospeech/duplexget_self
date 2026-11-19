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
python src/prepare_moshi.py --input incoming --out processed/moshi --chunk-sec 100
# If your corpus uses right channel = assistant:
python src/prepare_moshi.py --input incoming --out processed/moshi --swap-channels
```

This creates `processed/moshi/data_stereo/*.wav` (24 kHz 16-bit stereo), `processed/moshi/train.jsonl`, `provenance.jsonl`, optional energy-based `.activity.json` (80 ms bins), and `errors.json`.

**Attention:** stereo does not automatically mean isolated speakers. If stereo is spatial room recording or podcast mix, first verify channel provenance and perform appropriate speaker separation. Mono sources are skipped by default; `--mono-policy duplicate` only tests the pipeline and does **not** reconstruct two speakers. No arbitrary output silence/padding is added. The 100-second slicing may cut turns at clip boundaries; production quality checks should account for this.

### Optional ASR preview

```bash
python src/transcribe_channels.py --root processed/moshi --model small --device cpu --language ko
```

Produces `.asr.json`, **not** the official word-aligned transcript expected by Moshi. For true training compatibility, obtain Kyutai's repository and run its documented annotator:

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git
cd moshi-finetune
# Resolve paths per checked-out commit and annotate transcript files:
python annotate.py /ABS/PATH/TO/processed/moshi/train.jsonl
```

The aligned `.json` next to each `.wav` should be verified manually against the precise Kyutai commit. **This project prepares a training-input scaffold, not turnkey training-ready tokenized samples.** Mimi encoding and temporal/depth delay logic remain the responsibility of Moshi's training pipeline; the 80 ms RMS activity is neither VAD nor diarization or ground-truth overlap labels.

## GitHub publishing

Create or identify an existing GitHub repository and share its `https://github.com/OWNER/REPO` URL in ChatGPT. Upload **only code, catalog, docs and tests**; keep `raw/`, `processed/`, personal recordings, credentials, and accepted license documents private. A connected GitHub account with write access is required.

## References

- [Kyutai Moshi Fine-tune README](https://github.com/kyutai-labs/moshi-finetune#prepare-dataset)
- [HF snapshot_download docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)