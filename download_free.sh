#!/usr/bin/env bash
# Public datasets -> audio inspection -> stereo-only 48 kHz -> GPU Sidon / Whisper.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
MODE="run"
RAW="$ROOT/raw"
OUT="$ROOT/processed/free_stereo_48k"
LANGS=()
WORKERS=4
ACK=0
DEVICE="cuda:0"
SIDON_DEVICE="cuda:0"
WHISPER_MODEL="medium"
CHUNK_SEC=100
CHANNELS="both"
ASR_LANG="auto"
usage() {
  cat <<'HELP'
Usage:
  bash download_free.sh --download --acknowledge-rights
  bash download_free.sh --check
  bash download_free.sh --dry-run
  bash download_free.sh --process-existing --raw-dir raw/
Options:
  --download              Download catalog-marked public repos; inspect/process
  --check                 Audit public catalog entries only
  --dry-run               Estimate downloadable sizes; do not process audio
  --process-existing      Skip downloading and process local audio
  --acknowledge-rights    Required for download; does not grant training rights
  --lang CODE             Repeatable catalog language selector
  --asr-lang CODE         Whisper language (default auto)
  --raw-dir PATH          Download/inspect location
  --out PATH              48 kHz stereo output
  --device cuda:N         Whisper GPU (default cuda:0)
  --sidon-device cuda:N   Sidon GPU (default cuda:0)
  --whisper-model NAME    Whisper model (default medium)
  --channels assistant|both (default both)
  --chunk-sec SECONDS     Output clip length (default 100)
  --workers N             HF workers per repo
  --help
Notes:
  Only accessible, non-gated, catalog-marked public HF repositories with a
  declared license are downloaded. That does not establish commercial rights.
  Exactly two-channel media is retained, but speaker isolation is NOT proven.
  Mono/multichannel are not converted into fake stereo.
  Archives/Parquet need separate unpacking and are not automatically processed.
HELP
}
while (($#)); do
  case "$1" in
    --download) MODE=run; shift;;
    --check) MODE=check; shift;;
    --dry-run) MODE=dry-run; shift;;
    --process-existing) MODE=existing; shift;;
    --acknowledge-rights) ACK=1; shift;;
    --lang) LANGS+=("$2"); shift 2;;
    --asr-lang) ASR_LANG="$2"; shift 2;;
    --raw-dir) RAW="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --sidon-device) SIDON_DEVICE="$2"; shift 2;;
    --whisper-model) WHISPER_MODEL="$2"; shift 2;;
    --channels) CHANNELS="$2"; shift 2;;
    --chunk-sec) CHUNK_SEC="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument $1" >&2; usage; exit 2;;
  esac
done
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$CHANNELS" == both || "$CHANNELS" == assistant ]] || ! [[ "$DEVICE" =~ ^cuda(:[0-9]+)?$ && "$SIDON_DEVICE" =~ ^cuda(:[0-9]+)?$ ]]; then
  echo "Invalid worker/channel/GPU option" >&2; exit 2
fi
"$PYTHON" -c 'import huggingface_hub' || { echo "Install requirements.txt first" >&2; exit 1; }
COMMON=(--all --public-only)
for lang in "${LANGS[@]}"; do COMMON+=(--lang "$lang"); done
mkdir -p "$ROOT/reports"
if [[ "$MODE" == check ]]; then
  "$PYTHON" src/catalog_tool.py audit "${COMMON[@]}" --out "$ROOT/reports/free_audit.jsonl"
  "$PYTHON" src/public_stats.py --catalog data/catalog.json --audit "$ROOT/reports/free_audit.jsonl"
  exit 0
fi
if [[ "$MODE" == dry-run ]]; then
  "$PYTHON" src/catalog_tool.py download "${COMMON[@]}" --out "$RAW" --dry-run --acknowledge-rights
  "$PYTHON" src/public_stats.py --catalog data/catalog.json --audit "$RAW/download_results.jsonl"
  echo "Dataset-level file-size estimates: $RAW/download_results.jsonl"
  echo "No audio downloaded (dry-run). Use 'bash run.sh' for actual download."
  exit 0
fi
if [[ "$MODE" == run ]]; then
  if (( ACK != 1 )); then echo "Add --acknowledge-rights after checking licenses" >&2; exit 2; fi
  mkdir -p "$RAW"
  RESULT_LOG="$RAW/download_results.jsonl"
  START_LINES=0
  [[ -f "$RESULT_LOG" ]] && START_LINES=$(wc -l < "$RESULT_LOG")
  "$PYTHON" src/catalog_tool.py download "${COMMON[@]}" --out "$RAW" --acknowledge-rights --workers "$WORKERS"
  "$PYTHON" - "$RESULT_LOG" "$START_LINES" <<'PY'
import json, sys
log, offset = sys.argv[1], int(sys.argv[2])
with open(log, encoding="utf-8") as f:
    entries = [json.loads(line) for line in f.readlines()[offset:] if line.strip()]
completed = [x for x in entries if x.get("status") == "downloaded"]
failed = [x for x in entries if x.get("status") == "error"]
print(f"Download summary: {len(completed)} successful, {len(failed)} errors, {len(entries)-len(completed)-len(failed)} skipped")
if not completed:
    print("No datasets downloaded. Skipping audio inspection, Sidon, and Whisper.", file=sys.stderr)
    sys.exit(2)
PY
fi
command -v ffmpeg >/dev/null || { echo "ffmpeg required" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe required" >&2; exit 1; }
"$PYTHON" -c 'import torch,sys; n=torch.cuda.device_count(); print("CUDA GPUs:",n); sys.exit(0 if torch.cuda.is_available() else 1)' || {
  echo "CUDA unavailable; model inference will not fall back to CPU" >&2; exit 1;
}
echo "[1] Audio Inspection"
"$PYTHON" src/inspect_audio.py scan --input "$RAW" --out "$ROOT/reports/free_audio_inspection.jsonl"
echo "[2] Stereo-only audio preparation; Sidon <48 kHz -> 48 kHz"
"$PYTHON" src/prepare_moshi.py --input "$RAW" --out "$OUT" --chunk-sec "$CHUNK_SEC" \
  --lowrate-method sidon --sidon-device "$SIDON_DEVICE" --auto-stereo
if [[ ! -s "$OUT/train.jsonl" ]]; then
  echo "No eligible directly readable stereo audio; inspect report and extract containers first" >&2
  exit 1
fi
echo "[3] GPU Whisper word-level annotation"
"$PYTHON" src/annotate.py "$OUT/train.jsonl" --lang "$ASR_LANG" \
  --whisper_model "$WHISPER_MODEL" --device "$DEVICE" --channels "$CHANNELS" --local
echo "Finished: $OUT/train.jsonl"
echo "CAUTION: Stereo channels not confirmed to contain different speakers."
echo "Stock Moshi/Mimi 24 kHz encoder is not directly compatible with stored 48 kHz WAV."
