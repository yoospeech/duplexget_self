#!/usr/bin/env bash
# Download -> Moshi audio preparation -> Whisper-timestamped word alignments.
# Example: bash download_proc.sh --id 4 --lang id --acknowledge-rights
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
IDS=()
LANG=""
INPUT=""
RAW="$ROOT/raw"
OUT="$ROOT/processed/moshi"
WHISPER_MODEL="medium"
DEVICE="auto"
CHANNELS="assistant"
CHUNK_SEC="100"
DOWNLOAD_ONLY=0
PREP_ONLY=0
ANNOTATE_ONLY=0
ACK=0
INCLUDE_RESTRICTED=0
ALLOW_UNKNOWN=0
INSPECT_ONLY=0
APPROVAL="$ROOT/reports/approved_stereo.jsonl"
ALL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash download_proc.sh --id 4 [--id 5] --lang id --acknowledge-rights
  bash download_proc.sh --prep-only --input /path/to/stereo_audio --lang ko
  bash download_proc.sh --annotate-only --lang ko --whisper-model medium
Options:
  --id N                   Catalog dataset id; repeatable
  --all                    Explicitly select all catalog entries (very large)
  --lang CODE              Dataset language filter; annotation language (en, ko, id, zh, auto)
  --input PATH             Audio input directory; default raw/ downloads
  --raw-dir PATH           Download destination (default raw/)
  --out PATH               Moshi output (default processed/moshi/)
  --whisper-model NAME     Whisper model (default medium)
  --device auto|cuda|cpu   Annotation device
  --channels assistant|both
                          assistant: channel 0 to clip.json (upstream style);
                          both: channel 1 additionally to clip.user.json sidecar
  --chunk-sec SEC          Segment length (default 100)
  --inspect-only           Download if requested, then inspect; no training prep
  --approval-manifest PATH Human-reviewed stereo allowlist (default reports/approved_stereo.jsonl)
  --download-only          Only download
  --prep-only              No download; inspect/prepare/annotate local audio
  --annotate-only          No download/preparation; annotate existing output
  --acknowledge-rights     Confirm you reviewed terms and rights
  --include-restricted     Request restricted datasets if independently authorized
  --allow-unknown-license  Override metadata gate (does not grant rights)
  --dry-run                List intended downloads only; no processing
  --help                   This help
NOTE: Even stereo files require manual channel-identity approval before preprocessing.
NOTE: .tar/.zip/.parquet archives are not extracted automatically.
EOF
}
while (($#)); do
  case "$1" in
    --id) IDS+=("$2"); shift 2;;
    --all) ALL=1; shift;;
    --lang) LANG="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --raw-dir) RAW="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --whisper-model) WHISPER_MODEL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --channels) CHANNELS="$2"; shift 2;;
    --chunk-sec) CHUNK_SEC="$2"; shift 2;;
    --inspect-only) INSPECT_ONLY=1; shift;;
    --approval-manifest) APPROVAL="$2"; shift 2;;
    --download-only) DOWNLOAD_ONLY=1; shift;;
    --prep-only) PREP_ONLY=1; shift;;
    --annotate-only) ANNOTATE_ONLY=1; shift;;
    --acknowledge-rights) ACK=1; shift;;
    --include-restricted) INCLUDE_RESTRICTED=1; shift;;
    --allow-unknown-license) ALLOW_UNKNOWN=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
if (( DOWNLOAD_ONLY + PREP_ONLY + ANNOTATE_ONLY + INSPECT_ONLY > 1 )); then
  echo "Only one --download-only / --prep-only / --annotate-only is allowed" >&2
  exit 2
fi
if [[ "$CHANNELS" != assistant && "$CHANNELS" != both ]]; then
  echo "--channels must be assistant or both" >&2; exit 2
fi
if [[ "$DEVICE" != auto && "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  echo "--device must be auto, cuda, or cpu" >&2; exit 2
fi
command -v "$PYTHON" >/dev/null || { echo "Missing Python: $PYTHON" >&2; exit 1; }
if (( ! PREP_ONLY && ! ANNOTATE_ONLY && !(INSPECT_ONLY && ${#IDS[@]} == 0 && ! ALL) )); then
  if (( ${#IDS[@]} == 0 && ! ALL )); then
    echo "Specify one or more --id N or --all" >&2; exit 2
  fi
  if (( ! ACK )); then
    echo "Specify --acknowledge-rights after reviewing licenses." >&2; exit 2
  fi
  dargs=(download --out "$RAW" --acknowledge-rights)
  for id in "${IDS[@]}"; do dargs+=(--id "$id"); done
  (( ALL )) && dargs+=(--all)
  [[ -n "$LANG" && "$LANG" != auto ]] && dargs+=(--lang "$LANG")
  (( INCLUDE_RESTRICTED )) && dargs+=(--include-restricted)
  (( ALLOW_UNKNOWN )) && dargs+=(--allow-unknown-license)
  (( DRY_RUN )) && dargs+=(--dry-run)
  echo "[1/3] Downloading via Hugging Face"
  "$PYTHON" src/catalog_tool.py "${dargs[@]}"
fi
if (( DOWNLOAD_ONLY || DRY_RUN )); then
  echo "Download stage complete. Stopping as requested."
  exit 0
fi
if (( ! ANNOTATE_ONLY )); then
  input_dir="${INPUT:-$RAW}"
  [[ -d "$input_dir" ]] || { echo "Missing input folder: $input_dir" >&2; exit 1; }
  command -v ffmpeg >/dev/null || { echo "Missing ffmpeg" >&2; exit 1; }
  command -v ffprobe >/dev/null || { echo "Missing ffprobe" >&2; exit 1; }
  echo "[2/4] Inspecting original audio channels and formats"
  "$PYTHON" src/inspect_audio.py scan --input "$input_dir" --out "$OUT/audio_inspection.jsonl"
  if (( INSPECT_ONLY )); then
    echo "Audio inspection completed; no source automatically approved for training."
    exit 0
  fi
  if [[ ! -s "$APPROVAL" ]]; then
    echo "No reviewed stereo allowlist: $APPROVAL" >&2
    echo "Manually verify speakers in original audio, then approve each input file:" >&2
    echo "  python src/inspect_audio.py approve --file /path/to/file.wav --roles left-assistant --reviewed-speakers --out '$APPROVAL'" >&2
    echo "Then run again with --prep-only --input '$input_dir'." >&2
    exit 2
  fi
  echo "[3/4] 24 kHz stereo Moshi preparation (approved files only)"
  pargs=(--input "$input_dir" --out "$OUT" --chunk-sec "$CHUNK_SEC" --approved-manifest "$APPROVAL")
  "$PYTHON" src/prepare_moshi.py "${pargs[@]}"
fi
manifest="$OUT/train.jsonl"
if [[ ! -s "$manifest" ]]; then
  echo "No stereo WAV clips in $manifest. Archives/parquet need dataset-specific extraction;" >&2
  echo "check audio-channel separation and re-run with --prep-only --input /path/to/audio" >&2
  exit 1
fi
echo "[4/4] Word alignment via whisper_timestamped"
lang="${LANG:-en}"
"$PYTHON" src/annotate.py "$manifest" --lang "$lang" \
  --whisper_model "$WHISPER_MODEL" --device "$DEVICE" --channels "$CHANNELS" --local
echo "Done: $manifest"
echo "Alignment files are next to clips. For --channels both the .user.json files are sidecars, not consumed by upstream Moshi automatically."
