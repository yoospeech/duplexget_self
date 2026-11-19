#!/usr/bin/env bash
# Download only freely accessible PUBLIC Hugging Face datasets in data/catalog.json.
# No gated/paid/private repositories, no unknown-license override, no processing.
# Usage: bash download_free.sh --check
#        bash download_free.sh --download --acknowledge-rights
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
MODE="check"
OUT="$ROOT/raw"
LANGS=()
WORKERS=4
ACK=0
usage() {
  cat <<'HELP'
Usage:
  bash download_free.sh --check
  bash download_free.sh --download --acknowledge-rights
  bash download_free.sh --download --acknowledge-rights --lang en
  bash download_free.sh --download --acknowledge-rights --out /mnt/storage/raw

Options:
  --check                 Audit catalog-marked public repos (default). No audio download.
  --dry-run               Ask Hugging Face to estimate downloadable files/sizes. No audio download.
  --download              Download from accessible public repositories.
  --acknowledge-rights    Required for --download. Confirm review of terms.
  --lang CODE             Optional language filter; repeatable (en, ko, id...)
  --out PATH              Directory for downloaded files (default raw/)
  --workers N             Max HF workers per repository (default 4)
  --help                  Print usage

Only entries marked '공개' in the catalog are considered. Before any download,
the downloader rejects inaccessible, gated/private, and missing-license repos.
Some accessible 'free' datasets are NC/research-only, so this script does NOT
imply commercial training rights. ZIP/TAR/Parquet files are not unpacked.
No Sidon, Whisper, or Moshi preprocessing is run.
HELP
}
while (($#)); do
  case "$1" in
    --check) MODE="check"; shift;;
    --dry-run) MODE="dry-run"; shift;;
    --download) MODE="download"; shift;;
    --acknowledge-rights) ACK=1; shift;;
    --lang) LANGS+=("$2"); shift 2;;
    --out) OUT="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer" >&2
  exit 2
fi
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
"$PYTHON" -c 'import huggingface_hub; print("huggingface_hub:", huggingface_hub.__version__)' || {
  echo "Install dependencies first: pip install -r requirements.txt" >&2; exit 1;
}
COMMON=(--all --public-only)
for code in "${LANGS[@]}"; do COMMON+=(--lang "$code"); done
if [[ "$MODE" == "check" ]]; then
  mkdir -p "$ROOT/reports"
  "$PYTHON" src/catalog_tool.py list --public-only "${COMMON[@]:2}"
  echo "Auditing remote Hugging Face accessibility, gated status and license..."
  "$PYTHON" src/catalog_tool.py audit "${COMMON[@]}" --out "$ROOT/reports/free_dataset_audit.jsonl"
  echo "Audit results: reports/free_dataset_audit.jsonl"
elif [[ "$MODE" == "dry-run" ]]; then
  "$PYTHON" src/catalog_tool.py download "${COMMON[@]}" --acknowledge-rights \
    --dry-run --workers "$WORKERS" --out "$OUT"
  echo "Dry-run results: $OUT/download_results.jsonl"
else
  if (( ACK != 1 )); then
    echo "Review dataset licenses, then provide --acknowledge-rights" >&2
    exit 2
  fi
  mkdir -p "$OUT"
  "$PYTHON" src/catalog_tool.py download "${COMMON[@]}" --acknowledge-rights \
    --workers "$WORKERS" --out "$OUT"
  echo "Download results: $OUT/download_results.jsonl"
fi
