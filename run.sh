#!/usr/bin/env bash
# One-command entry point: public datasets -> inspection -> stereo -> Sidon -> Whisper.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash run.sh                       Download free public datasets + GPU processing
  bash run.sh --check               Audit datasets only
  bash run.sh --check --catalog data/unified_catalog_500.json
  bash run.sh --dry-run --catalog data/unified_catalog_500.json --include-restricted
  bash run.sh --catalog data/unified_catalog_500.json --include-restricted
  bash run.sh --dry-run             Show estimated download sizes
  bash run.sh --inspect             Inspect existing raw audio (no GPU/models)
  bash run.sh --inspect --raw-dir /path/to/audio
  bash run.sh --process-existing    Process already downloaded audio under raw/
  bash run.sh --lang id             Limit acquisition to a catalog language
  bash run.sh --device cuda:1       Select GPU for Whisper
  bash run.sh --skip-install        Do not install Python dependencies
All additional options are forwarded to download_free.sh.
EOF
  exit 0
fi

MODE=download
SKIP_INSTALL=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --check) MODE=check; ARGS+=("$arg");;
    --dry-run) MODE=dry-run; ARGS+=("$arg");;
    --inspect) MODE=inspect; ARGS+=("$arg");;
    --process-existing) MODE=existing; ARGS+=("$arg");;
    --download) MODE=download; ARGS+=("$arg");;
    --skip-install) SKIP_INSTALL=1;;
    --acknowledge-rights) echo "Use the interactive license confirmation in run.sh instead."; exit 2;;
    *) ARGS+=("$arg");;
  esac
done

command -v python3 >/dev/null || { echo "Python 3.10+ required" >&2; exit 1; }
if [[ "$MODE" == download || "$MODE" == existing || "$MODE" == inspect ]]; then
  command -v ffmpeg >/dev/null || { echo "ffmpeg missing; install: sudo apt-get install ffmpeg" >&2; exit 1; }
  command -v ffprobe >/dev/null || { echo "ffprobe missing; install: sudo apt-get install ffmpeg" >&2; exit 1; }
fi

if [[ ! -f .venv/bin/python ]]; then
  python3 -m venv .venv || { echo "Python venv failed; install python3-venv" >&2; exit 1; }
fi
export PYTHON="$ROOT/.venv/bin/python"
if (( ! SKIP_INSTALL )); then
  if [[ "$MODE" == inspect ]]; then
    "$PYTHON" -m pip install "numpy>=1.26,<3"
  else
    "$PYTHON" -m pip install -r requirements-sidon.txt
  fi
fi

# Download/inspection alone are CPU-capable, but model inference never falls back.
if [[ "$MODE" == download || "$MODE" == existing ]]; then
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU unavailable. Install matching CUDA torch/torchaudio; no CPU fallback.")
print("CUDA available:", torch.cuda.device_count(), "GPU(s)")
PY
fi

if [[ "$MODE" == download ]]; then
  echo
  echo "By default, only catalog-marked public datasets will be attempted."
  echo "Use --catalog data/unified_catalog_500.json --include-restricted for HF sources in unified catalog."
  echo "Some free-to-download datasets forbid commercial use; you must review each source license."
  echo "Downloads may be very large, and archives/Parquet are not automatically unpacked."
  echo "This does not purchase, bypass gates, or grant training rights."
  if [[ ! -t 0 ]]; then
    echo "Interactive terminal required to confirm terms. Run directly in a terminal." >&2
    exit 2
  fi
  read -r -p "I have reviewed the rights and want to download publicly accessible files [yes/no]: " answer
  if [[ "$answer" != yes ]]; then
    echo "Cancelled."
    exit 0
  fi
  ARGS+=(--acknowledge-rights)
fi

if [[ "$MODE" == download ]]; then
  ARGS=(--download "${ARGS[@]}")
fi
exec bash "$ROOT/download_free.sh" "${ARGS[@]}"
