#!/usr/bin/env bash
# Run a command with repo .venv first on PATH so python3 == .venv (for torch-sys + LIBTORCH_USE_PYTORCH=1).
# Usage: scripts/with_dev_venv.sh cargo test -p boltr-backend-tch --features tch-backend

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_BIN="$ROOT/.venv/bin"
if [[ ! -x "$VENV_BIN/python" ]]; then
  echo "Missing $VENV_BIN/python — run: bash scripts/bootstrap_dev_venv.sh" >&2
  exit 1
fi
if ! "$VENV_BIN/python" -c 'import sys; v=sys.version_info; raise SystemExit(0 if v.major==3 and 10<=v.minor<=12 else 1)'; then
  VNOW="$("$VENV_BIN/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  echo "ERROR: .venv is Python ${VNOW}; tch 0.16 needs 3.10–3.12 + torch 2.3.x headers." >&2
  echo "Recreate: bash scripts/bootstrap_dev_venv.sh --force   (Arch: yay -S python312 or pyenv 3.12 — see DEVELOPMENT.md)" >&2
  exit 1
fi
export PATH="$VENV_BIN:$PATH"
export LIBTORCH_USE_PYTORCH=1
# tch 0.16 expects LibTorch ~2.3; newer pip `torch` (e.g. 2.11) triggers a build-script check unless bypassed.
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
exec "$@"
