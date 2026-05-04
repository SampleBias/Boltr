#!/usr/bin/env bash
# Start boltr-web with env vars so upstream `boltz` and repo `.venv` resolve reliably
# (Boltz preprocess / auto), even when the shell cwd is not the repo root.
#
# After `bootstrap_webui_env.sh` or `./Boltr_go`:  bash scripts/run_boltr_web.sh
# Pass boltr-web flags as extra args, e.g.:  bash scripts/run_boltr_web.sh --listen 127.0.0.1:3000
#
# Env: BOLTR_INSTALL_BOLTZ=0 during bootstrap skips pip install boltz — install manually or set BOLTR_BOLTZ_COMMAND.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CARGO_TARGET_DIR:-}" && -d /workspace ]]; then
  export CARGO_TARGET_DIR=/tmp/boltr-target
fi
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
WEB_BIN="$TARGET_DIR/release/boltr-web"

if [[ ! -x "$WEB_BIN" ]]; then
  echo "ERROR: missing $WEB_BIN — run bash scripts/bootstrap_webui_env.sh or cargo build -p boltr-web --release" >&2
  exit 1
fi

export BOLTR_REPO="$ROOT"
export BOLTR="${TARGET_DIR}/release/boltr"
if [[ -z "${BOLTR_BOLTZ_COMMAND:-}" && -x "$ROOT/scripts/boltz-gpu" ]]; then
  export BOLTR_BOLTZ_COMMAND="$ROOT/scripts/boltz-gpu"
elif [[ -z "${BOLTR_BOLTZ_COMMAND:-}" && -x "$ROOT/.venv/bin/boltz" ]]; then
  export BOLTR_BOLTZ_COMMAND="$ROOT/.venv/bin/boltz"
fi

exec "$ROOT/scripts/with_dev_venv.sh" "$WEB_BIN" "$@"
