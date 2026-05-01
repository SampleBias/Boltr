#!/usr/bin/env bash
# One-shot setup: dev venv, LibTorch via PyTorch, tch-enabled boltr + boltr-web, model download,
# Lightning .ckpt → .safetensors export, and boltz2_hparams.json (from boltz2_conf.ckpt) for native predict.
#
# Usage (from repo root):
#   bash scripts/bootstrap_webui_env.sh
#
# Env:
#   BOLTZ_CACHE   — model cache (default: ~/.cache/boltr)
#   BOLTR_TORCH_VERSION — passed to bootstrap_dev_venv (default 2.3.0)
#   CARGO_TARGET_DIR — cargo build output (default: /tmp/boltr-target on /workspace systems)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CARGO_TARGET_DIR:-}" && -d /workspace ]]; then
  export CARGO_TARGET_DIR=/tmp/boltr-target
fi
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  mkdir -p "$CARGO_TARGET_DIR"
fi
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

bash scripts/bootstrap_dev_venv.sh

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

echo "==> Building boltr (tch) and boltr-web (release)"
scripts/cargo-tch build --release -p boltr-cli --features tch
scripts/cargo-tch build --release -p boltr-web

CACHE="${BOLTZ_CACHE:-$HOME/.cache/boltr}"
mkdir -p "$CACHE"
export BOLTZ_CACHE="$CACHE"

BOLTR_BIN="$TARGET_DIR/release/boltr"
if [[ ! -x "$BOLTR_BIN" ]]; then
  echo "ERROR: missing $BOLTR_BIN after build" >&2
  exit 1
fi

echo "==> Downloading Boltz2 assets into $CACHE"
"$ROOT/scripts/with_dev_venv.sh" "$BOLTR_BIN" download --version boltz2 --cache-dir "$CACHE"

PY="$ROOT/.venv/bin/python"
CONF_CKPT="$CACHE/boltz2_conf.ckpt"
CONF_SF="$CACHE/boltz2_conf.safetensors"
AFF_CKPT="$CACHE/boltz2_aff.ckpt"
AFF_SF="$CACHE/boltz2_aff.safetensors"

echo "==> Exporting safetensors for Rust-native load"
"$PY" "$ROOT/scripts/export_checkpoint_to_safetensors.py" "$CONF_CKPT" "$CONF_SF"
"$PY" "$ROOT/scripts/export_checkpoint_to_safetensors.py" "$AFF_CKPT" "$AFF_SF"

HPARAMS_JSON="$CACHE/boltz2_hparams.json"
echo "==> Exporting Lightning hyperparameters JSON ($HPARAMS_JSON)"
if [[ -f "$CONF_CKPT" ]]; then
  "$PY" "$ROOT/scripts/export_hparams_from_ckpt.py" "$CONF_CKPT" "$HPARAMS_JSON"
else
  echo "WARN: missing $CONF_CKPT — skip boltz2_hparams.json (re-run boltr download)" >&2
fi

echo
echo "==> Optional: verify VarStore keys (adjust --token-s / --blocks if your checkpoint differs)"
echo "    scripts/cargo-tch run -p boltr-backend-tch --bin verify_boltz2_safetensors --features tch-backend -- \\"
echo "      $CONF_SF"
echo
echo "==> Sanity: LibTorch runtime"
"$ROOT/scripts/with_dev_venv.sh" "$BOLTR_BIN" doctor --json

echo
echo "Done. Cache: $CACHE"
echo "Run Web UI:  $ROOT/scripts/with_dev_venv.sh $TARGET_DIR/release/boltr-web"
echo "Set BOLTR=$BOLTR_BIN so boltr-web can probe the tch-enabled binary (optional)."
