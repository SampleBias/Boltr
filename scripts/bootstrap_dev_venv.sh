#!/usr/bin/env bash
# Create repo-local .venv with a PyTorch version whose **C++ headers match tch 0.16**
# (torch-sys bundles libtch generated for LibTorch / PyTorch 2.3.0).
#
# - Uses Python 3.12 / 3.11 / 3.10 (NOT 3.13+): PyTorch 2.3 wheels + stable ABI for tch.
# - Pins torch==2.3.0 so C++ compile of libtch succeeds (bypassing the version *string* check is NOT enough).
#
# Run:  bash scripts/bootstrap_dev_venv.sh
#       bash scripts/bootstrap_dev_venv.sh --force   # remove existing .venv first
#
# Then: scripts/cargo-tch test -p boltr-backend-tch --features tch-backend

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Must match boltr-backend-tch's tch (torch-sys) expected LibTorch API.
TORCH_PIN="${BOLTR_TORCH_VERSION:-2.3.0}"

if [[ "${1:-}" == "--force" ]]; then
  rm -rf .venv
  shift || true
fi

pick_venv_python() {
  if [[ -n "${BOLTR_VENV_PYTHON:-}" ]]; then
    if command -v "$BOLTR_VENV_PYTHON" >/dev/null 2>&1; then
      echo "$BOLTR_VENV_PYTHON"
      return
    fi
    echo "BOLTR_VENV_PYTHON=$BOLTR_VENV_PYTHON not found" >&2
    exit 1
  fi
  local p
  for p in python3.12 python3.11 python3.10; do
    if command -v "$p" >/dev/null 2>&1; then
      echo "$p"
      return
    fi
  done
  echo "No python3.12 / python3.11 / python3.10 on PATH." >&2
  echo "tch 0.16 + latest pip 'torch' on Python 3.13+ breaks C++ build (libtch vs ATen headers)." >&2
  echo "Install e.g. Arch: sudo pacman -S python312   then re-run this script." >&2
  exit 1
}

VPY="$(pick_venv_python)"
echo "Using $VPY for .venv ($(command -v "$VPY"))"

# Reject e.g. BOLTR_VENV_PYTHON=python3.14 — libtch in tch 0.16 won't compile against that torch.
if ! "$VPY" -c 'import sys; v=sys.version_info; raise SystemExit(0 if v.major==3 and 10<=v.minor<=12 else 1)'; then
  echo "ERROR: $VPY must be Python 3.10, 3.11, or 3.12 for tch 0.16 + torch ${TORCH_PIN}." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$VPY" -m venv .venv
fi

# Without --force, an old .venv (e.g. python3.14) is kept; pip then installs torch into wrong headers → hasORT / _scaled_mm errors.
if ! .venv/bin/python -c 'import sys; v=sys.version_info; raise SystemExit(0 if v.major==3 and 10<=v.minor<=12 else 1)'; then
  VNOW="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  echo "ERROR: Existing .venv uses Python ${VNOW}, not 3.10–3.12." >&2
  echo "Recreate it:  bash scripts/bootstrap_dev_venv.sh --force" >&2
  exit 1
fi

.venv/bin/pip install -U pip
.venv/bin/pip install "torch==${TORCH_PIN}" safetensors

echo
echo "Installed torch==${TORCH_PIN} + safetensors into $ROOT/.venv"
.venv/bin/python -c "import torch; print('torch:', torch.__version__, 'python:', __import__('sys').version.split()[0])"
echo
echo "Build:"
echo "  scripts/cargo-tch test -p boltr-backend-tch --features tch-backend"
echo "Or:"
echo "  source .venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && cargo test -p boltr-backend-tch --features tch-backend"
