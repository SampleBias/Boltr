#!/usr/bin/env bash
# Create repo-local .venv with a PyTorch version whose **C++ headers match tch 0.16**
# (torch-sys bundles libtch generated for LibTorch / PyTorch 2.3.0).
#
# - Uses Python 3.12 / 3.11 / 3.10 (NOT 3.13+): PyTorch 2.3 wheels + stable ABI for tch.
# - Pins torch==2.3.0 so C++ compile of libtch succeeds (bypassing the version *string* check is NOT enough).
#
# Run:  bash scripts/bootstrap_dev_venv.sh
#       bash scripts/bootstrap_dev_venv.sh --force   # remove existing .venv first
#       BOLTR_INSTALL_BOLTZ=1 bash scripts/bootstrap_dev_venv.sh   # also install upstream boltz CLI
#
# Then: scripts/cargo-tch test -p boltr-backend-tch --features tch-backend

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Must match boltr-backend-tch's tch (torch-sys) expected LibTorch API.
TORCH_PIN="${BOLTR_TORCH_VERSION:-2.3.0}"

WANT_FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  WANT_FORCE=1
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
  echo "Arch: there is no official 'python312' in core/extra (only latest 'python'). Pick one:" >&2
  echo "  • AUR:  yay -S python312   or   paru -S python312" >&2
  echo "  • Repo: sudo pacman -S pyenv && pyenv install 3.12 && export BOLTR_VENV_PYTHON=\"\$HOME/.pyenv/versions/<3.12.x>/bin/python\"" >&2
  echo "  • Or skip Python torch: DEVELOPMENT.md Path A (LibTorch 2.3.0 zip), unset LIBTORCH_USE_PYTORCH." >&2
  exit 1
}

VPY="$(pick_venv_python)"
echo "Using $VPY for .venv ($(command -v "$VPY"))"

# Reject e.g. BOLTR_VENV_PYTHON=python3.14 — libtch in tch 0.16 won't compile against that torch.
if ! "$VPY" -c 'import sys; v=sys.version_info; raise SystemExit(0 if v.major==3 and 10<=v.minor<=12 else 1)'; then
  echo "ERROR: $VPY must be Python 3.10, 3.11, or 3.12 for tch 0.16 + torch ${TORCH_PIN}." >&2
  exit 1
fi

# Only remove .venv after we know a compatible interpreter exists (so --force can't strand you with no venv).
if [[ "$WANT_FORCE" -eq 1 ]]; then
  rm -rf .venv
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
# torch-sys probes Python via torch.utils.cpp_extension → requires setuptools (not always pulled by torch wheels).
# omegaconf: Lightning/Boltz .ckpt unpickling (torch.load) resolves OmegaConf types in the checkpoint.
# numpy: safetensors.torch serializes tensors through NumPy when exporting checkpoints.
# Keep NumPy <2 so the optional upstream `boltz` CLI remains dependency-compatible.
.venv/bin/pip install setuptools wheel "torch==${TORCH_PIN}" safetensors omegaconf "numpy<2"
if [[ "${BOLTR_INSTALL_BOLTZ:-0}" == "1" ]]; then
  .venv/bin/pip install boltz
fi

echo
echo "Installed setuptools, torch==${TORCH_PIN}, safetensors, omegaconf, numpy into $ROOT/.venv"
if [[ "${BOLTR_INSTALL_BOLTZ:-0}" == "1" ]]; then
  echo "Installed upstream boltz CLI into $ROOT/.venv"
fi
.venv/bin/python -c "import torch; print('torch:', torch.__version__, 'python:', __import__('sys').version.split()[0])"
echo
echo "Build:"
echo "  scripts/cargo-tch test -p boltr-backend-tch --features tch-backend"
echo "Or:"
echo "  source .venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && cargo test -p boltr-backend-tch --features tch-backend"
