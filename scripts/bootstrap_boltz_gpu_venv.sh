#!/usr/bin/env bash
# Create a separate upstream-Boltz environment for GPU preprocessing/prediction.
#
# Keep this distinct from repo .venv: Boltr's Rust/tch build currently needs
# torch==2.3.0 headers, while newer Blackwell GPUs need a newer PyTorch wheel.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_VENV="$ROOT/.venv-boltz"
if [[ -d /workspace ]]; then
  DEFAULT_VENV="/workspace/boltr-envs/boltz-gpu"
fi

VENV="${BOLTR_BOLTZ_VENV:-$DEFAULT_VENV}"
PYTHON="${BOLTR_BOLTZ_PYTHON:-python3}"
TMPBASE="${BOLTR_BOLTZ_TMPDIR:-${TMPDIR:-}}"
if [[ -z "$TMPBASE" && -d /workspace ]]; then
  TMPBASE="/workspace/tmp"
fi
if [[ -z "${PIP_CACHE_DIR:-}" && -d /workspace ]]; then
  export PIP_CACHE_DIR="/workspace/pip-cache"
fi

mkdir -p "$(dirname "$VENV")"
if [[ -n "$TMPBASE" ]]; then
  mkdir -p "$TMPBASE"
  export TMPDIR="$TMPBASE"
fi
if [[ -n "${PIP_CACHE_DIR:-}" ]]; then
  mkdir -p "$PIP_CACHE_DIR"
fi

if [[ ! -d "$VENV" ]]; then
  # --system-site-packages lets RunPod images with a compatible system torch
  # reuse that large CUDA wheel instead of duplicating it into the venv.
  "$PYTHON" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install boltz

"$VENV/bin/python" - <<'PY'
import json
import sys

import torch

cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
sm = f"sm_{cap[0]}{cap[1]}" if cap else None
archs = torch.cuda.get_arch_list()
print(json.dumps({
    "python": sys.executable,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_arch": sm,
    "torch_archs": archs,
    "compatible": (sm in archs) if sm else None,
}, indent=2))
if sm and sm not in archs:
    raise SystemExit(f"PyTorch does not support visible GPU architecture {sm}")
PY

echo
echo "Boltz CLI: $VENV/bin/boltz"
echo "Use with:"
echo "  export BOLTR_BOLTZ_COMMAND=\"$VENV/bin/boltz\""
echo "  export BOLTR_BOLTZ_USE_KERNELS=0"
