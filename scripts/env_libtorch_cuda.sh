#!/usr/bin/env bash
# Source from repo root:  source scripts/env_libtorch_cuda.sh
# Sets LIBTORCH to third_party/libtorch (CUDA cu118 2.3.0) and extends LD_LIBRARY_PATH.
# Requires: unpacked LibTorch at third_party/libtorch (see DEVELOPMENT.md Path A).

set -euo pipefail
_THIS="${BASH_SOURCE[0]:-$0}"
ROOT="$(cd "$(dirname "$_THIS")/.." && pwd)"
export LIBTORCH="${BOLTR_LIBTORCH:-$ROOT/third_party/libtorch}"
# Official cu118/cu121 zips (not *-cxx11-abi-*) ship libtorch built with pre–C++11 ABI.
export LIBTORCH_CXX11_ABI="${LIBTORCH_CXX11_ABI:-0}"
unset LIBTORCH_USE_PYTORCH
if [[ ! -f "$LIBTORCH/lib/libtorch_cuda.so" ]]; then
  echo "Missing $LIBTORCH/lib/libtorch_cuda.so — download cu118 LibTorch 2.3.0 into third_party/libtorch (DEVELOPMENT.md)." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
