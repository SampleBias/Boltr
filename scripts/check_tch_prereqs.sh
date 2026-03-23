#!/usr/bin/env bash
# Quick diagnostics for LibTorch / PyTorch vs boltr-backend-tch (tch-rs).
# Does not modify your system. See DEVELOPMENT.md § Build with tch-rs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== boltr tch-rs prerequisites ==="
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3: not found (install python; for Arch: sudo pacman -S python)"
  exit 1
fi

echo "python3: $(command -v python3)"
python3 -c 'import sys; print("  executable:", sys.executable)'

if python3 -m pip --version >/dev/null 2>&1; then
  echo "pip: ok"
else
  echo "pip: MISSING — Arch: sudo pacman -S python-pip"
  echo "       or: python3 -m ensurepip --user"
fi

if python3 -c 'import torch' >/dev/null 2>&1; then
  python3 -c 'import torch; print("torch: ok —", torch.__version__)'
else
  echo "torch: MISSING on PATH python3 — needed for LIBTORCH_USE_PYTORCH=1 and golden scripts"
  echo "       Arch: system pip cannot install torch (PEP 668) — run: bash scripts/bootstrap_dev_venv.sh"
  echo "       Then:  scripts/with_dev_venv.sh cargo test -p boltr-backend-tch --features tch-backend"
  echo "       Or use standalone LibTorch (DEVELOPMENT.md Path A) and unset LIBTORCH_USE_PYTORCH."
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VPY="$("$ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  echo "repo .venv: Python $VPY"
  case "$VPY" in
    3.13|3.14|3.15)
      echo "  WARNING: tch 0.16 libtch C++ is built for PyTorch 2.3 + older Pythons; 3.13+ + latest torch → compile errors (hasORT, _scaled_mm, …)."
      echo "           Fix: rm -rf .venv && bash scripts/bootstrap_dev_venv.sh   (uses python3.12 if installed)"
      ;;
  esac
  if "$ROOT/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
    TV="$("$ROOT/.venv/bin/python" -c 'import torch; print(torch.__version__.split("+")[0])')"
    echo "repo .venv: torch $TV"
    case "$TV" in
      2.3.*) echo "  (aligned with tch 0.16 / torch-sys 2.3.0 line)" ;;
      *)
        echo "  WARNING: C++ headers may not match tch 0.16 — expect libtch build failures unless you use torch 2.3.x or Path A LibTorch 2.3.0."
        ;;
    esac
    echo "  use: scripts/cargo-tch … or source .venv/bin/activate"
  else
    echo "repo .venv: torch missing — run: bash scripts/bootstrap_dev_venv.sh"
  fi
else
  echo "repo .venv: (not created yet — bash scripts/bootstrap_dev_venv.sh)"
fi

echo
if [[ -n "${LIBTORCH:-}" ]]; then
  echo "LIBTORCH=$LIBTORCH"
else
  echo "LIBTORCH: (unset)"
fi

if [[ "${LIBTORCH_USE_PYTORCH:-}" == "1" ]]; then
  echo "LIBTORCH_USE_PYTORCH=1 (torch-sys will probe python3 + torch)"
else
  echo "LIBTORCH_USE_PYTORCH: (unset or not 1)"
fi

if [[ "${LIBTORCH_BYPASS_VERSION_CHECK:-}" == "1" ]]; then
  echo "LIBTORCH_BYPASS_VERSION_CHECK=1 (skip tch vs PyTorch version guard — see DEVELOPMENT.md)"
elif [[ -n "${LIBTORCH_BYPASS_VERSION_CHECK:-}" ]]; then
  echo "LIBTORCH_BYPASS_VERSION_CHECK=$LIBTORCH_BYPASS_VERSION_CHECK"
else
  echo "LIBTORCH_BYPASS_VERSION_CHECK: (unset — torch-sys may fail if pip torch >> tch 0.16 expectation; use 1 or scripts/with_dev_venv.sh)"
fi

echo
echo "Next: DEVELOPMENT.md — Path A (LIBTORCH zip), or Path B (venv + LIBTORCH_USE_PYTORCH=1)."
