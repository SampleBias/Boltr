#!/usr/bin/env bash
# regression_compare_predict.sh — Compare `boltz predict` vs `boltr predict` on identical input.
#
# This is the §7 regression harness referenced in TODO.md and docs/TENSOR_CONTRACT.md §6.
# It runs both CLIs on the same YAML input + preprocessed data and checks numerical parity
# of NPZ outputs (coordinates, pLDDT, PAE/PDE) and prints confidence JSON field diffs.
#
# PREREQUISITES (when BOLTR_REGRESSION=1):
#   - `boltz` on PATH (pip install boltz) — upstream Python CLI
#   - `boltr` on PATH — `cargo build --release -p boltr-cli --features tch`
#   - Python 3 with numpy
#   - LibTorch via system / venv as required by `boltr` (CPU is fine)
#
# USAGE:
#   BOLTR_REGRESSION=1 bash scripts/regression_compare_predict.sh <input.yaml> [output_dir]
#
# EXAMPLE:
#   BOLTR_REGRESSION=1 bash scripts/regression_compare_predict.sh \
#       path/to/input.yaml /tmp/regression_run
#
# ENVIRONMENT:
#   BOLTR_REGRESSION        Must be "1" to run (default: exit 0 skip)
#   BOLTR_REGRESSION_TOL_FILE  Optional: path to a file `source`d after defaults (export KEY=value)
#   BOLTR_CACHE_DIR         Cache for model weights (passed only if you extend the script to use it)
#   BOLTR_COORD_RTOL        Relative tolerance for coordinate NPZ floats (default: 1e-3)
#   BOLTR_COORD_ATOL        Absolute tolerance for coordinate NPZ floats (default: 1e-4)
#   BOLTR_PLDDT_RTOL        Relative tolerance for keys containing "plddt" (default: 1e-3)
#   BOLTR_PLDDT_ATOL        Absolute tolerance for keys containing "plddt" (default: 1e-2)
#   BOLTR_PAE_RTOL          PAE/PDE/plddt NPZ files (default: same as BOLTR_PLDDT_RTOL)
#   BOLTR_PAE_ATOL          (default: same as BOLTR_PLDDT_ATOL)
#
# EXIT CODES:
#   0 — Skipped (BOLTR_REGRESSION!=1) or comparison passed
#   1 — Missing dependency, invalid args, CLI failure, or numeric mismatch
#
# Comparator implementation: scripts/regression_compare_outputs.py
# Tolerance registry: docs/NUMERICAL_TOLERANCES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Gate ────────────────────────────────────────────────────────────────────────
if [[ "${BOLTR_REGRESSION:-}" != "1" ]]; then
    echo "regression_compare_predict: skipped (set BOLTR_REGRESSION=1 when parity is ready)."
    echo "See docs/TENSOR_CONTRACT.md §6 and docs/NUMERICAL_TOLERANCES.md."
    exit 0
fi

# ── Args ───────────────────────────────────────────────────────────────────────
INPUT_YAML="${1:?Usage: $0 <input.yaml> [output_dir]}"
OUTDIR="${2:-/tmp/boltr_regression_$$}"

# Optional tol file (export BOLTR_COORD_RTOL=..., etc.)
if [[ -n "${BOLTR_REGRESSION_TOL_FILE:-}" ]]; then
    if [[ ! -f "${BOLTR_REGRESSION_TOL_FILE}" ]]; then
        echo "ERROR: BOLTR_REGRESSION_TOL_FILE not found: ${BOLTR_REGRESSION_TOL_FILE}" >&2
        exit 1
    fi
    set -a
    # shellcheck source=/dev/null
    source "${BOLTR_REGRESSION_TOL_FILE}"
    set +a
fi

COORD_RTOL="${BOLTR_COORD_RTOL:-1e-3}"
COORD_ATOL="${BOLTR_COORD_ATOL:-1e-4}"
PLDDT_RTOL="${BOLTR_PLDDT_RTOL:-1e-3}"
PLDDT_ATOL="${BOLTR_PLDDT_ATOL:-1e-2}"
PAE_RTOL="${BOLTR_PAE_RTOL:-${PLDDT_RTOL}}"
PAE_ATOL="${BOLTR_PAE_ATOL:-${PLDDT_ATOL}}"

echo "=== Boltr regression harness (boltz vs boltr) ==="
echo "Repo:    ${REPO_ROOT}"
echo "Input:   ${INPUT_YAML}"
echo "Output:  ${OUTDIR}"
echo "Tolerances:"
echo "  coords:     rtol=${COORD_RTOL}  atol=${COORD_ATOL}"
echo "  plddt keys: rtol=${PLDDT_RTOL}  atol=${PLDDT_ATOL}"
echo "  pae npz:    rtol=${PAE_RTOL}  atol=${PAE_ATOL}"
echo ""

# ── Prerequisites ──────────────────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found on PATH. $2" >&2
        exit 1
    fi
}

check_cmd boltz  "Install: pip install boltz"
check_cmd boltr  "Build:   cargo build --release -p boltr-cli --features tch"
check_cmd python3 "Required for scripts/regression_compare_outputs.py"

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "ERROR: Python package 'numpy' required (pip install numpy)" >&2
    exit 1
fi

if [[ ! -f "${INPUT_YAML}" ]]; then
    echo "ERROR: Input YAML not found: ${INPUT_YAML}" >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/regression_compare_outputs.py" ]]; then
    echo "ERROR: Missing ${SCRIPT_DIR}/regression_compare_outputs.py" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

BOLTZ_DIR="${OUTDIR}/boltz_output"
BOLTR_DIR="${OUTDIR}/boltr_output"

# ── Step 1: Run `boltz predict` (Python reference) ─────────────────────────────
echo "--- Step 1: boltz predict (Python) ---"
boltz predict "${INPUT_YAML}" \
    --output "${BOLTZ_DIR}" \
    --override \
    --diffusion_samples 1 \
    --recycling_steps 0 \
    --sampling_steps 10 \
    2>&1 || {
        echo "ERROR: boltz predict failed (see above)" >&2
        exit 1
    }
echo "boltz predict: done"
echo ""

# ── Step 2: Run `boltr predict` (Rust) — section 6 CLI; do not change flags here lightly ──
echo "--- Step 2: boltr predict (Rust) ---"
boltr predict "${INPUT_YAML}" \
    --output "${BOLTR_DIR}" \
    --device cpu \
    --diffusion-samples 1 \
    --recycling-steps 0 \
    --sampling-steps 10 \
    2>&1 || {
        echo "ERROR: boltr predict failed (see above)" >&2
        exit 1
    }
echo "boltr predict: done"
echo ""

# ── Step 3: Compare outputs ───────────────────────────────────────────────────
echo "--- Step 3: Comparing outputs ---"

set +e
python3 "${SCRIPT_DIR}/regression_compare_outputs.py" \
    "${BOLTZ_DIR}" \
    "${BOLTR_DIR}" \
    --coord-rtol "${COORD_RTOL}" \
    --coord-atol "${COORD_ATOL}" \
    --plddt-rtol "${PLDDT_RTOL}" \
    --plddt-atol "${PLDDT_ATOL}" \
    --pae-rtol "${PAE_RTOL}" \
    --pae-atol "${PAE_ATOL}"
compare_rc=$?
set -e

echo ""
if [[ ${compare_rc} -eq 0 ]]; then
    echo "=== REGRESSION: PASS ==="
else
    echo "=== REGRESSION: FAIL (exit ${compare_rc}) ==="
fi

exit "${compare_rc}"
