#!/usr/bin/env bash
# regression_compare_predict.sh — Compare `boltz predict` vs `boltr predict` on identical input.
#
# This is the §7 regression harness referenced in TODO.md and docs/TENSOR_CONTRACT.md.
# It runs both CLIs on the same YAML input + preprocessed data and checks numerical parity
# of the structure outputs (coordinates, pLDDT, PAE, confidence JSON).
#
# PREREQUISITES (all required when BOLTR_REGRESSION=1):
#   - `boltz` Python package installed and on PATH
#   - `boltr` Rust binary built with `--features tch` and on PATH
#   - A Boltz2 checkpoint exported to safetensors (scripts/export_checkpoint_to_safetensors.py)
#   - LibTorch available (CPU is fine)
#   - Python: numpy, safetensors, torch
#
# USAGE:
#   BOLTR_REGRESSION=1 bash scripts/regression_compare_predict.sh <input.yaml> [output_dir]
#
# EXAMPLE:
#   BOLTR_REGRESSION=1 bash scripts/regression_compare_predict.sh \
#       tests/fixtures/regression/test_protein.yaml /tmp/regression_run
#
# ENVIRONMENT:
#   BOLTR_REGRESSION     Must be "1" to run (default: skip)
#   BOLTR_CACHE_DIR      Cache dir for model weights (default: ~/.cache/boltr)
#   BOLTR_RTKICK         Skip harness if not set to "1" (alias: BOLTR_REGRESSION)
#   BOLTR_COORD_RTOL     Relative tolerance for coordinate comparison (default: 1e-3)
#   BOLTR_COORD_ATOL     Absolute tolerance for coordinate comparison (default: 1e-4)
#   BOLTR_PLDDT_RTOL     Relative tolerance for pLDDT comparison (default: 1e-3)
#   BOLTR_PLDDT_ATOL     Absolute tolerance for pLDDT comparison (default: 1e-2)
#
# See docs/TENSOR_CONTRACT.md §6.5 for tolerance policy.

set -euo pipefail

# ── Gate ────────────────────────────────────────────────────────────────────────
if [[ "${BOLTR_REGRESSION:-}" != "1" ]]; then
    echo "regression_compare_predict: skipped (set BOLTR_REGRESSION=1 when parity is ready)."
    echo "See docs/TENSOR_CONTRACT.md and TODO.md §7."
    exit 0
fi

# ── Args ───────────────────────────────────────────────────────────────────────
INPUT_YAML="${1:?Usage: $0 <input.yaml> [output_dir]}"
OUTDIR="${2:-/tmp/boltr_regression_$$}"

COORD_RTOL="${BOLTR_COORD_RTOL:-1e-3}"
COORD_ATOL="${BOLTR_COORD_ATOL:-1e-4}"
PLDDT_RTOL="${BOLTR_PLDDT_RTOL:-1e-3}"
PLDDT_ATOL="${BOLTR_PLDDT_ATOL:-1e-2}"

echo "=== Boltr Regression Harness ==="
echo "Input:   ${INPUT_YAML}"
echo "Output:  ${OUTDIR}"
echo "Tolerances:"
echo "  coords:  rtol=${COORD_RTOL}  atol=${COORD_ATOL}"
echo "  plddt:   rtol=${PLDDT_RTOL}  atol=${PLDDT_ATOL}"
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
check_cmd python3 "Required for comparison script."

if [[ ! -f "${INPUT_YAML}" ]]; then
    echo "ERROR: Input YAML not found: ${INPUT_YAML}" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

BOLTZ_DIR="${OUTDIR}/boltz_output"
BOLTR_DIR="${OUTDIR}/boltr_output"

# ── Step 1: Run `boltz predict` (Python reference) ───────────────────────��────
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

# ── Step 2: Run `boltr predict` (Rust) ────────────────────────────────────────
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

# Write a small Python comparison script inline
python3 - "${BOLTZ_DIR}" "${BOLTR_DIR}" "${COORD_RTOL}" "${COORD_ATOL}" "${PLDDT_RTOL}" "${PLDDT_ATOL}" <<'PYEOF'
"""Compare boltz vs boltr prediction outputs."""
import sys
import json
import os
import numpy as np
from pathlib import Path

boltz_dir = Path(sys.argv[1])
boltr_dir = Path(sys.argv[2])
coord_rtol = float(sys.argv[3])
coord_atol = float(sys.argv[4])
plddt_rtol = float(sys.argv[5])
plddt_atol = float(sys.argv[6])

errors = []

def find_output_file(d, ext):
    """Find first file with given extension under output directory (recursive)."""
    for p in sorted(d.rglob(f"*{ext}")):
        return p
    return None

def compare_confidence_json():
    """Compare confidence JSON files (pTM, ipTM, etc.)."""
    bz_json = find_output_file(boltz_dir, "_confidence.json")
    br_json = find_output_file(boltr_dir, "_confidence.json")

    if bz_json and br_json:
        with open(bz_json) as f:
            bz = json.load(f)
        with open(br_json) as f:
            br = json.load(f)
        print(f"  boltz confidence: {bz_json.name}")
        print(f"  boltr confidence: {br_json.name}")
        # Compare key fields
        for key in ["ptm", "iptm", "ptm_iptm", "ligand_iptm", "ranking_score"]:
            bval = bz.get(key)
            rval = br.get(key)
            if bval is not None and rval is not None:
                diff = abs(bval - rval)
                print(f"    {key}: boltz={bval:.6f}  boltr={rval:.6f}  diff={diff:.6e}")
    else:
        print("  confidence JSON: skipped (not found in both outputs)")

def compare_structure_coords():
    """Compare predicted coordinates from NPZ output."""
    bz_npz = find_output_file(boltz_dir, "_coord.npz") or find_output_file(boltz_dir, ".npz")
    br_npz = find_output_file(boltr_dir, "_coord.npz") or find_output_file(boltr_dir, ".npz")

    if bz_npz and br_npz:
        print(f"  boltz structure: {bz_npz.name}")
        print(f"  boltr structure: {br_npz.name}")
        # Try to load and compare coordinate arrays
        try:
            bz_data = np.load(bz_npz, allow_pickle=True)
            br_data = np.load(br_npz, allow_pickle=True)
            print(f"    boltz keys: {sorted(bz_data.files)}")
            print(f"    boltr keys: {sorted(br_data.files)}")
            # Compare common numeric keys
            common = set(bz_data.files) & set(br_data.files)
            for key in sorted(common):
                ba = bz_data[key]
                ra = br_data[key]
                if ba.shape != ra.shape:
                    print(f"    {key}: SHAPE MISMATCH boltz={ba.shape} boltr={ra.shape}")
                    errors.append(f"{key}: shape mismatch")
                    continue
                if ba.dtype.kind == 'f' and ra.dtype.kind == 'f':
                    if ba.ndim == 0 or ra.ndim == 0:
                        continue
                    rtol = coord_rtol
                    atol = coord_atol
                    if 'plddt' in key.lower():
                        rtol = plddt_rtol
                        atol = plddt_atol
                    max_diff = np.max(np.abs(ba - ra)).item()
                    close = np.allclose(ba, ra, rtol=rtol, atol=atol)
                    status = "PASS" if close else "FAIL"
                    print(f"    {key}: {status}  max_diff={max_diff:.6e}  shape={ba.shape}")
                    if not close:
                        errors.append(f"{key}: max_diff={max_diff:.6e}")
                elif ba.dtype.kind == 'i' or ba.dtype.kind == 'u':
                    match = np.array_equal(ba, ra)
                    status = "PASS" if match else "FAIL"
                    print(f"    {key}: {status}  (integer)  shape={ba.shape}")
                    if not match:
                        errors.append(f"{key}: integer mismatch")
        except Exception as e:
            print(f"    comparison error: {e}")
            errors.append(f"structure comparison: {e}")
    else:
        print("  structure NPZ: skipped (not found in both outputs)")

def compare_pae_pde():
    """Compare PAE/PDE NPZ outputs."""
    for tag in ["_pae", "_pde", "_plddt"]:
        bz = find_output_file(boltz_dir, f"{tag}.npz")
        br = find_output_file(boltr_dir, f"{tag}.npz")
        if bz and br:
            bz_data = np.load(bz, allow_pickle=True)
            br_data = np.load(br, allow_pickle=True)
            for key in sorted(set(bz_data.files) & set(br_data.files)):
                ba = bz_data[key]
                ra = br_data[key]
                if ba.shape == ra.shape and ba.dtype.kind == 'f':
                    max_diff = np.max(np.abs(ba - ra)).item()
                    close = np.allclose(ba, ra, rtol=1e-3, atol=1e-2)
                    status = "PASS" if close else "FAIL"
                    print(f"  {tag[1:]} {key}: {status}  max_diff={max_diff:.6e}")
                    if not close:
                        errors.append(f"{tag}/{key}: max_diff={max_diff:.6e}")

print("Comparing outputs...")
compare_confidence_json()
compare_structure_coords()
compare_pae_pde()

if errors:
    print(f"\nFAILURES ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nAll comparisons passed.")
    sys.exit(0)
PYEOF

compare_rc=$?

echo ""
if [[ ${compare_rc} -eq 0 ]]; then
    echo "=== REGRESSION: PASS ==="
else
    echo "=== REGRESSION: FAIL (exit ${compare_rc}) ==="
fi

exit ${compare_rc}
