#!/usr/bin/env python3
"""Compare `boltz predict` vs `boltr predict` output trees (NPZ + confidence JSON).

Used by `scripts/regression_compare_predict.sh` when `BOLTR_REGRESSION=1`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def find_output_file(root: Path, suffix: str) -> Path | None:
    """First file whose name ends with `suffix` under root (recursive, sorted)."""
    matches = sorted(root.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def compare_confidence_json(
    boltz_dir: Path, boltr_dir: Path, errors: list[str]
) -> None:
    bz_json = find_output_file(boltz_dir, "_confidence.json")
    br_json = find_output_file(boltr_dir, "_confidence.json")

    if bz_json and br_json:
        bz = json.loads(bz_json.read_text(encoding="utf-8"))
        br = json.loads(br_json.read_text(encoding="utf-8"))
        print(f"  boltz confidence: {bz_json.name}")
        print(f"  boltr confidence: {br_json.name}")
        for key in ["ptm", "iptm", "ptm_iptm", "ligand_iptm", "ranking_score"]:
            bval = bz.get(key)
            rval = br.get(key)
            if bval is not None and rval is not None:
                diff = abs(float(bval) - float(rval))
                print(
                    f"    {key}: boltz={float(bval):.6f}  boltr={float(rval):.6f}  diff={diff:.6e}"
                )
    else:
        print("  confidence JSON: skipped (not found in both outputs)")


def compare_structure_coords(
    boltz_dir: Path,
    boltr_dir: Path,
    coord_rtol: float,
    coord_atol: float,
    plddt_rtol: float,
    plddt_atol: float,
    errors: list[str],
) -> None:
    bz_npz = find_output_file(boltz_dir, "_coord.npz") or find_output_file(
        boltz_dir, ".npz"
    )
    br_npz = find_output_file(boltr_dir, "_coord.npz") or find_output_file(
        boltr_dir, ".npz"
    )

    if not (bz_npz and br_npz):
        print("  structure NPZ: skipped (not found in both outputs)")
        return

    print(f"  boltz structure: {bz_npz.name}")
    print(f"  boltr structure: {br_npz.name}")
    try:
        bz_data = np.load(bz_npz, allow_pickle=True)
        br_data = np.load(br_npz, allow_pickle=True)
        print(f"    boltz keys: {sorted(bz_data.files)}")
        print(f"    boltr keys: {sorted(br_data.files)}")
        common = set(bz_data.files) & set(br_data.files)
        for key in sorted(common):
            ba = bz_data[key]
            ra = br_data[key]
            if ba.shape != ra.shape:
                print(f"    {key}: SHAPE MISMATCH boltz={ba.shape} boltr={ra.shape}")
                errors.append(f"{key}: shape mismatch")
                continue
            if ba.dtype.kind == "f" and ra.dtype.kind == "f":
                if ba.ndim == 0 or ra.ndim == 0:
                    continue
                rtol, atol = coord_rtol, coord_atol
                if "plddt" in key.lower():
                    rtol, atol = plddt_rtol, plddt_atol
                max_diff = float(np.max(np.abs(ba - ra)))
                close = np.allclose(ba, ra, rtol=rtol, atol=atol)
                status = "PASS" if close else "FAIL"
                print(
                    f"    {key}: {status}  max_diff={max_diff:.6e}  shape={ba.shape}"
                )
                if not close:
                    errors.append(f"{key}: max_diff={max_diff:.6e}")
            elif ba.dtype.kind in "iu" and ra.dtype.kind in "iu":
                match = np.array_equal(ba, ra)
                status = "PASS" if match else "FAIL"
                print(f"    {key}: {status}  (integer)  shape={ba.shape}")
                if not match:
                    errors.append(f"{key}: integer mismatch")
    except Exception as e:
        print(f"    comparison error: {e}")
        errors.append(f"structure comparison: {e}")


def compare_pae_pde(
    boltz_dir: Path,
    boltr_dir: Path,
    pae_rtol: float,
    pae_atol: float,
    errors: list[str],
) -> None:
    for tag in ["_pae", "_pde", "_plddt"]:
        bz = find_output_file(boltz_dir, f"{tag}.npz")
        br = find_output_file(boltr_dir, f"{tag}.npz")
        if bz and br:
            bz_data = np.load(bz, allow_pickle=True)
            br_data = np.load(br, allow_pickle=True)
            for key in sorted(set(bz_data.files) & set(br_data.files)):
                ba = bz_data[key]
                ra = br_data[key]
                if ba.shape == ra.shape and ba.dtype.kind == "f":
                    max_diff = float(np.max(np.abs(ba - ra)))
                    close = np.allclose(ba, ra, rtol=pae_rtol, atol=pae_atol)
                    status = "PASS" if close else "FAIL"
                    print(
                        f"  {tag[1:]} {key}: {status}  max_diff={max_diff:.6e}"
                    )
                    if not close:
                        errors.append(f"{tag}/{key}: max_diff={max_diff:.6e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("boltz_dir", type=Path, help="boltz predict output root")
    p.add_argument("boltr_dir", type=Path, help="boltr predict output root")
    p.add_argument("--coord-rtol", type=float, default=1e-3)
    p.add_argument("--coord-atol", type=float, default=1e-4)
    p.add_argument("--plddt-rtol", type=float, default=1e-3)
    p.add_argument("--plddt-atol", type=float, default=1e-2)
    p.add_argument(
        "--pae-rtol",
        type=float,
        default=None,
        help="PAE/PDE/plddt NPZ compare rtol (default: same as --plddt-rtol)",
    )
    p.add_argument(
        "--pae-atol",
        type=float,
        default=None,
        help="PAE/PDE/plddt NPZ compare atol (default: same as --plddt-atol)",
    )
    args = p.parse_args()
    pae_rtol = args.pae_rtol if args.pae_rtol is not None else args.plddt_rtol
    pae_atol = args.pae_atol if args.pae_atol is not None else args.plddt_atol

    boltz_dir = args.boltz_dir.resolve()
    boltr_dir = args.boltr_dir.resolve()
    if not boltz_dir.is_dir():
        print(f"ERROR: not a directory: {boltz_dir}", file=sys.stderr)
        return 1
    if not boltr_dir.is_dir():
        print(f"ERROR: not a directory: {boltr_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    print("Comparing outputs...")
    compare_confidence_json(boltz_dir, boltr_dir, errors)
    compare_structure_coords(
        boltz_dir,
        boltr_dir,
        args.coord_rtol,
        args.coord_atol,
        args.plddt_rtol,
        args.plddt_atol,
        errors,
    )
    compare_pae_pde(boltz_dir, boltr_dir, pae_rtol, pae_atol, errors)

    if errors:
        print(f"\nFAILURES ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll comparisons passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
