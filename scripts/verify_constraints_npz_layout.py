#!/usr/bin/env python3
"""Validate a Boltz `ResidueConstraints` `.npz` written by `ResidueConstraints.dump()` (see
`boltz.data.types.ResidueConstraints`: `np.savez_compressed(..., **asdict(self))`).

Expected top-level keys (6 structured arrays) match upstream Boltz2 `types.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Mirrors `boltz-reference/src/boltz/data/types.py` (ResidueConstraints + sub-dtypes).
DT_RDKIT = np.dtype(
    [
        ("atom_idxs", np.dtype("2i4")),
        ("is_bond", np.dtype("?")),
        ("is_angle", np.dtype("?")),
        ("upper_bound", np.dtype("f4")),
        ("lower_bound", np.dtype("f4")),
    ]
)
DT_CHIRAL = np.dtype(
    [
        ("atom_idxs", np.dtype("4i4")),
        ("is_reference", np.dtype("?")),
        ("is_r", np.dtype("?")),
    ]
)
DT_STEREO = np.dtype(
    [
        ("atom_idxs", np.dtype("4i4")),
        ("is_reference", np.dtype("?")),
        ("is_e", np.dtype("?")),
    ]
)
DT_PLANAR_BOND = np.dtype([("atom_idxs", np.dtype("6i4"))])
DT_PLANAR_R5 = np.dtype([("atom_idxs", np.dtype("5i4"))])
DT_PLANAR_R6 = np.dtype([("atom_idxs", np.dtype("6i4"))])

EXPECTED = {
    "rdkit_bounds_constraints": DT_RDKIT,
    "chiral_atom_constraints": DT_CHIRAL,
    "stereo_bond_constraints": DT_STEREO,
    "planar_bond_constraints": DT_PLANAR_BOND,
    "planar_ring_5_constraints": DT_PLANAR_R5,
    "planar_ring_6_constraints": DT_PLANAR_R6,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="Path to constraints .npz")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    keys = set(data.files)
    missing = set(EXPECTED) - keys
    extra = keys - set(EXPECTED)
    if missing or extra:
        print(f"Bad keys in {args.npz}", file=sys.stderr)
        if missing:
            print(f"  missing: {sorted(missing)}", file=sys.stderr)
        if extra:
            print(f"  unexpected: {sorted(extra)}", file=sys.stderr)
        return 1

    ok = True
    for name, want_dt in EXPECTED.items():
        arr = data[name]
        if arr.dtype != want_dt:
            print(
                f"{name}: dtype mismatch\n  got:  {arr.dtype}\n  want: {want_dt}",
                file=sys.stderr,
            )
            ok = False
        if arr.ndim != 1:
            print(f"{name}: expected 1-D structured array, got shape {arr.shape}", file=sys.stderr)
            ok = False

    if not ok:
        return 1

    print(f"OK: {args.npz} has valid ResidueConstraints layout ({len(EXPECTED)} keys).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
