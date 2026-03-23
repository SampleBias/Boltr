#!/usr/bin/env python3
"""Build a Boltz-style StructureV2 `.npz` with NumPy packed structured dtypes.

This mirrors `src/boltz/data/types.py` from boltz (jwohlwend/boltz main): `AtomV2`,
`Residue`, `Chain`, `BondV2`, `Coords`, `Ensemble`, `Interface`. NumPy uses **packed**
layouts by default (e.g. AtomV2 itemsize 37), matching typical `np.array(..., dtype=...)`
output from preprocessing — not necessarily `align=True`.

Regenerate the checked-in fixture:

    python3 -m pip install numpy
    python3 scripts/gen_structure_v2_numpy_golden.py

Output: `boltr-io/tests/fixtures/structure_v2_numpy_packed_ala.npz`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "boltr-io/tests/fixtures/structure_v2_numpy_packed_ala.npz"

# --- Same field lists as boltz `data/types.py` (StructureV2 components) ---

AtomV2 = [
    ("name", np.dtype("<U4")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("bfactor", np.dtype("f4")),
    ("plddt", np.dtype("f4")),
]

BondV2 = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Coords = [("coords", np.dtype("3f4"))]

Ensemble = [
    ("atom_coord_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]


def main() -> None:
    assert np.dtype(AtomV2).itemsize == 37, "packed AtomV2"
    assert np.dtype(Residue).itemsize == 43, "packed Residue"
    assert np.dtype(Chain).itemsize == 53, "packed Chain"
    assert np.dtype(BondV2).itemsize == 25, "packed BondV2"
    assert np.dtype(Coords).itemsize == 12, "Coords row"
    assert np.dtype(Ensemble).itemsize == 8, "Ensemble row"
    assert np.dtype(Interface).itemsize == 8, "Interface row"

    # Same geometry as `boltr_io::fixtures::structure_v2_single_ala`.
    coords_data = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ]

    atoms = np.zeros(5, dtype=AtomV2)
    for i, c in enumerate(coords_data):
        atoms[i]["coords"] = np.array(c, dtype=np.float32)
        atoms[i]["is_present"] = True
        atoms[i]["bfactor"] = 0.0
        atoms[i]["plddt"] = 0.0

    residues = np.zeros(1, dtype=Residue)
    residues[0]["name"] = "ALA"
    residues[0]["res_type"] = 2  # token_ids["ALA"] in boltz / boltr boltz_const
    residues[0]["res_idx"] = 0
    residues[0]["atom_idx"] = 0
    residues[0]["atom_num"] = 5
    residues[0]["atom_center"] = 1
    residues[0]["atom_disto"] = 4
    residues[0]["is_standard"] = True
    residues[0]["is_present"] = True

    chains = np.zeros(1, dtype=Chain)
    chains[0]["mol_type"] = 0  # PROTEIN
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = 5
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 1
    chains[0]["cyclic_period"] = 0

    bonds = np.zeros(0, dtype=BondV2)
    # Non-empty `interfaces` array (Rust reader ignores it today; ensures zip layout).
    interfaces = np.array([(0, 0)], dtype=Interface)
    mask = np.array([True], dtype=np.bool_)
    coords = np.zeros(5, dtype=Coords)
    for i, c in enumerate(coords_data):
        coords[i]["coords"] = np.array(c, dtype=np.float32)

    # Multiple ensemble rows: tokenizer / Rust keep `ensemble[0]` only.
    ensemble = np.array([(0, 5), (0, 5), (0, 5)], dtype=Ensemble)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
