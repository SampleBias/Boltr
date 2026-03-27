#!/usr/bin/env python3
"""Export `process_atom_features` tensors from Boltz for Rust golden tests.

Writes `atom_features_ala_golden.safetensors` (default) under
`boltr-io/tests/fixtures/collate_golden/`, matching inference-style
`Boltz2Featurizer.process` wiring: `process_ensemble_features` then
`process_atom_features` with `load_canonicals(mol_dir)`.

Requires:
  - A directory of CCD ``*.pkl`` files (canonical amino acids). Obtain by
    extracting Boltz ``mols.tar`` (see ``boltr-io`` download URLs) or
    ``boltr download`` cache assets.
  - ``PYTHONPATH=boltz-reference/src``
  - ``pip install torch safetensors numpy mashumaro rdkit tqdm einops dm-tree scipy numba``

Do **not** import ``boltz.data.module.inferencev2`` (pytorch_lightning).

Example:

    tar xf ~/.cache/boltr/mols.tar -C /path/to/mols_parent   # yields .../mols/*.pkl
    export PYTHONPATH=boltz-reference/src
    python3 scripts/dump_atom_features_golden.py --mol-dir /path/to/mols_parent/mols
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="StructureV2 .npz (default: collate_golden/ala_structure_v2.npz)",
    )
    p.add_argument(
        "--mol-dir",
        type=Path,
        default=None,
        help="Directory containing ALA.pkl, … (canonical). Or set BOLTZ_MOL_DIR.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .safetensors (default: collate_golden/atom_features_ala_golden.safetensors)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (matches inference smoke)")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    # Use NumPy-packed StructureV2 Python can load (`StructureV2.load`); must include PDB atom
    # names for `process_atom_features` (see `gen_structure_v2_numpy_golden.py`).
    npz_path = args.npz or (
        root / "boltr-io/tests/fixtures/structure_v2_numpy_packed_ala.npz"
    )
    out_path = args.out or (
        root / "boltr-io/tests/fixtures/collate_golden/atom_features_ala_golden.safetensors"
    )

    mol_dir = args.mol_dir or os.environ.get("BOLTZ_MOL_DIR")
    if not mol_dir:
        print(
            "Pass --mol-dir (directory with ALA.pkl etc.) or set BOLTZ_MOL_DIR.",
            file=sys.stderr,
        )
        return 1
    mol_dir = Path(mol_dir).resolve()
    if not (mol_dir / "ALA.pkl").is_file():
        print(f"Expected {mol_dir}/ALA.pkl — extract from Boltz mols.tar", file=sys.stderr)
        return 1

    boltz_on_path = any(
        Path(x).joinpath("boltz").exists()
        for x in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if x
    )
    if not boltz_on_path:
        sys.path.insert(0, str(root / "boltz-reference/src"))

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print("Install torch, safetensors, numpy:", e, file=sys.stderr)
        return 1

    try:
        from boltz.data.feature.featurizerv2 import (
            process_atom_features,
            process_ensemble_features,
        )
        from boltz.data.mol import load_canonicals
        from boltz.data.tokenize.boltz2 import tokenize_structure
        from boltz.data.types import StructureV2, Tokenized
    except ImportError as e:
        print(f"Boltz import failed: {e}", file=sys.stderr)
        return 1

    if not npz_path.is_file():
        print(f"Missing structure NPZ: {npz_path}", file=sys.stderr)
        return 1

    struct = StructureV2.load(npz_path)
    tokens, bonds = tokenize_structure(struct, None)
    tok = Tokenized(
        tokens=tokens,
        bonds=bonds,
        structure=struct,
        msa={},
        record=None,
        residue_constraints=None,
        templates=None,
        template_tokens=None,
        template_bonds=None,
        extra_mols=None,
    )
    rng = np.random.default_rng(args.seed)
    # Inference: single ensemble, no replacement
    ensemble_features = process_ensemble_features(
        data=tok,
        random=rng,
        num_ensembles=1,
        ensemble_sample_replacement=False,
        fix_single_ensemble=True,
    )
    molecules = load_canonicals(str(mol_dir))
    atom_feats = process_atom_features(
        data=tok,
        random=rng,
        ensemble_features=ensemble_features,
        molecules=molecules,
        atoms_per_window_queries=32,
        min_dist=2.0,
        max_dist=22.0,
        num_bins=64,
        max_atoms=None,
        max_tokens=None,
        disto_use_ensemble=False,
        override_bfactor=False,
        compute_frames=False,
        override_coords=None,
        bfactor_md_correction=False,
    )

    tensors = {k: v.detach().cpu().contiguous() for k, v in atom_feats.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"Wrote {out_path} ({len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
