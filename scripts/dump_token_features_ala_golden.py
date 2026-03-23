#!/usr/bin/env python3
"""Regenerate ALA `process_token_features` golden safetensors (authoritative: upstream Boltz).

Writes (next to this repo's collate fixtures):
  - boltr-io/tests/fixtures/collate_golden/token_features_ala_golden.safetensors
  - boltr-io/tests/fixtures/collate_golden/token_features_ala_collated_golden.safetensors

Requires a **full** Boltz checkout (with `boltz.data`), not the model-only `boltz-reference` tree:

  export BOLTZ_SRC=/path/to/jwohlwend/boltz
  pip install torch safetensors numpy

  python3 scripts/dump_token_features_ala_golden.py

The structure NPZ must match Rust [`structure_v2_single_ala`]. Generate it with:

  cargo run -p boltr-io --bin write_token_features_ala_golden

(that also refreshes `ala_structure_v2.npz` in the same directory).

If Boltz is unavailable, rely on the Rust binary for fixture bytes; CI compares Rust output to the
checked-in safetensors.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _prepend_batch(tensors: dict) -> dict:
    import torch

    out = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).contiguous()
        else:
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boltz-src",
        type=Path,
        default=None,
        help="Root of full Boltz repo (contains src/boltz). Default: $BOLTZ_SRC",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to ala_structure_v2.npz (default: under boltr-io collate_golden)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    default_npz = (
        root
        / "boltr-io/tests/fixtures/collate_golden/ala_structure_v2.npz"
    )
    npz_path = args.npz or default_npz
    if not npz_path.is_file():
        print(f"Missing structure NPZ: {npz_path}\nRun: cargo run -p boltr-io --bin write_token_features_ala_golden", file=sys.stderr)
        return 1

    boltz_src = args.boltz_src or os.environ.get("BOLTZ_SRC")
    if not boltz_src:
        print(
            "Set BOLTZ_SRC to a full Boltz checkout (e.g. clone jwohlwend/boltz) or pass --boltz-src.",
            file=sys.stderr,
        )
        return 1
    boltz_src = Path(boltz_src).resolve()
    sys.path.insert(0, str(boltz_src / "src"))

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Install dependencies: pip install torch safetensors numpy ({e})", file=sys.stderr)
        return 1

    try:
        from boltz.data.feature.featurizerv2 import process_token_features
        from boltz.data.tokenize.boltz2 import tokenize_structure
        from boltz.data.types import StructureV2, Tokenized
    except ImportError as e:
        print(
            f"Failed to import Boltz from {boltz_src}/src: {e}",
            file=sys.stderr,
        )
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
    rng = np.random.default_rng(0)
    feats = process_token_features(
        tok,
        rng,
        max_tokens=None,
        binder_pocket_conditioned_prop=0.0,
        contact_conditioned_prop=0.0,
        inference_pocket_constraints=None,
        inference_contact_constraints=None,
    )

    out_dir = root / "boltr-io/tests/fixtures/collate_golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    per = {k: v.detach().cpu().contiguous() for k, v in feats.items()}
    per_path = out_dir / "token_features_ala_golden.safetensors"
    save_file(per, str(per_path))
    print(f"Wrote {per_path} ({len(per)} tensors)")

    col = _prepend_batch(per)
    col_path = out_dir / "token_features_ala_collated_golden.safetensors"
    save_file(col, str(col_path))
    print(f"Wrote {col_path} ({len(col)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
