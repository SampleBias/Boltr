#!/usr/bin/env python3
"""Export `process_msa_features` tensors from Boltz for Rust golden tests.

Writes `msa_features_load_input_smoke_golden.safetensors` next to the
`load_input_smoke` fixture (same tensors as Boltz inference: seed 42,
`max_seqs` = `const.max_msa_seqs`, `pad_to_max_seqs=False`, `msa_sampling=False`).

Requires a Boltz tree on ``PYTHONPATH`` (this repo: ``boltz-reference/src``) and:

    pip install torch safetensors numpy mashumaro rdkit tqdm einops dm-tree scipy numba

Do **not** import ``boltz.data.module.inferencev2`` (it pulls pytorch_lightning at import time).
Use ``Manifest`` + ``StructureV2`` + ``MSA`` + ``Input`` + ``Boltz2Tokenizer`` like
``scripts/dump_token_features_ala_golden.py``.

Example:

    cd /path/to/Boltr
    export PYTHONPATH=boltz-reference/src
    python3 scripts/dump_msa_features_golden.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="Directory with manifest.json, test_ala.npz, 0.npz (default: boltr-io load_input_smoke)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .safetensors path (default: <fixture-dir>/msa_features_load_input_smoke_golden.safetensors)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    fixture_dir = args.fixture_dir or (
        root / "boltr-io/tests/fixtures/load_input_smoke"
    )
    out_path = args.out or (
        fixture_dir / "msa_features_load_input_smoke_golden.safetensors"
    )

    boltz_on_path = any(
        Path(x).joinpath("boltz").exists() for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x
    )
    if not boltz_on_path:
        sys.path.insert(0, str(root / "boltz-reference/src"))

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(
            "Install torch, safetensors, numpy, and Boltz deps (see docstring).",
            e,
            file=sys.stderr,
        )
        return 1

    try:
        from boltz.data import const
        from boltz.data.feature.featurizerv2 import process_msa_features
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
        from boltz.data.types import Input, Manifest, MSA, StructureV2
    except ImportError as e:
        print(
            f"Failed to import Boltz (set PYTHONPATH=boltz-reference/src): {e}",
            file=sys.stderr,
        )
        return 1

    manifest = Manifest.load(fixture_dir / "manifest.json")
    record = manifest.records[0]
    structure = StructureV2.load(fixture_dir / f"{record.id}.npz")
    msas: dict = {}
    for chain in record.chains:
        mid = chain.msa_id
        if mid == -1:
            continue
        msas[chain.chain_id] = MSA.load(fixture_dir / f"{mid}.npz")

    input_data = Input(
        structure,
        msas,
        record=record,
        residue_constraints=None,
        templates=None,
        extra_mols=None,
    )
    tokenized = Boltz2Tokenizer().tokenize(input_data)
    rng = np.random.default_rng(42)
    feats = process_msa_features(
        tokenized,
        rng,
        max_seqs_batch=const.max_msa_seqs,
        max_seqs=const.max_msa_seqs,
        max_tokens=None,
        pad_to_max_seqs=False,
        msa_sampling=False,
    )

    tensors = {k: v.detach().cpu().contiguous() for k, v in feats.items()}
    save_file(tensors, str(out_path))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
