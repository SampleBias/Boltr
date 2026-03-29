#!/usr/bin/env python3
"""Export full trunk `InputEmbedder` forward golden (tiny shapes).

Requires: `torch`, `safetensors`, Boltz on `PYTHONPATH`:

  PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py

Writes:
  boltr-backend-tch/tests/fixtures/input_embedder_golden/input_embedder_golden.safetensors

Keys `input_embedder.*` match Rust `VarStore` under `input_embedder/`; `golden.s_inputs` is the reference output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(
            "boltr-backend-tch/tests/fixtures/input_embedder_golden/input_embedder_golden.safetensors"
        ),
        help="Output safetensors path",
    )
    args = p.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Install safetensors: pip install safetensors", file=sys.stderr)
        return 1

    try:
        from boltz.model.modules.trunkv2 import InputEmbedder
    except ImportError as e:
        print(
            "Add boltz to PYTHONPATH, e.g. PYTHONPATH=boltz-reference/src",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    torch.manual_seed(2)
    b, n_tok = 1, 4
    n_atoms = 32
    atom_s, atom_z = 16, 8
    token_s, token_z = 32, 16
    wq, wk = 32, 128
    atom_feature_dim = 8
    depth, heads = 2, 2

    m = InputEmbedder(
        atom_s=atom_s,
        atom_z=atom_z,
        token_s=token_s,
        token_z=token_z,
        atoms_per_window_queries=wq,
        atoms_per_window_keys=wk,
        atom_feature_dim=atom_feature_dim,
        atom_encoder_depth=depth,
        atom_encoder_heads=heads,
        activation_checkpointing=False,
        use_no_atom_char=True,
        use_atom_backbone_feat=False,
        use_residue_feats_atoms=False,
    )
    m.eval()

    atom_to_token = torch.zeros(b, n_atoms, n_tok)
    for t in range(n_tok):
        atom_to_token[:, t * 8 : (t + 1) * 8, t] = 1.0

    feats = {
        "ref_pos": torch.randn(b, n_atoms, 3),
        "ref_charge": torch.randn(b, n_atoms),
        "ref_element": torch.randn(b, n_atoms, 4),
        "atom_pad_mask": torch.ones(b, n_atoms),
        "ref_space_uid": torch.zeros(b, n_atoms, dtype=torch.long),
        "atom_to_token": atom_to_token,
        "res_type": torch.randn(b, n_tok, 33),
        "profile": torch.randn(b, n_tok, 33),
        "deletion_mean": torch.randn(b, n_tok),
    }
    with torch.no_grad():
        s_inputs = m(feats)

    sd = m.state_dict()
    tensors: dict[str, torch.Tensor] = {
        f"input_embedder.{k}": v.detach().float() for k, v in sd.items()
    }
    tensors["golden.s_inputs"] = s_inputs.float()
    for name, t in feats.items():
        key = f"golden.in_{name}"
        if t.dtype in (torch.int32, torch.int64):
            tensors[key] = t.to(torch.int64)
        else:
            tensors[key] = t.detach().float()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
