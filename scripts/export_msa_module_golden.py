#!/usr/bin/env python3
"""Export `MSAModule` weights + I/O for Rust golden tests (`tests/msa_module_golden.rs`).

Requires: `torch`, `safetensors`, and Boltz on `PYTHONPATH`:

  PYTHONPATH=boltz-reference/src python scripts/export_msa_module_golden.py

Writes:
  boltr-backend-tch/tests/fixtures/msa_module_golden/msa_module_golden.safetensors

Tensor names prefixed `msa_module.` match `tch` VarStore keys under `Boltz2Model` root.
Names prefixed `golden.` are inputs / reference output (not loaded into VarStore).
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
        default=Path("boltr-backend-tch/tests/fixtures/msa_module_golden/msa_module_golden.safetensors"),
        help="Output safetensors path",
    )
    args = p.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Install safetensors: pip install safetensors", file=sys.stderr)
        return 1

    try:
        from boltz.model.modules.trunkv2 import MSAModule
    except ImportError as e:
        print(
            "Add boltz to PYTHONPATH, e.g. PYTHONPATH=boltz-reference/src",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    torch.manual_seed(0)
    token_s = 32
    token_z = 24
    msa_s = 16
    msa_blocks = 2
    b, s_msa, n = 1, 3, 4

    module = MSAModule(
        msa_s=msa_s,
        token_z=token_z,
        token_s=token_s,
        msa_blocks=msa_blocks,
        msa_dropout=0.0,
        z_dropout=0.0,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        use_paired_feature=True,
        subsample_msa=False,
    )
    module.eval()

    z = torch.randn(b, n, n, token_z, dtype=torch.float32)
    emb = torch.randn(b, n, token_s, dtype=torch.float32)
    feats = {
        "msa": torch.zeros(b, s_msa, n, dtype=torch.long),
        "msa_mask": torch.ones(b, s_msa, n, dtype=torch.float32),
        "has_deletion": torch.zeros(b, s_msa, n, dtype=torch.long),
        "deletion_value": torch.zeros(b, s_msa, n, dtype=torch.float32),
        "msa_paired": torch.zeros(b, s_msa, n, dtype=torch.long),
        "token_pad_mask": torch.ones(b, n, dtype=torch.float32),
    }

    with torch.no_grad():
        z_out = module(z, emb, feats, use_kernels=False)

    sd = {}
    prefix = "msa_module."
    for k, v in module.state_dict().items():
        t = v
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        sd[prefix + k] = t.contiguous()

    sd["golden.in_z"] = z
    sd["golden.in_s"] = emb
    sd["golden.msa"] = feats["msa"]
    sd["golden.msa_mask"] = feats["msa_mask"]
    sd["golden.has_deletion"] = feats["has_deletion"]
    sd["golden.deletion_value"] = feats["deletion_value"]
    sd["golden.msa_paired"] = feats["msa_paired"]
    sd["golden.token_pad_mask"] = feats["token_pad_mask"]
    sd["golden.z_out"] = z_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(args.out))
    print(f"Wrote {len(sd)} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
