#!/usr/bin/env python3
"""Export `rel_pos`, `s_init`, `z_init_*`, `token_bonds` tensors for Rust golden tests.

Requires: `torch`, `safetensors`, Boltz on `PYTHONPATH`:

  PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py

Writes:
  boltr-backend-tch/tests/fixtures/trunk_init_golden/trunk_init_golden.safetensors

`golden.*` keys are inputs / reference outputs. Module weights use the same names as Lightning `state_dict`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(
            "boltr-backend-tch/tests/fixtures/trunk_init_golden/trunk_init_golden.safetensors"
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
        from boltz.model.modules.encodersv2 import RelativePositionEncoder
    except ImportError as e:
        print(
            "Add boltz to PYTHONPATH, e.g. PYTHONPATH=boltz-reference/src",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    torch.manual_seed(0)
    b, n = 2, 7
    token_s = 32
    token_z = 24

    rel = RelativePositionEncoder(
        token_z, r_max=32, s_max=2, fix_sym_check=False, cyclic_pos_enc=False
    )
    s_init = nn.Linear(token_s, token_s, bias=False)
    z_init_1 = nn.Linear(token_s, token_z, bias=False)
    z_init_2 = nn.Linear(token_s, token_z, bias=False)
    token_bonds = nn.Linear(1, token_z, bias=False)

    asym_id = torch.zeros(b, n, dtype=torch.long)
    residue_index = torch.arange(n, dtype=torch.long).view(1, n).expand(b, n).clone()
    entity_id = torch.zeros(b, n, dtype=torch.long)
    token_index = residue_index.clone()
    sym_id = torch.zeros(b, n, dtype=torch.long)
    cyclic_period = torch.zeros(b, n, dtype=torch.long)

    feats_rel = {
        "asym_id": asym_id,
        "residue_index": residue_index,
        "entity_id": entity_id,
        "token_index": token_index,
        "sym_id": sym_id,
        "cyclic_period": cyclic_period,
    }
    rel_out = rel(feats_rel)

    s_in = torch.randn(b, n, token_s, dtype=torch.float32)
    s_init_out = s_init(s_in)
    z_pair = z_init_1(s_in)[:, :, None] + z_init_2(s_in)[:, None, :]

    token_bonds_in = torch.randn(b, n, n, 1, dtype=torch.float32)
    z_bonds = token_bonds(token_bonds_in)

    tensors: dict[str, torch.Tensor] = {
        "rel_pos.linear_layer.weight": rel.linear_layer.weight.detach().float(),
        "s_init.weight": s_init.weight.detach().float(),
        "z_init_1.weight": z_init_1.weight.detach().float(),
        "z_init_2.weight": z_init_2.weight.detach().float(),
        "token_bonds.weight": token_bonds.weight.detach().float(),
        "golden.rel_pos_out": rel_out.float(),
        "golden.s_in": s_in,
        "golden.s_init_out": s_init_out.float(),
        "golden.z_pair": z_pair.float(),
        "golden.token_bonds_in": token_bonds_in,
        "golden.z_bonds": z_bonds.float(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
