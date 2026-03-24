#!/usr/bin/env python3
"""Export one Boltz2 `PairformerLayer` (v2 attention) weights + I/O for Rust golden tests.

Requires: `torch`, `safetensors`, Boltz on `PYTHONPATH`:

  PYTHONPATH=boltz-reference/src python scripts/export_pairformer_golden.py

Writes:
  boltr-backend-tch/tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors

Tensor names prefixed `layers.0.` match `PairformerLayer::new(..., path.sub("layers").sub("0"), ...)`.
Names prefixed `golden.*` are inputs / reference outputs (not loaded into VarStore).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(
            "boltr-backend-tch/tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors"
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
        from boltz.model.layers.pairformer import PairformerLayer
    except ImportError as e:
        print(
            "Add boltz to PYTHONPATH, e.g. PYTHONPATH=boltz-reference/src",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    import torch

    torch.manual_seed(0)
    # Small shapes; token_s divisible by num_heads (AttentionPairBiasV2).
    token_s = 32
    token_z = 24
    num_heads = 4
    pairwise_head_width = 32
    pairwise_num_heads = 4
    b, n = 1, 5

    layer = PairformerLayer(
        token_s=token_s,
        token_z=token_z,
        num_heads=num_heads,
        dropout=0.0,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        post_layer_norm=False,
        v2=True,
    )
    layer.eval()

    s = torch.randn(b, n, token_s, dtype=torch.float32)
    z = torch.randn(b, n, n, token_z, dtype=torch.float32)
    mask = torch.ones(b, n, n, dtype=torch.float32)
    pair_mask = torch.ones(b, n, n, dtype=torch.float32)

    with torch.no_grad():
        s_out, z_out = layer(
            s,
            z,
            mask,
            pair_mask,
            chunk_size_tri_attn=None,
            use_kernels=False,
        )

    sd: dict[str, torch.Tensor] = {}
    prefix = "layers.0."
    for k, v in layer.state_dict().items():
        t = v
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        sd[prefix + k] = t.contiguous()

    sd["golden.in_s"] = s
    sd["golden.in_z"] = z
    sd["golden.mask"] = mask
    sd["golden.pair_mask"] = pair_mask
    sd["golden.s_out"] = s_out
    sd["golden.z_out"] = z_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(args.out))
    print(f"Wrote {len(sd)} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
