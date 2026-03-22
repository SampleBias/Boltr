#!/usr/bin/env python3
"""Export a Lightning Boltz `.ckpt` state dict to Safetensors for Rust (`boltr-backend-tch`) loading.

Requires: `pip install torch safetensors`

Example:
  python scripts/export_checkpoint_to_safetensors.py \\
    ~/.cache/boltz/boltz2_conf.ckpt \\
    ~/.cache/boltr/boltz2_conf.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path, help="Path to boltz*.ckpt")
    p.add_argument("out", type=Path, help="Output .safetensors path")
    p.add_argument(
        "--strip-prefix",
        default="",
        help="Prefix to remove from keys (e.g. 'model.')",
    )
    args = p.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Install safetensors: pip install safetensors", file=sys.stderr)
        return 1

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        print("Unexpected checkpoint format", file=sys.stderr)
        return 1

    prefix = args.strip_prefix
    out_sd = {}
    for k, v in sd.items():
        if prefix and k.startswith(prefix):
            k = k[len(prefix) :]
        if hasattr(v, "dtype") and v.dtype in (torch.bfloat16, torch.float16):
            v = v.float()
        out_sd[k] = v

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(args.out))
    print(f"Wrote {len(out_sd)} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
