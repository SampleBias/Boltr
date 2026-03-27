#!/usr/bin/env python3
"""Compare tensor entry counts between a Lightning `.ckpt` state dict and a `.safetensors` export.

Useful after `scripts/export_checkpoint_to_safetensors.py` with `--strip-prefix model.` to ensure
no keys were dropped silently. Mismatches print a short diff of key names.

Requires: `pip install torch safetensors`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError("Unexpected checkpoint format (expected dict)")
    return ckpt


def export_style_keys(sd: dict, strip_prefix: str) -> set[str]:
    """Match `export_checkpoint_to_safetensors.py`: strip prefix only when key starts with it."""
    out: set[str] = set()
    for k, v in sd.items():
        if not (hasattr(v, "shape") and hasattr(v, "dtype")):
            continue
        if strip_prefix and k.startswith(strip_prefix):
            out.add(k[len(strip_prefix) :])
        else:
            out.add(k)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path, help="Lightning .ckpt path")
    p.add_argument("safetensors", type=Path, help="Exported .safetensors path")
    p.add_argument(
        "--strip-prefix",
        default="",
        dest="strip_prefix",
        help="Prefix removed from ckpt keys when present (e.g. 'model.') — same as export script",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print keys only in ckpt or only in safetensors",
    )
    args = p.parse_args()

    try:
        from safetensors import safe_open
    except ImportError:
        print("Install safetensors: pip install safetensors", file=sys.stderr)
        return 1

    sd = load_state_dict(args.ckpt)
    ckpt_keys = export_style_keys(sd, args.strip_prefix)

    with safe_open(args.safetensors, framework="pt") as f:
        sf_keys = set(f.keys())

    n_ckpt = len(ckpt_keys)
    n_sf = len(sf_keys)
    print(f"ckpt tensor keys (after strip): {n_ckpt}")
    print(f"safetensors tensor keys: {n_sf}")

    if ckpt_keys == sf_keys:
        print("OK: key sets match.")
        return 0

    only_ckpt = sorted(ckpt_keys - sf_keys)
    only_sf = sorted(sf_keys - ckpt_keys)
    print("MISMATCH: key sets differ.", file=sys.stderr)
    print(f"  only in ckpt ({len(only_ckpt)}):", file=sys.stderr)
    if args.verbose:
        for k in only_ckpt[:200]:
            print(f"    {k}", file=sys.stderr)
        if len(only_ckpt) > 200:
            print(f"    ... and {len(only_ckpt) - 200} more", file=sys.stderr)
    print(f"  only in safetensors ({len(only_sf)}):", file=sys.stderr)
    if args.verbose:
        for k in only_sf[:200]:
            print(f"    {k}", file=sys.stderr)
        if len(only_sf) > 200:
            print(f"    ... and {len(only_sf) - 200} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
