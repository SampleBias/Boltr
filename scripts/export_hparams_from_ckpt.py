#!/usr/bin/env python3
"""Export Lightning hyperparameters from a Boltz `.ckpt` to JSON for Rust `Boltz2Hparams`.

Requires: `pip install torch`

Example:
  python3 scripts/export_hparams_from_ckpt.py ~/.cache/boltz/boltz2_conf.ckpt \\
    boltr-backend-tch/tests/fixtures/hparams/boltz2_hparams.sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    try:
        import torch
    except ImportError:
        print("Install torch: pip install torch", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out_json", type=Path)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("Unexpected checkpoint root", file=sys.stderr)
        return 1

    hparams = ckpt.get("hyper_parameters")
    if hparams is None:
        hparams = {}

    def default(o):
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                return str(o)
        return str(o)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(hparams, f, indent=2, default=default, sort_keys=True)
    print(f"Wrote {args.out_json} ({len(hparams)} top-level keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
