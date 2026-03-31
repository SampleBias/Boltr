#!/usr/bin/env python3
"""Export the full Lightning hyper_parameters dict from a Boltz .ckpt to JSON.

Deserializes into boltr-backend-tch Boltz2Hparams (unknown top-level keys go to serde flatten other).

Requires: `pip install torch`. Checkpoints that store **`omegaconf.DictConfig`** need **`pip install omegaconf`** (Boltz does); we convert those with `OmegaConf.to_container` before JSON.

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
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(o):
                return default(OmegaConf.to_container(o, resolve=True))
        except ImportError:
            pass
        if isinstance(o, dict):
            return {k: default(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [default(x) for x in o]
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        from collections.abc import Mapping

        if isinstance(o, Mapping) and not isinstance(o, (str, bytes)):
            try:
                return {str(k): default(v) for k, v in o.items()}
            except RecursionError:
                return str(o)
        return str(o)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(hparams, f, indent=2, default=default, sort_keys=True)
    print(f"Wrote {args.out_json} ({len(hparams)} top-level keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
