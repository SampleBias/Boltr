#!/usr/bin/env python3
"""Export `process_msa_features` tensors from Boltz for Rust golden tests (stub).

Requires full Boltz env: `PYTHONPATH=boltz-reference/src`, `torch`, `safetensors`.
See `boltr-io` tests: `msa_features_from_inference_input` for the Rust path.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="msa_features_golden.safetensors")
    args = p.parse_args()
    try:
        import torch  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError:
        print("Install torch + safetensors; set PYTHONPATH to boltz-reference/src.", file=sys.stderr)
        return 1
    print("Stub: wire Boltz2Tokenizer + process_msa_features, then save_file.", file=sys.stderr)
    print(f"Would write: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
