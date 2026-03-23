#!/usr/bin/env python3
"""Export one `PairformerLayer` (or block) I/O for Rust golden tests.

**Preferred:** run the Rust generator when LibTorch is available:

  cargo run -p boltr-backend-tch --bin gen_pairformer_golden --features tch-backend

**Optional (full Python parity):** with `boltz` + `torch` installed, extend this script to:

1. Build a `PairformerLayer` from `boltz.model.layers.pairformer` with `use_kernels=False`.
2. Save fixed-seed inputs `s`, `z`, `mask`, `pair_mask` and outputs to
   `boltr-backend-tch/tests/fixtures/pairformer_golden/layer0.safetensors`.

Currently this file is a **documentation stub**; CI does not require PyTorch/Boltz.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    print(
        "Stub: implement boltz PairformerLayer export or use "
        "`cargo run -p boltr-backend-tch --bin gen_pairformer_golden --features tch-backend`.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
