#!/usr/bin/env python3
"""Write minimal collated-feature tensors for Boltr trunk / featurizer contract tests.

Produces `boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors` with
deterministic (seeded) values and shapes consistent with `manifest.json` in that directory.

Requires: `pip install torch safetensors`

See `boltr-io/tests/fixtures/collate_golden/README.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("Install: pip install torch safetensors", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "boltr-io/tests/fixtures/collate_golden"
    out_path = out_dir / "trunk_smoke_collate.safetensors"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    b, n, s_msa = 1, 4, 8
    num_tokens = 33
    token_s = 384
    tdim = 1

    s_inputs = torch.randn(b, n, token_s, dtype=torch.float32)
    token_pad_mask = torch.ones(b, n, dtype=torch.float32)
    msa = torch.zeros(b, s_msa, n, dtype=torch.int64)
    msa_paired = torch.zeros(b, s_msa, n, dtype=torch.int64)
    msa_mask = torch.ones(b, s_msa, n, dtype=torch.int64)
    has_deletion = torch.zeros(b, s_msa, n, dtype=torch.int64)
    deletion_value = torch.zeros(b, s_msa, n, dtype=torch.float32)
    deletion_mean = torch.zeros(b, n, dtype=torch.float32)
    profile = torch.randn(b, n, num_tokens, dtype=torch.float32)
    template_restype = torch.zeros(b, tdim, n, num_tokens, dtype=torch.float32)
    template_mask = torch.zeros(b, tdim, n, dtype=torch.float32)

    tensors = {
        "s_inputs": s_inputs,
        "token_pad_mask": token_pad_mask,
        "msa": msa,
        "msa_paired": msa_paired,
        "msa_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "template_restype": template_restype,
        "template_mask": template_mask,
    }
    save_file(tensors, str(out_path))
    print(f"Wrote {out_path} ({len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
