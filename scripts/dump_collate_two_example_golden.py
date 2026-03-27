#!/usr/bin/env python3
"""Export a two-example collated batch matching Boltz `pad_to_max` / Rust `collate_inference_batches`.

Writes `collate_two_msa_golden.safetensors` under `boltr-io/tests/fixtures/collate_golden/`:
two `msa` tensors with different last dimensions, padded and stacked with batch axis 0.

Uses a NumPy reimplementation of [`boltz.data.pad.pad_to`](../../../boltz-reference/src/boltz/data/pad.py)
so this script does **not** require PyTorch.

Only `msa` is stored: Rust [`collate_inference_batches`](boltr-io/src/collate_pad.rs) pads i64 the same way
but does not emit a separate padding-mask tensor.

Requires:

    pip install safetensors numpy

Example:

    python3 scripts/dump_collate_two_example_golden.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _pad_np(data: np.ndarray, pad_tuple_torch_order: list[int], value: float) -> np.ndarray:
    """Mirror `torch.nn.functional.pad` layout (last dimension first in the flat tuple)."""
    nd = len(data.shape)
    assert len(pad_tuple_torch_order) == 2 * nd
    np_pad = []
    for ax in range(nd):
        lo = pad_tuple_torch_order[2 * (nd - 1 - ax)]
        hi = pad_tuple_torch_order[2 * (nd - 1 - ax) + 1]
        np_pad.append((lo, hi))
    return np.pad(data, np_pad, mode="constant", constant_values=value)


def pad_to_max_np(data: list[np.ndarray], value: float = 0.0) -> np.ndarray:
    """Match `boltz.data.pad.pad_to_max` for numeric tensors (same-shape fast path + pad + stack)."""
    if not data:
        raise ValueError("empty")
    if all(d.shape == data[0].shape for d in data):
        return np.stack(data, axis=0)

    num_dims = len(data[0].shape)
    max_dims = [max(d.shape[i] for d in data) for i in range(num_dims)]
    pad_lengths = []
    for d in data:
        dims: list[int] = []
        for i in range(num_dims):
            dims.append(0)
            dims.append(max_dims[num_dims - i - 1] - d.shape[num_dims - i - 1])
        pad_lengths.append(dims)

    padded = [_pad_np(d, pl, value) for d, pl in zip(data, pad_lengths)]
    return np.stack(padded, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: collate_golden/collate_two_msa_golden.safetensors)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = args.out or (
        root / "boltr-io/tests/fixtures/collate_golden/collate_two_msa_golden.safetensors"
    )

    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        print("Install safetensors numpy:", e, file=sys.stderr)
        return 1

    msa_a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    msa_b = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    stacked = pad_to_max_np([msa_a, msa_b], 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"msa": np.ascontiguousarray(stacked)}, str(out_path))
    print(f"Wrote {out_path} msa shape={stacked.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
