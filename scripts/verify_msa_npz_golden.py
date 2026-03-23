#!/usr/bin/env python3
"""
Golden check: Boltz-style MSA .npz written by Rust matches NumPy savez_compressed.

Requires: numpy, cargo (Rust toolchain).

Fixture definition must stay in sync with:
  boltr-io/src/bin/msa_npz_golden.rs (golden_msa)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def boltz_msa_dtypes():
    import numpy as np

    MSAResidue = [("res_type", np.dtype("i1"))]
    MSADeletion = [("res_idx", np.dtype("i2")), ("deletion", np.dtype("i2"))]
    MSASequence = [
        ("seq_idx", np.dtype("i2")),
        ("taxonomy", np.dtype("i4")),
        ("res_start", np.dtype("i4")),
        ("res_end", np.dtype("i4")),
        ("del_start", np.dtype("i4")),
        ("del_end", np.dtype("i4")),
    ]
    return MSAResidue, MSADeletion, MSASequence


def golden_arrays():
    """Same logical MSA as `msa_npz_golden` Rust binary."""
    import numpy as np

    MSAResidue, MSADeletion, MSASequence = boltz_msa_dtypes()
    residues = np.array([2, 3, 4], dtype=MSAResidue)
    deletions = np.array([(1, 2)], dtype=MSADeletion)
    sequences = np.array([(0, 9606, 0, 3, 0, 1)], dtype=MSASequence)
    return sequences, deletions, residues


def run_cargo_msa_npz_golden(*args: str) -> None:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "boltr-io",
        "--bin",
        "msa_npz_golden",
        "--",
        *args,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    try:
        import numpy as np
    except ImportError:
        msg = "verify_msa_npz_golden: numpy is required (pip install numpy)"
        if os.environ.get("CI"):
            print(msg, file=sys.stderr)
            return 1
        print(f"{msg}; skipping outside CI", file=sys.stderr)
        return 0

    sequences, deletions, residues = golden_arrays()

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        python_npz = t / "from_python.npz"
        rust_npz = t / "from_rust.npz"

        np.savez_compressed(
            python_npz, sequences=sequences, deletions=deletions, residues=residues
        )
        run_cargo_msa_npz_golden("write", str(rust_npz))

        py = np.load(python_npz)
        rs = np.load(rust_npz)
        for key in ("sequences", "deletions", "residues"):
            np.testing.assert_equal(
                py[key], rs[key], err_msg=f"Rust vs NumPy mismatch for {key!r}"
            )

        # NumPy-written file must round-trip through Rust reader.
        run_cargo_msa_npz_golden("check", str(python_npz))

    print("verify_msa_npz_golden: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"cargo failed with exit {e.returncode}", file=sys.stderr)
        raise SystemExit(1)
    except AssertionError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
