#!/usr/bin/env python3
"""Run upstream Boltz with GPU-friendly PyTorch defaults.

This wrapper is intentionally small: it preserves the normal `boltz` CLI
arguments while setting PyTorch knobs before Boltz imports and initializes.
"""

import os
import sys


def _configure_torch() -> None:
    try:
        import torch
    except Exception as exc:
        if os.getenv("BOLTR_BOLTZ_WRAPPER_LOG", "1").strip() != "0":
            print(f"[boltr-boltz-wrapper] torch import failed: {exc}", file=sys.stderr)
        return

    precision = os.getenv("BOLTR_BOLTZ_MATMUL_PRECISION", "high").strip().lower()
    if precision in {"highest", "high", "medium"}:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception as exc:
            print(
                f"[boltr-boltz-wrapper] could not set matmul precision {precision!r}: {exc}",
                file=sys.stderr,
            )

    threads = os.getenv("BOLTR_PREPROCESS_THREADS", "").strip()
    if threads:
        try:
            n = int(threads)
            if n > 0:
                torch.set_num_threads(n)
                torch.set_num_interop_threads(max(1, min(4, n)))
        except Exception:
            pass

    if os.getenv("BOLTR_BOLTZ_WRAPPER_LOG", "1").strip() != "0":
        get_precision = getattr(torch, "get_float32_matmul_precision", None)
        active_precision = get_precision() if callable(get_precision) else "unknown"
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "none"
        print(
            "[boltr-boltz-wrapper] "
            f"matmul_precision={active_precision} cuda_available={cuda} cuda_device={device}",
            file=sys.stderr,
        )


def main() -> int:
    _configure_torch()
    from boltz.main import cli

    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    return int(cli() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
