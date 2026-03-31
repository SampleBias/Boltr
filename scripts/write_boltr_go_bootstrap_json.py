#!/usr/bin/env python3
"""Write boltr_go_bootstrap.json under the model cache (read by boltr-web /api/status)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, help="BOLTZ_CACHE (e.g. ~/.cache/boltr)")
    ap.add_argument("--repo-root", default="", help="Boltr repo root")
    ap.add_argument(
        "--download-ok",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 if boltr download / bootstrap cache step completed",
    )
    ap.add_argument(
        "--verify-ok",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if verify_boltz2_safetensors succeeded",
    )
    ap.add_argument(
        "--verify-skipped",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if verify was skipped (BOLTR_GO_SKIP_VERIFY)",
    )
    ap.add_argument(
        "--tests-ok",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if test suite(s) passed",
    )
    ap.add_argument(
        "--tests-skipped",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if tests were skipped (BOLTR_GO_SKIP_TESTS)",
    )
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {
        "version": 1,
        "completed_at": completed_at,
        "cache_dir": str(cache.resolve()),
        "repo_root": args.repo_root or None,
        "download_ok": bool(args.download_ok),
        "verify_safetensors_ok": None
        if args.verify_skipped
        else bool(args.verify_ok),
        "verify_skipped": bool(args.verify_skipped),
        "tests_ok": None if args.tests_skipped else bool(args.tests_ok),
        "tests_skipped": bool(args.tests_skipped),
    }
    out = cache / "boltr_go_bootstrap.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
