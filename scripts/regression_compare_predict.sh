#!/usr/bin/env bash
# Compare `boltz predict` vs `boltr predict` on the same preprocess output (TODO.md §7).
# Set BOLTR_REGRESSION=1 after both CLIs implement full writers + same manifest.

set -euo pipefail
if [[ "${BOLTR_REGRESSION:-}" != "1" ]]; then
  echo "regression_compare_predict: skipped (set BOLTR_REGRESSION=1 when parity is ready)."
  echo "See docs/TENSOR_CONTRACT.md and TODO.md §7."
  exit 0
fi
echo "regression_compare_predict: BOLTR_REGRESSION=1 but harness not wired yet."
exit 1
