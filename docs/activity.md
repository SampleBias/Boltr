# Boltr — Activity Log

Activity log for the Boltr project. See [docs/PROJECT_README.md](docs/PROJECT_README.md) for project context.

---
*Activity logging format:*
*## YYYY-MM-DD HH:MM - Action Description*
*- Detailed description of what was done*
*- Files created/modified*
*- Commands executed*
*- Any important notes or decisions*

## 2026-03-28 16:15 - Project Initialization
- Created project structure files (tasks/todo.md, docs/activity.md, docs/PROJECT_README.md)
- Initialized todo.md with project template

## 2026-03-29 §2.8 — Comprehensive YAML Parsing Tests
- Audited every item in tasks/todo.md — found 2.1–2.7 already implemented in config.rs
- Created 20 YAML fixture files under boltr-io/tests/fixtures/yaml/
- Wrote 45 comprehensive tests covering all schema paths
- Fixed `PropertyEntry` missing `#[serde(untagged)]` bug in config.rs
- All 185 tests passing, 0 regressions

## 2026-03-29 §7 — Testing Strategy Assessment & Implementation

Assessed all 5 sub-items of TODO.md §7. Current status:

| Item | Status | Assessment |
|------|--------|-----------|
| Golden fixture layout | `[~]` | 3 missing READMEs → created |
| Python export scripts | `[~]` | 17 scripts exist, `dump_full_collate_golden.py` is functional |
| Numerical tolerances | `[~]` | §6.5 table is vague → expanded with exact values from codebase |
| Regression harness | `[~]` | Script was corrupted → rewrote as a proper 350-line harness |
| Backend unit tests | `[~]` | 6 opt-in golden tests + 4 shape-only smokes exist |

Completed:
1. **Regression harness rewrite** (`scripts/regression_compare_predict.sh`):
   - Replaced corrupted placeholder with 350-line syntactically valid bash script
   - Gated behind `BOLTR_REGRESSION=1` (exit 0 when unset)
   - 3-step flow: run `boltz predict` → run `boltr predict` → compare outputs
   - Inline Python comparison script for NPZ/JSON output comparison
   - Configurable tolerances via environment variables
   - Proper error reporting with per-key pass/fail status
   - Documented prerequisites, usage, and environment in header

2. **Fixture READMEs created** (3 directories that were missing them):
   - `boltr-io/tests/fixtures/README.md` — top-level fixture index
   - `boltr-io/tests/fixtures/load_input_smoke/README.md` — smoke fixture doc
   - `boltr-io/tests/fixtures/yaml/README.md` — YAML fixture coverage table

3. **Backend fixture index** (`boltr-backend-tch/tests/fixtures/README.md`):
   - Documents all 6 golden fixture directories with regeneration commands
   - Notes which `.safetensors` are checked in vs generated on demand
   - Links to opt-in env vars and test files

4. **`docs/TESTING_STRATEGY.md`** — comprehensive testing strategy document:
   - Per-test coverage map (test name, fixture, tolerance, module, opt-in)
   - Numerical tolerance reference table (rtol/atol per category)
   - Fixture regeneration guide (every script with deps)
   - CI coverage summary
   - Adding new golden tests checklist

5. **Numerical tolerances audit**:
   - Catalogued every tolerance value across codebase
   - Deterministic features: rtol=1e-5, atol=1e-6
   - Neural network forward passes: rtol=1e-4, atol=1e-5
   - Exact/structural: no tolerance needed
   - Sampling/stochastic: looser tolerances
