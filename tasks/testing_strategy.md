# §7 — Testing Strategy (Cross-cutting)

## Context
From TODO.md §7: Expand golden fixture coverage, document tolerances, document regeneration, wire CI, and create a regression harness placeholder.

## Assessment (current state)

### What exists
| Area | Status |
|------|--------|
| **boltr-io lib tests** | 120 passing — token, atom, MSA, template, symmetry, constraint, NPZ roundtrips |
| **boltr-io integration tests** | 59 passing (45 YAML + 3 collate_golden + 1 full_collate + 3 smoke + 6 load_input + 1 post_collate) |
| **boltr-backend-tch tests** | 6 smoke + 5 opt-in goldens (msa, pairformer, trunk_init, input_embedder, template_diffusion) |
| **Fixture directories** | 4 directories: `collate_golden/`, `load_input_smoke/`, `yaml/`, `structure_v2_numpy_packed_ala.npz` |
| **Python export scripts** | 17 scripts in `scripts/` |
| **CI workflows** | 2: `libtorch-backend-smoke.yml` (workflow_dispatch), `msa-npz-golden.yml` (push/PR) |
| **Regression harness** | Placeholder script (`scripts/regression_compare_predict.sh`) — exits 1 |
| **Tolerances** | Inconsistent: atom_features `rtol=1e-4 atol=1e-5`, token_features `rtol=1e-5 atol=1e-6`, backend `rtol=1e-4 atol=1e-5` |

### What's missing (the gaps)
| Gap | Impact |
|-----|--------|
| No `boltr-io/tests/fixtures/README.md` — top-level fixture index | Developers don't know what fixtures exist or how to regenerate |
| No `boltr-backend-tch/tests/fixtures/README.md` — backend fixture index | Same |
| No `docs/NUMERICAL_TOLERANCES.md` — central tolerance registry | Tolerances scattered across files, inconsistent |
| `load_input_smoke/` has no README | No docs on fixture provenance or regeneration |
| `collate_golden/README.md` exists but is outdated | Doesn't mention all current golden files |
| CI only runs `--lib` for backend | Integration tests (collate_predict_trunk etc.) not in CI |
| No `cargo test` full-suite CI workflow (boltr-io only tested via path-trigger) | |
| Regression harness is a stub | |

## Subtasks

### 7.1 Fixture layout — top-level index README
- [ ] 7.1.1 Write `boltr-io/tests/fixtures/README.md` — index every subdirectory, files, purpose, regeneration commands
- [ ] 7.1.2 Write `boltr-io/tests/fixtures/load_input_smoke/README.md`
- [ ] 7.1.3 Update `boltr-io/tests/fixtures/collate_golden/README.md` to be current
- [ ] 7.1.4 Write `boltr-backend-tch/tests/fixtures/README.md` — index all backend fixture subdirectories

### 7.2 Numerical tolerances document
- [ ] 7.2.1 Write `docs/NUMERICAL_TOLERANCES.md` — central registry of rtol/atol per test category
- [ ] 7.2.2 Audit and normalize tolerance constants across test files

### 7.3 Python export scripts — documentation
- [ ] 7.3.1 Write `scripts/README.md` — index every script, prerequisites, usage

### 7.4 Regression harness
- [ ] 7.4.1 Replace placeholder `scripts/regression_compare_predict.sh` with a documented stub that explains prerequisites and exit codes

### 7.5 CI coverage
- [ ] 7.5.1 Add `cargo-test-boltr-io.yml` workflow (push/PR on boltr-io paths)
- [ ] 7.5.2 Update `libtorch-backend-smoke.yml` to also run integration tests

### 7.6 Update TODO.md §7 status
- [ ] 7.6.1 Mark completed items in TODO.md
