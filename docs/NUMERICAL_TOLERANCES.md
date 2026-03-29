# Numerical tolerances (registry)

Central reference for `rtol` / `atol` used in golden and integration tests. Policy aligns with [TENSOR_CONTRACT.md](TENSOR_CONTRACT.md) §6.5.

## Convention

For float tensors we use the usual bound:

\[
|a - b| \le \mathrm{atol} + \mathrm{rtol} \cdot \max(|a|, |b|, \epsilon)
\]

Backend goldens (`tch`) compare using the same scale-based check as `tch` tests (`scale = max(|a|,|b|)`). Featurizer tests in `boltr-io` use elementwise `atol + rtol * |b|` unless noted.

## `boltr-io` — featurizer goldens

| Area | rtol | atol | Location |
|------|------|------|----------|
| Token features (`process_token_features`) | `1e-5` | `1e-6` | [`token_features_golden.rs`](../boltr-io/src/featurizer/token_features_golden.rs) |
| MSA features | `1e-5` | `1e-6` | [`msa_features_golden.rs`](../boltr-io/src/featurizer/msa_features_golden.rs) |
| Atom features | `1e-4` | `1e-5` | [`atom_features_golden.rs`](../boltr-io/src/featurizer/atom_features_golden.rs) |

## `boltr-io` — post-collate parity

| Test | atol (f32) | rtol (f32) | Notes |
|------|------------|------------|--------|
| `post_collate_trunk_smoke_matches_golden_allclose` | `1e-6` | `1e-5` | [`post_collate_golden.rs`](../boltr-io/tests/post_collate_golden.rs) vs `trunk_smoke_collate.safetensors` |

Argument order in [`compare_inference_collate_to_safetensors`](../boltr-io/src/inference_collate_serialize.rs): `(coll, golden_bytes, f32_atol, f32_rtol)`.

## `boltr-backend-tch` — opt-in module goldens (LibTorch)

All use `rtol = 1e-4`, `atol = 1e-5` unless a test file says otherwise:

| Golden | Test file |
|--------|-----------|
| MSAModule | [`msa_module_golden.rs`](../boltr-backend-tch/tests/msa_module_golden.rs) |
| PairformerLayer | [`pairformer_golden.rs`](../boltr-backend-tch/tests/pairformer_golden.rs) |
| Trunk init (`rel_pos`, `s_init`) | [`trunk_init_golden.rs`](../boltr-backend-tch/tests/trunk_init_golden.rs) |
| Input embedder | [`input_embedder_golden.rs`](../boltr-backend-tch/tests/input_embedder_golden.rs) |

Run with `BOLTR_RUN_*_GOLDEN=1` and [`scripts/cargo-tch`](../scripts/cargo-tch) (see [scripts/README.md](../scripts/README.md)).

## End-to-end regression (`boltr` vs `boltz`)

Default env-based tolerances in [`scripts/regression_compare_predict.sh`](../scripts/regression_compare_predict.sh) and [`scripts/regression_tol.env.example`](../scripts/regression_tol.env.example). These are **looser** than layer goldens: full-pipeline outputs vary with dtype, ordering, and writer details.

| Quantity | Default rtol | Default atol |
|----------|--------------|--------------|
| Coordinates / generic float NPZ keys | `1e-3` | `1e-4` |
| Keys whose name contains `plddt` | `1e-3` | `1e-2` |
| PAE / PDE NPZ (in-script) | `1e-3` | `1e-2` |

Override via env or optional tol-file (see script header).

## Changing tolerances

1. Prefer tightening goldens after fixing a real bug, not loosening without cause.
2. Update this file and the matching `#[test]` / const when changing a checked-in policy.
3. For new tests, add a row here in the appropriate section.
