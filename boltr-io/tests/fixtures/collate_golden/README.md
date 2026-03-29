# Collated batch golden (tensor contract)

This directory holds **machine-readable** artifacts for the Boltz2 inference batch that
`Boltz2Featurizer.process` + `collate()` produce in Python (`data/module/inferencev2.py`,
`data/feature/featurizerv2.py`).

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Authoritative **key names** and notes for `process_token_features`, MSA, dummy templates, and the merged `process()` return. |
| `trunk_smoke_collate.safetensors` | **Rust-native** single-example collate from [`load_input_smoke`](../load_input_smoke): `trunk_smoke_feature_batch_from_inference_input` → `collate_inference_batches`, plus zero `s_inputs` `[B,N,384]` for embedder smoke. Regenerate: `cargo run -p boltr-io --bin write_trunk_collate_from_fixture`. (Legacy synthetic script: [`scripts/dump_collate_golden.py`](../../../scripts/dump_collate_golden.py) — not aligned to `load_input_smoke`.) |
| `ala_structure_v2.npz` | Single-chain ALA `StructureV2` matching Rust [`fixtures::structure_v2_single_ala`](../../src/fixtures.rs). Written by `write_token_features_ala_golden`. |
| `token_features_ala_golden.safetensors` | Full **`process_token_features`** dict for that structure (no leading batch axis). Rust golden + tests: `cargo run -p boltr-io --bin write_token_features_ala_golden`. |
| `token_features_ala_collated_golden.safetensors` | Same tensors with **`B=1`** prepended on every key (collate-style). |
| `atom_features_ala_golden.safetensors` | Python **`process_atom_features`** for single-token ALA + canonical `mols/*.pkl`. Regenerate: [`scripts/dump_atom_features_golden.py`](../../../scripts/dump_atom_features_golden.py) (`--mol-dir` / `BOLTZ_MOL_DIR`). Rust schema tests: [`atom_features_golden.rs`](../../src/featurizer/atom_features_golden.rs). |
| `collate_two_msa_golden.safetensors` | Two-example **`msa`** after `pad_to_max` (variable last dim), for Rust `collate_inference_batches` parity. Regenerate: [`scripts/dump_collate_two_example_golden.py`](../../../scripts/dump_collate_two_example_golden.py) (NumPy + `safetensors` only). |

## Regeneration

**Authoritative (aligned with `load_input_smoke` + parity test):**

```bash
cargo run -p boltr-io --bin write_trunk_collate_from_fixture
```

After featurizer changes (e.g. new template or MSA keys), always re-run the command above and commit the updated `trunk_smoke_collate.safetensors` so `post_collate_golden` and `collate_golden_fixture` stay green.

Legacy synthetic shapes only (not fixture-aligned):

```bash
cargo run -p boltr-io --bin write_collate_golden
# or: python3 scripts/dump_collate_golden.py
```

**Token features (Boltz Python, authoritative):** with a full [jwohlwend/boltz](https://github.com/jwohlwend/boltz) checkout and `BOLTZ_SRC` set:

```bash
cargo run -p boltr-io --bin write_token_features_ala_golden   # refresh npz + Rust-derived safetensors
pip install torch safetensors numpy
python3 scripts/dump_token_features_ala_golden.py
```

CI compares the checked-in safetensors to live Rust `process_token_features`; after regenerating from Python, run `cargo test -p boltr-io --lib token_features_ala` and fix any Rust drift.

**Full Python collate batch (optional, upstream Boltz):** [`scripts/dump_full_collate_golden.py`](../../../scripts/dump_full_collate_golden.py) saves the post-`collate()` dict from `Boltz2InferenceDataModule` to safetensors (requires a full `boltz` install with `boltz.data`, not model-only `boltz-reference`). Use for cross-checks when Rust collate is compared against Python end-to-end.

**Post-collate parity (Rust):** [`post_collate_golden.rs`](../../post_collate_golden.rs) compares live `collate_inference_batches` output to `trunk_smoke_collate.safetensors` with atol=`1e-6`, rtol=`1e-5` (see [NUMERICAL_TOLERANCES.md](../../../docs/NUMERICAL_TOLERANCES.md)).

**Backend wiring:** [`boltr-backend-tch/tests/collate_predict_trunk.rs`](../../../boltr-backend-tch/tests/collate_predict_trunk.rs) loads this file and runs `Boltz2Model::predict_step_trunk` with `MsaFeatures`. Prefer [`scripts/cargo-tch`](../../../scripts/cargo-tch) for LibTorch:

```bash
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend --test collate_predict_trunk
```

## Reference

Upstream merge in `featurizerv2.py` (boltz main):

`{**token_features, **atom_features, **msa_features, **template_features, **ensemble_features, …}`

See also [docs/TENSOR_CONTRACT.md](../../../docs/TENSOR_CONTRACT.md).
