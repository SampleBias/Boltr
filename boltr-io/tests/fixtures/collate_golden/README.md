# Collated batch golden (tensor contract)

This directory holds **machine-readable** artifacts for the Boltz2 inference batch that
`Boltz2Featurizer.process` + `collate()` produce in Python (`data/module/inferencev2.py`,
`data/feature/featurizerv2.py`).

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Authoritative **key names** and notes for `process_token_features`, MSA, dummy templates, and the merged `process()` return. |
| `trunk_smoke_collate.safetensors` | Minimal **single-example** tensors (`batch=1`) for trunk smoke tests: `s_inputs`, `token_pad_mask`, MSA-related keys, template dummy keys. Regenerate with [`scripts/dump_collate_golden.py`](../../../scripts/dump_collate_golden.py). |
| `ala_structure_v2.npz` | Single-chain ALA `StructureV2` matching Rust [`fixtures::structure_v2_single_ala`](../../src/fixtures.rs). Written by `write_token_features_ala_golden`. |
| `token_features_ala_golden.safetensors` | Full **`process_token_features`** dict for that structure (no leading batch axis). Rust golden + tests: `cargo run -p boltr-io --bin write_token_features_ala_golden`. |
| `token_features_ala_collated_golden.safetensors` | Same tensors with **`B=1`** prepended on every key (collate-style). |

## Regeneration

Preferred (no PyTorch):

```bash
cargo run -p boltr-io --bin write_collate_golden
```

Alternative:

```bash
pip install torch safetensors
python3 scripts/dump_collate_golden.py
```

**Token features (Boltz Python, authoritative):** with a full [jwohlwend/boltz](https://github.com/jwohlwend/boltz) checkout and `BOLTZ_SRC` set:

```bash
cargo run -p boltr-io --bin write_token_features_ala_golden   # refresh npz + Rust-derived safetensors
pip install torch safetensors numpy
python3 scripts/dump_token_features_ala_golden.py
```

CI compares the checked-in safetensors to live Rust `process_token_features`; after regenerating from Python, run `cargo test -p boltr-io --lib token_features_ala` and fix any Rust drift.

Optional: extend [`scripts/dump_collate_golden.py`](../../../scripts/dump_collate_golden.py) to dump a full merged `collate()` dict from `Boltz2InferenceDataModule`.

**Backend wiring:** [`boltr-backend-tch/tests/collate_predict_trunk.rs`](../../../boltr-backend-tch/tests/collate_predict_trunk.rs) loads this file and runs `Boltz2Model::predict_step_trunk` with `MsaFeatures` (default `cargo test -p boltr-backend-tch --features tch-backend` when LibTorch is available).

## Reference

Upstream merge in `featurizerv2.py` (boltz main):

`{**token_features, **atom_features, **msa_features, **template_features, **ensemble_features, …}`

See also [docs/TENSOR_CONTRACT.md](../../../docs/TENSOR_CONTRACT.md).
