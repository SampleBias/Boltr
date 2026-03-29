# Backend test fixtures (`boltr-backend-tch`)

**Status:** See **[`TODO.md`](../../../TODO.md) §5** for backend graph completion; this folder holds **pinned weights** and **opt-in Python goldens** (env `BOLTR_RUN_*_GOLDEN=1`).

Safetensors / JSON used by **LibTorch** (`tch`) tests. Run tests via [`scripts/cargo-tch`](../../../scripts/cargo-tch) so `torch-sys` resolves Python’s LibTorch.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [`boltz2_smoke/`](boltz2_smoke/) | Minimal pinned weights for strict load / smoke. |
| [`hparams/`](hparams/) | Hyperparameter JSON samples for [`Boltz2Hparams`](../../src/boltz_hparams.rs). |
| [`msa_module_golden/`](msa_module_golden/) | MSAModule I/O + weights golden — opt-in test. |
| [`pairformer_golden/`](pairformer_golden/) | One `PairformerLayer` golden — opt-in test. |
| [`trunk_init_golden/`](trunk_init_golden/) | `rel_pos` / `s_init` init golden — opt-in test. |
| [`input_embedder_golden/`](input_embedder_golden/) | Full input embedder forward golden — opt-in test. |

Each subdirectory with generated files has its own **README** (regeneration commands).

## Default vs opt-in tests

- `cargo test -p boltr-backend-tch --features tch-backend --lib` — library tests (CI smoke).
- Integration-style tests under `tests/*.rs` (e.g. [`collate_predict_trunk.rs`](../collate_predict_trunk.rs)) run with the same feature flag when LibTorch is available.
- Module goldens require `BOLTR_RUN_MSA_GOLDEN=1`, `BOLTR_RUN_PAIRFORMER_GOLDEN=1`, etc.; see subdirectory READMEs and [scripts/README.md](../../../scripts/README.md).

## See also

- [NUMERICAL_TOLERANCES.md](../../../docs/NUMERICAL_TOLERANCES.md) — rtol/atol for goldens.
- [TENSOR_CONTRACT.md](../../../docs/TENSOR_CONTRACT.md) — tensor naming and checkpoint export.
