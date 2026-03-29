## 2026-03-29 14:30 — `boltr-cli` §6 Full implementation
                   |
|--------------------|
| **Completed:** |
| - All 9 tests pass: `cargo test -p boltr-cli` (0 warnings in boltr-io, pre-existing) |
| - CLI flags now match Boltz upstream `boltz predict` interface |
| - Full predict pipeline with checkpoint loading + output writers |
| - Test coverage: CLI integration tests + unit tests |

| **Activity:** |
| - Rewrote `main.rs` and `predict_tch.rs` for full predict pipeline |
- Added `PredictFlowArgs` struct with all Boltz-compatible flags |
- Moved flags out of global CLI (fixed subcommand short-flag clash) |
- Implemented checkpoint resolution ( hparams, model building, weight loading |
- Added `write_prediction_outputs` and `write_structure_file` functions |
- Spike-only path preserved and enhanced |
- `BOLTZ_CACHE` env var support |
- `--output-format` flag (pdb/mmcif) |
- `--checkpoint`, and `--affinity-checkpoint` flags |
- `--step-scale`, (default 1.638) |
- `--max-msa-seqs` (default 8192) |
- `--override`, `--write-full-pae`, `--write-full-pde` flags |
- `--affinity-mw-correction`, flag |
- `--sampling-steps-affinity`, and `--diffusion-samples-affinity` flags |
- `--preprocessing-threads` flag |
