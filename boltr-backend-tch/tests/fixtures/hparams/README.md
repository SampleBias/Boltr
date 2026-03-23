# Hyperparameter JSON fixtures

Run (with a real Boltz checkpoint):

```bash
python3 scripts/export_hparams_from_ckpt.py /path/to/boltz2.ckpt \
  boltr-backend-tch/tests/fixtures/hparams/boltz2_exported.json
```

Rust type: [`Boltz2Hparams`](../../src/boltz2/hparams.rs) — partial `serde` mapping; extend fields as `Boltz2Model::from_hparams` grows.
