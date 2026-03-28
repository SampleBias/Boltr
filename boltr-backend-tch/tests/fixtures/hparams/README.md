# Hyperparameter JSON fixtures

Run (with a real Boltz checkpoint):

```bash
python3 scripts/export_hparams_from_ckpt.py /path/to/boltz2.ckpt \
  boltr-backend-tch/tests/fixtures/hparams/boltz2_exported.json
```

Rust type: [`Boltz2Hparams`](../../src/boltz_hparams.rs) — maps common top-level Lightning keys plus nested `embedder_args` / `msa_args` / `training_args` / …; unknown keys land in `other`.

Fixtures:

| File | Purpose |
|------|---------|
| `minimal.json` | `token_s` / `token_z` / `num_blocks` only |
| `sample_full.json` | Representative nested dicts + flags in `other` |
