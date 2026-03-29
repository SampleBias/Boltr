# `load_input_smoke`

Minimal **on-disk** fixture for [`load_input`](../../../src/inference_dataset.rs) and downstream **trunk smoke** tests.

## Contents

| File | Role |
|------|------|
| `manifest.json` | [`Manifest`](../../../src/inference_dataset.rs) listing one (or more) records and paths relative to this directory. |

Supporting inputs (structures, MSA, constraints, etc.) live **next to** `manifest.json` as referenced by the manifest. Tests resolve paths from [`CARGO_MANIFEST_DIR`]/`tests/fixtures/load_input_smoke`.

## Used by

- Integration tests under `boltr-io/tests/` that call `load_input` + featurizer smoke.
- [`trunk_smoke_feature_batch_from_inference_input`](../../../src/inference_dataset.rs) → [`collate_inference_batches`](../../../src/collate_pad.rs) → golden in [`collate_golden/`](../collate_golden/).

## Regeneration

Edit `manifest.json` and any referenced files when extending coverage; then re-run:

```bash
cargo run -p boltr-io --bin write_trunk_collate_from_fixture
```

and update [`collate_golden/trunk_smoke_collate.safetensors`](../collate_golden/) if the featurizer contract changes.
