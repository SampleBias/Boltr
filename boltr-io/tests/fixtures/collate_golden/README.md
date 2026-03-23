# Collated batch golden (tensor contract)

This directory holds **machine-readable** artifacts for the Boltz2 inference batch that
`Boltz2Featurizer.process` + `collate()` produce in Python (`data/module/inferencev2.py`,
`data/feature/featurizerv2.py`).

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Authoritative **key names** and notes for `process_token_features`, MSA, dummy templates, and the merged `process()` return. |
| `trunk_smoke_collate.safetensors` | Minimal **single-example** tensors (`batch=1`) for trunk smoke tests: `s_inputs`, `token_pad_mask`, MSA-related keys, template dummy keys. Regenerate with [`scripts/dump_collate_golden.py`](../../../scripts/dump_collate_golden.py). |

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

Optional: with the full `boltz` package installed, extend the script to dump a real `collate()` dict from `Boltz2InferenceDataModule` and merge into this directory.

## Reference

Upstream merge in `featurizerv2.py` (boltz main):

`{**token_features, **atom_features, **msa_features, **template_features, **ensemble_features, …}`

See also [docs/TENSOR_CONTRACT.md](../../../docs/TENSOR_CONTRACT.md).
