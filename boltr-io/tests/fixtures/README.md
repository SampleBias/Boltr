# Test fixtures (`boltr-io`)

Layout of **`tests/fixtures/`** for parsing, featurizer, and collate tests. Paths are relative to this directory.

## Layout

| Path | Purpose |
|------|---------|
| [`yaml/`](yaml/) | YAML inputs for [`yaml_parse.rs`](../yaml_parse.rs) and related parser tests — entities, constraints, templates, affinity `properties`, etc. |
| [`minimal_protein.yaml`](minimal_protein.yaml) | Minimal single-protein example (legacy / shared references). |
| [`load_input_smoke/`](load_input_smoke/) | **Manifest + on-disk inputs** for [`load_input`](../integration_smoke.rs) and trunk smoke pipelines. See [`load_input_smoke/README.md`](load_input_smoke/README.md). |
| [`collate_golden/`](collate_golden/) | **Safetensors / npz goldens** for token, atom, MSA, and trunk collate contract. See [`collate_golden/README.md`](collate_golden/README.md). |

## Regeneration (quick reference)

| Artifact | Command / script |
|----------|-------------------|
| Trunk collate golden (`trunk_smoke_collate.safetensors`) | `cargo run -p boltr-io --bin write_trunk_collate_from_fixture` |
| Token ALA goldens | `cargo run -p boltr-io --bin write_token_features_ala_golden` |
| Legacy synthetic collate only | `cargo run -p boltr-io --bin write_collate_golden` or `python3 scripts/dump_collate_golden.py` |

Full-Python Boltz collate dump (requires a **full** [jwohlwend/boltz](https://github.com/jwohlwend/boltz) tree with `boltz.data` on `PYTHONPATH`, **not** model-only `boltz-reference`):

- [`scripts/dump_full_collate_golden.py`](../../../scripts/dump_full_collate_golden.py) — writes `full_collate_golden.safetensors` for optional cross-checks.

## Related docs

- [TENSOR_CONTRACT.md](../../../docs/TENSOR_CONTRACT.md) — tensor names and pipeline.
- [NUMERICAL_TOLERANCES.md](../../../docs/NUMERICAL_TOLERANCES.md) — rtol/atol registry.
- [scripts/README.md](../../../scripts/README.md) — all Python helper scripts.
