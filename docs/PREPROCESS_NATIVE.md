# Native preprocess (Tier 2) — scope and testing strategy

## Supported profile

`boltr preprocess --mode native` and `boltr predict --preprocess native|auto` use [`write_native_preprocess_bundle`](../boltr-io/src/preprocess/native.rs), which is **only** valid when:

- Every `sequences:` entry is a **`protein:`** with **canonical** one-letter amino acid sequences.
- There are **no** `dna:`, `rna:`, or `ligand:` entries.
- **`templates:`** and **`constraints:`** are absent from the YAML.
- **`protein.modifications:`** is absent.

**Coordinates** are placeholders (chains separated along +X; residues spaced along a line). They are sufficient for the Rust `load_input` → collate → `predict_step` bridge, but **not** for biologically realistic starting geometry. For that, use **upstream Boltz** (`boltr preprocess --mode boltz` or `--preprocess boltz`).

## MSA handling

- Explicit `msa: path/to.a3m` or `msa: empty` are resolved relative to the YAML directory.
- With **`boltr predict --preprocess native|auto`**, **`--use-msa-server`**, and **no** `msa:` in YAML, the CLI writes MSAs to **`./msa/<CHAIN>.a3m`** next to the YAML, then native preprocess reads **`first_chain.a3m`** per protein entity when building MSA `.npz`. Prefer adding explicit `msa:` paths in YAML for multi-entity cases.

## Golden / parity testing

- **Unit tests:** [`boltr-io/tests/preprocess_bundle.rs`](../boltr-io/tests/preprocess_bundle.rs) — manifest discovery, `copy_flat_preprocess_bundle`, native bundle smoke.
- **Parity with Python Boltz** (for future work): follow [`docs/TENSOR_CONTRACT.md`](TENSOR_CONTRACT.md) — export a small fixture from Python `load_input` on the same YAML + compare `StructureV2` npz decode or `tokenize` invariants, not raw zip bytes.

## Affinity

Native preprocess does **not** produce `pre_affinity_{id}.npz` under `{id}/`. Do not use **`--affinity`** with native-only bundles until that layout is implemented.
