# Native preprocess roadmap (Rust-first, fewer Boltz `predict` runs)

## Goal

`boltr preprocess --mode native` / `boltr predict --preprocess native|auto` today only supports a **narrow** YAML profile (see [PREPROCESS_NATIVE.md](PREPROCESS_NATIVE.md)). Every input outside that profile falls back to **upstream `boltz predict`**, which runs a **full** PyTorch inference only to materialize `manifest.json` + flat `.npz` beside the YAML—then Boltr runs **LibTorch `predict_step`** again.

Widening **native** featurization reduces duplicate work and avoids GPU contention between Boltz and LibTorch.

## Eligibility expansion (ordered)

1. **Protein–protein complexes** with the same constraints as today (no ligands/DNA/RNA, empty `templates:` / `constraints:`) but **multiple protein entities**—already partially supported; verify parity with Python `load_input` + featurizer on shared fixtures ([TENSOR_CONTRACT.md](TENSOR_CONTRACT.md)).
2. **Templates / constraints** (residue-level): port or mirror `process_template_features` / constraint tensors into `boltr-io` with golden exports from Python.
3. **Ligands / non-protein polymers**: highest effort; depends on CCD paths and `StructureV2` parity.

## Acceptance

For each milestone: golden tensor dumps from `boltz-reference` on fixed YAMLs, then Rust collate + `predict_step` smoke on the same bundle without calling `boltz predict`.

## Non-goals (here)

Upstream **preprocess-only** CLI in Boltz (if it appears) would be tracked separately; this roadmap is **Rust featurizer** growth.
