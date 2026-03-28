# Todo: featurizer + inference pipeline (rolling)

## Completed

- **`process_atom_features` Rust port:** [`boltr-io/src/featurizer/process_atom_features.rs`](../boltr-io/src/featurizer/process_atom_features.rs).
- **Inference wiring:** [`atom_features_from_inference_input`](../boltr-io/src/inference_dataset.rs); [`trunk_smoke_feature_batch_from_inference_input`](../boltr-io/src/inference_dataset.rs) merges token + MSA + atoms + dummy templates.
- **Exports:** [`featurizer/mod.rs`](../boltr-io/src/featurizer/mod.rs), [`lib.rs`](../boltr-io/src/lib.rs) — `process_atom_features`, `AtomFeatureTensors`, `AtomRefDataProvider`, `StandardAminoAcidRefData`, `AtomFeatureConfig`, `inference_ensemble_features`.
- **Golden:** [`atom_features_ala_rust_matches_python_golden_allclose`](../boltr-io/src/featurizer/atom_features_golden.rs) (skips keys tied to RDKit NPZ / conformer — see `ATOM_GOLDEN_SKIP_ALCLOSE`); [`manifest.json`](../boltr-io/tests/fixtures/collate_golden/manifest.json) `atom_features_ala_golden_keys`.

## Next (priority order)

1. **Real `process_template_features`** — §2b **3b** / replace [`dummy_templates.rs`](../boltr-io/src/featurizer/dummy_templates.rs) path when templates present.
2. **Full atom allclose** — run Rust and Python on **identical** `StructureV2` NPZ + `mols/`; drop skip list for `coords`, `ref_pos`, `disto_*`, `ref_chirality` when parity allows.
3. **Full trunk post-collate dict golden** — §2b phase **5** / §4.4 acceptance.
4. **Ligands / noncanonical residues** — extend `AtomRefDataProvider` (CCD).

Master checklist: [TODO.md](../TODO.md).
