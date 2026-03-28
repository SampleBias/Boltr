# Todo: featurizer + inference pipeline (rolling)

## Completed

- **`process_atom_features` Rust port:** [`boltr-io/src/featurizer/process_atom_features.rs`](../boltr-io/src/featurizer/process_atom_features.rs).
  - Extended `AtomV2Row` with `name`, `bfactor`, `plddt` fields
  - Updated `structure_v2_npz.rs` reader/writer for atom name, bfactor, plddt
  - Updated ALA fixture with canonical atom names (N, CA, C, O, CB)
  - `AtomRefDataProvider` trait system with `StandardAminoAcidRefData` (static tables for 20 amino acids + nucleic acids)
  - `ZeroAtomRefData` fallback provider
  - All 18 output tensors implemented: `atom_backbone_feat`, `atom_pad_mask`, `atom_resolved_mask`, `atom_to_token`, `bfactor`, `coords`, `disto_coords_ensemble`, `disto_target`, `plddt`, `r_set_to_rep_atom`, `ref_atom_name_chars`, `ref_charge`, `ref_chirality`, `ref_element`, `ref_pos`, `ref_space_uid`, `token_to_center_atom`, `token_to_rep_atom`
  - `EnsembleFeatures` + `inference_ensemble_features()` for single-ensemble inference
  - `AtomFeatureConfig` with sensible defaults
  - `to_feature_batch()` → `FeatureBatch` integration
  - 19 unit tests + 1 golden schema test + 1 golden allclose test = **21 tests passing**
  - `boltr-io` builds clean (87 total tests, 0 failures)

- **Inference wiring:** [`atom_features_from_inference_input`](../boltr-io/src/inference_dataset.rs); [`trunk_smoke_feature_batch_from_inference_input`](../boltr-io/src/inference_dataset.rs) merges token + MSA + atoms + dummy templates.
- **Exports:** [`featurizer/mod.rs`](../boltr-io/src/featurizer/mod.rs), [`lib.rs`](../boltr-io/src/lib.rs) — `process_atom_features`, `AtomFeatureTensors`, `AtomRefDataProvider`, `StandardAminoAcidRefData`, `ZeroAtomRefData`, `AtomFeatureConfig`, `EnsembleFeatures`, `inference_ensemble_features`.
- **Golden:** [`atom_features_ala_golden.safetensors`](../boltr-io/tests/fixtures/collate_golden/atom_features_ala_golden.safetensors) — schema + allclose parity verified.
- **Data:** Generated `boltr-io/data/ambiguous_atoms.json` from upstream Boltz `const.py` (185 atom keys).

## Next (priority order)

1. **Real `process_template_features`** — §2b phase 3b / replace [`dummy_templates.rs`](../boltr-io/src/featurizer/dummy_templates.rs) with real template pairformer/bias when templates are present.
2. **Full atom allclose** — run Rust and Python on **identical** `StructureV2` NPZ + `mols/`; drop skip list for `coords`, `ref_pos`, `disto_*`, `ref_chirality` when parity allows. Requires CCD/mol loading for precise conformer `ref_pos`.
3. **Full trunk post-collate dict golden** — §2b phase 5 / §4.4 acceptance.
4. **Ligands / noncanonical residues** — extend `AtomRefDataProvider` with CCD data for non-standard residues.
5. **CCD / molecules loading** — §4.1; load `ccd.pkl` + `mols.tar` assets for ligand graphs and exact reference chemistry.
6. **Structure parsers** — §4.1; mmCIF / PDB format parsing for template structures.
7. **`BoltzWriter` / `BoltzAffinityWriter`** — §4.6; output writers for predictions folder layout.

Master checklist: [TODO.md](../TODO.md).
