# Todo: `process_atom_features` Rust Port

## Goal
Port Python `process_atom_features` (Boltz `featurizerv2.py` L1113-1540) to Rust in `boltr-io/src/featurizer/process_atom_features.rs`, producing tensors that match `atom_features_ala_golden.safetensors` on the ALA fixture.

## Tasks

### Phase 1: Extend data types for atom features
- [ ] Add `name`, `bfactor`, `plddt` fields to `AtomV2Row`
- [ ] Update `structure_v2_npz.rs` reader to decode atom name/bfactor/plddt
- [ ] Update `structure_v2_npz.rs` writer to encode atom name/bfactor/plddt
- [ ] Update `fixtures.rs` ALA fixture with correct atom names
- [ ] Fix any tests broken by the `AtomV2Row` change

### Phase 2: Implement `process_atom_features` core
- [ ] Define `AtomFeatureTensors` struct matching Python output dict
- [ ] Implement `convert_atom_name` (char → 4-element ASCII encoding)
- [ ] Implement backbone feature computation (protein + nucleic backbone index)
- [ ] Implement atom-to-token mapping, token-to-rep-atom, token-to-center-atom
- [ ] Implement `ref_space_uid` computation (chain-residue unique id)
- [ ] Implement distogram (`disto_target`) computation
- [ ] Implement padding to `atoms_per_window_queries` boundary
- [ ] Implement `to_feature_batch()` conversion
- [ ] Define `AtomRefData` trait for molecule-dependent fields (element, charge, chirality, conformer position)
- [ ] Implement `AtomRefData` for standard amino acids using static tables (no RDKit)

### Phase 3: Golden parity tests
- [ ] Add golden test: `atom_pad_mask` sum matches `ALA_STANDARD_HEAVY_ATOM_COUNT`
- [ ] Add golden test: `atom_to_token` shape matches
- [ ] Add golden test: `token_to_rep_atom` and `token_to_center_atom` indices match
- [ ] Add golden test: `backbone_feat` one-hot matches
- [ ] Add golden test: `ref_space_uid` matches
- [ ] Add golden test: `coords` after centering matches
- [ ] Add golden test: `disto_target` matches
- [ ] Add golden test: `ref_element`, `ref_charge`, `ref_chirality`, `ref_pos` match for ALA
- [ ] Add golden test: all tensor shapes match expected schema

### Phase 4: Wire into inference pipeline
- [ ] Add `process_atom_features` to `featurizer/mod.rs` exports
- [ ] Add `AtomFeatureTensors::to_feature_batch` to collate pipeline
- [ ] Update `lib.rs` re-exports
- [ ] Update `tasks/todo.md` and `docs/activity.md`
