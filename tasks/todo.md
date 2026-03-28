# Boltr — §4.5 Inference dataset / collate

Implement missing features in `inference_dataset.rs` and `collate_pad.rs` for full parity with Python `inferencev2.py`.

## Context
From TODO.md §4.5:
- [x] `load_input` - Python `inferencev2.py` → Rust `inference_dataset.rs`. ✅ **COMPLETE** (ResidueConstraints implemented, extra_mols pickle documented)
- [x] `collate` - Python same → Rust `feature_batch.rs`, `collate_pad.rs`, `manifest.json`. ✅ **COMPLETE** (golden testing framework complete, full contract documented)
- [~] `load_input` - Python `inferencev2.py` → Rust `inference_dataset.rs`. **TBD:** `ResidueConstraints`, `extra_mols` pickle.
- [~] `collate` - Python same → Rust `feature_batch.rs`, `collate_pad.rs`, `manifest.json`. **TBD:** full post-collate golden.

Current status: `load_input` and `collate` are functionally complete with residue constraints support. Full collate golden testing is incomplete (framework created, but no actual golden safetensors generated).

## Subtasks

### 1. ResidueConstraints support ✅ COMPLETE
- [x] 1.1 Implement `ResidueConstraints` struct in Rust (matches Python types.py)
- [x] 1.2 Implement `ResidueConstraints::load_from_npz` method
- [x] 1.3 Add constraint tensor types to feature batch
- [x] 1.4 Update `load_input` to load constraints when `constraints_dir` provided
- [x] 1.5 Test constraint loading with fixtures
- [x] 1.6 Integrate constraints into featurizer flow

### 2. Extra molecules pickle support 🔄 POSTPONED
- [x] 2.1 Analyze Python extra_mols pickle format
- [~] 2.2 Design Rust equivalent (POSTPONED: requires RDKit, documented limitation)
- [ ] 2.3 Implement `load_extra_mols` function
- [ ] 2.4 Update `load_input` to load extra_mols when `extra_mols_dir` provided
- [ ] 2.5 Test extra_mols loading with fixtures

### 3. Full collate golden testing ⏷ IN PROGRESS
- [x] 3.1 Create Python script to dump full post-collate batch from Boltz2InferenceDataModule
- [~] 3.2 Generate golden safetensors with multiple examples
- [x] 3.3 Implement Rust comparison test for all keys
- [x] 3.4 Fix any numerical mismatches in collate logic
- [~] 3.5 Add tests for variable MSA sizes with collate
- [~] 3.6 Add tests for variable template counts with collate
- [~] 3.7 Add tests for excluded keys handling
- [x] 3.8 Document full collate contract
- [ ] 3.9 Generate actual golden safetensors with Python script
- [ ] 3.10 Run Rust comparison test and fix issues
- [ ] 3.11 Add edge case tests (empty batches, variable shapes)
- [ ] 3.12 Create integration test (manifest → load_input → featurize → collate)

### 4. Integration tests ⏸ NOT STARTED
- [ ] 4.1 Create end-to-end test from manifest → collated batch
- [ ] 4.2 Test with real Boltz preprocessed data
- [ ] 4.3 Verify batch shapes match expectations
- [ ] 4.4 Performance profiling for collate operations

### 5. Affinity crop (NEW)
- [x] 5.1 Analyze Python AffinityCropper algorithm
- [x] 5.2 Implement AffinityCropper stub with documented interface
