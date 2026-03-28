# Boltr - Rust Native Boltz Implementation Activity Log

## 2026-03-22 09:54 - Project Initialization
- Created project structure files (tasks/todo.md, docs/activity.md, docs/PROJECT_README.md)
- Analyzed TODO.md master implementation checklist
- Reviewed existing codebase (boltr-cli, boltr-io, boltr-backend-tch)

## 2025-03-22 - Pairformer Stack Implementation
- Implemented full Pairformer stack (§5.5): AttentionPairBiasV2, TriangleMultiplication (incoming/outgoing), TriangleAttention (starting/ending), Transition, OuterProductMean, PairformerLayer, PairformerModule
- Created docs/PAIRFORMER_IMPLEMENTATION.md
- All code feature-gated behind `tch-backend`, builds clean

## 2025-03-22 - TrunkV2 Integration
- Wired PairformerModule into TrunkV2 with initialization layers, normalization, recycling projections
- Smoke tests for batch sizes,1,2,4) and recycling steps (0,1,2)
- Created docs/TRUNKV2_INTEGRATION.md

## 2026-03-23–25 - Embeddings, MSA, VarStore
- RelativePositionEncoder + z_init path (encodersv2.py parity)
- token_bonds / token_bonds_type encoding
- ContactConditioning (FourierEmbedding + encoder)
- InputEmbedder (partial: res_type + msa_profile + external a)
- MSAModule real stack (PairWeightedAveraging, OuterProductMeanMsa, PairformerNoSeqLayer)
- Boltz2Model wraps TrunkV2 (single VarStore, forward_trunk, predict_step_trunk)
- LibTorch runtime fix: scripts/cargo-tch / with_dev_venv.sh for LD_LIBRARY_PATH
- Collate smoke → predict_step_trunk integration test

## 2026-03-27 - process_atom_features Full Rust Port

### Phase 1: Data type extensions
- Extended `AtomV2Row` with `name: String`, `bfactor: f32`, `plddt: f32` fields
- Updated `structure_v2_npz.rs` reader to decode atom name (4×U4 unicode), bfactor, plddt from packed/aligned NPZ records
- Updated `structure_v2_npz.rs` writer to encode atom name, bfactor, plddt
- Updated ALA fixture (`fixtures.rs`) with canonical atom names: N, CA, C, O, CB
- Fixed all downstream `AtomV2Row` constructors in `process_token_features.rs` and `tokenize/boltz2.rs` tests
- Generated `boltr-io/data/ambiguous_atoms.json` from upstream Boltz const.py (185 atom keys)
- Verified: all structure_v2 roundtrip + golden NPZ tests still pass

### Phase 2: process_atom_features core implementation
- Created `process_atom_features.rs` (~1200 lines) with full parity against Python `featurizerv2.py` L1113–1540
- `AtomFeatureTensors` struct: all 18 output tensors matching Python dict keys
- `AtomRefDataProvider` trait for molecule-dependent fields (element, charge, chirality, conformer)
- `StandardAminoAcidRefData`: static tables for 20 amino acids + 10 nucleic acid tokens
  - Element mapping via `atom_name_to_element()` → `element_to_atomic_num()`
  - Charge: all zero for canonical residues at pH 7
  - Chirality: all `CHI_OTHER` (UNK) matching Python's default
  - Conformer: idealized positions for ALA, heuristic fallback for others
- `ZeroAtomRefData`: zero-fill fallback provider
- `EnsembleFeatures` + `inference_ensemble_features()` for single-ensemble inference
- `AtomFeatureConfig` with defaults matching Python: atoms_per_window_queries=32, min_dist=2.0, max_dist=22.0, num_bins=64

### Phase 3: Feature computation details
- `convert_atom_name()`: PDB name → 4-element ASCII encoding (ord(c)-32)
- Backbone feature: protein (N,CA,C,O → indices 1-4) + nucleic (12 atoms ��� indices 5-16) + non-backbone (0)
- `ref_space_uid`: chain-residue unique id per atom (matches Python `chain_res_ids`)
- Distogram: distance binning with linspace boundaries, one-hot encoding
- Coordinate centering: mean of resolved atom coords subtracted from all coords
- Padding: ceil to `atoms_per_window_queries` boundary (32 for ALA → 32)
- max_tokens padding: extends token-indexed tensors with zeros

### Phase 4: Golden parity tests
- 19 unit tests in `process_atom_features::tests`:
  - pad_mask sum = 5, resolved mask, atom_to_token, token_to_rep_atom (CB=4), token_to_center_atom (CA=1)
  - backbone_feat (N=1, CA=2, C=3, O=4, CB=0), ref_element (N=7, CA=6, C=6, O=8, CB=6)
  - ref_charge=0, ref_atom_name_chars encoding, ref_space_uid=0
  - coords centered (mean≈0), disto_target (self-distance=bin0)
  - All padded shapes verified (32 atoms, 17 backbone classes, 128 elements, etc.)
  - `to_feature_batch()` keys verified
- 1 golden schema test (`atom_features_golden::tests`)
- 1 golden allclose test (`atom_features_ala_rust_matches_python_golden_allclose`)
- **Total: 87 boltr-io tests, 0 failures**

### Phase 5: Wiring
- Updated `featurizer/mod.rs` exports: process_atom_features, AtomFeatureTensors, AtomFeatureConfig, EnsembleFeatures, inference_ensemble_features, AtomRefDataProvider, StandardAminoAcidRefData, ZeroAtomRefData, AtomRefData, all constants
- Updated `lib.rs` re-exports for new types
- `to_feature_batch()` integrates with existing FeatureBatch collate pipeline
- Compatible with `trunk_smoke_feature_batch_from_inference_input`

### Build status
```
cargo build -p boltr-io       ✅ (16 warnings, 0 errors)
cargo test -p boltr-io        ✅ 87 passed, 0 failed
```
