# Folding Prediction Upgrade Tasks

## Overview
Tasks identified from expert assessment of atom/residue utilization in folding prediction capability.

## Phase 1: Critical Fixes (High Priority)

- [ ] Implement ligand symmetry loading from CCD data
  - File: `boltr-io/src/featurizer/process_symmetry_features.rs`
  - Replace `get_ligand_symmetries_empty()` with actual CCD symmetry map
  - Wire symmetry map from preprocess pipeline
  - Test with ligand-containing complexes
  - Estimated effort: 3-4 days

- [ ] Enable multi-conformer ensemble sampling by default
  - File: `boltr-io/src/featurizer/process_ensemble_features.rs`
  - Change `inference_ensemble_features()` to return multiple conformers (5 instead of 1)
  - Update diffusion to handle multiple conformers
  - Implement ensemble averaging in conditioning
  - Estimated effort: 2-3 days

- [ ] Complete residue constraint integration
  - File: `boltr-io/src/featurizer/process_residue_constraint_features.rs`
  - Wire constraint tensors into diffusion sampling
  - Implement constraint-aware denoising steps
  - Add constraint violation penalties
  - File: `boltr-backend-tch/src/boltz2/diffusion.rs`
  - Estimated effort: 3-4 days

## Phase 2: Enhancements (Medium Priority)

- [ ] Implement frame-based local coordinates in diffusion
  - File: `boltr-backend-tch/src/boltz2/diffusion.rs`
  - Use computed frames as local coordinate systems for atom updates
  - Apply frame-based rotational constraints during sampling
  - Add frame-consistency checks
  - Estimated effort: 4-5 days

- [ ] Strengthen template integration
  - File: `boltr-io/src/featurizer/process_template_features.rs`
  - Increase template conditioning weight in diffusion
  - Add template-force features to atom encoder
  - Implement template-guided sampling pathways
  - Estimated effort: 3-4 days

- [ ] Enhance atom feature utilization in encoders
  - File: `boltr-backend-tch/src/boltz2/encoders.rs`
  - Review and optimize atom feature usage
  - Ensure all 388 atom features are properly weighted
  - Add feature importance analysis
  - Estimated effort: 2-3 days

## Phase 3: Validation & Testing (High Priority)

- [ ] Create unit tests for ligand symmetry handling
  - Test symmetry map loading and application
  - Test symmetry-aware sampling
  - Estimated effort: 1-2 days

- [ ] Create unit tests for multi-conformer features
  - Test conformer selection and averaging
  - Test ensemble feature generation
  - Estimated effort: 1 day

- [ ] Create unit tests for constraint integration
  - Test constraint tensor propagation
  - Test constraint satisfaction in diffusion
  - Estimated effort: 2 days

- [ ] Create integration tests for enhanced prediction
  - Test end-to-end prediction with ligands
  - Test multi-conformer diffusion
  - Test constraint satisfaction
  - Estimated effort: 2-3 days

- [ ] Create golden tensor tests
  - Export Python tensors for enhanced features
  - Compare Rust vs Python outputs numerically
  - Validate parity with strict tolerances
  - Estimated effort: 3-4 days

- [ ] Measure prediction accuracy improvements
  - Run RMSD tests on benchmark complexes
  - Compare to baseline predictions
  - Document improvements
  - Estimated effort: 2-3 days

## Documentation Tasks

- [ ] Update TODO.md with completion status
  - Mark completed items as [x]
  - Update partial items as [~]
  - Estimated effort: 1 hour

- [ ] Update DEVELOPMENT.md with new features
  - Document ligand symmetry loading
  - Document multi-conformer sampling
  - Document constraint integration
  - Estimated effort: 2 hours

- [ ] Create upgrade guide
  - Document the upgrade process
  - Include migration notes for existing users
  - Provide examples of new features
  - Estimated effort: 2-3 hours

## Total Estimated Effort
- Phase 1 (Critical): 8-11 days
- Phase 2 (Enhancements): 9-12 days
- Phase 3 (Validation): 9-13 days
- Documentation: 5-6 hours

**Total:** ~26-36 days of focused development work

## Dependencies
- Ligand symmetry requires CCD data loading infrastructure
- Multi-conformer requires ensemble preprocessing
- Constraints require constraint NPZ generation
- All enhancements require golden tensor export scripts

## Success Criteria
1. All ligand symmetries properly loaded and applied
2. Multi-conformer sampling enabled and working
3. Residue constraints enforced during diffusion
4. Frame-based local coordinates implemented
5. Template integration strengthened
6. All tests passing (unit, integration, golden)
7. Measurable accuracy improvements on benchmarks
8. Documentation complete and up-to-date
