# Step 1.1: Ligand Symmetry Loading - Implementation Progress

## Status: Part 1 Complete, Part 2 In Progress

## Part 1: CCD Symmetry Extraction Infrastructure ✅ COMPLETE

### What Was Done
1. Added `extract_symmetry_groups()` method to `CcdMolData`
   - Detects aromatic ring symmetries (180° rotations in 6-membered rings)
   - Detects symmetric terminal groups (equivalent atoms with same bond environment)
   - Returns groups of atom index pairs that can be swapped

2. Added helper methods to `CcdMolData`
   - `find_aromatic_ring()` - DFS-based ring detection
   - `are_atoms_equivalent()` - Check if two atoms have equivalent bond environments
   - `get_bond_type()` - Get bond type between two atoms

3. Added `build_symmetry_map()` to `CcdMolProvider`
   - Builds symmetry map for all loaded molecules
   - Returns `HashMap<String, Vec<Vec<(usize, usize)>>>`

4. Added comprehensive tests
   - All 8 CCD tests passing
   - Tests cover single atoms, aromatic rings, equivalent atoms, and provider-level mapping

### Code Quality
- Compiles without warnings
- No regressions in existing tests
- Well-documented with rustdoc comments
- Follows existing code style

## Part 2: Wiring Into Symmetry Features ⏳ IN PROGRESS

### What Needs to Be Done
1. Update `process_symmetry_features_with_ligand_symmetries` signature
   - Add `ccd_provider: Option<&CcdMolProvider>` parameter
   - Extract symmetry map from provider if available
   - Pass to `get_chain_symmetries`

2. Update `get_chain_symmetries` signature
   - Add `ligand_symmetry_map: Option<&HashMap<String, Vec<Vec<(usize, usize)>>>` parameter
   - Use the map to populate `ligand_symmetries` field
   - Fall back to empty list if not provided

3. Add integration tests
   - Test with actual ligand-containing structures
   - Verify symmetry groups are correctly populated
   - Check atom index mapping to crop space

### Current State
The infrastructure is in place:
- CCD molecules can extract symmetry groups
- Provider can build symmetry map
- `get_ligand_symmetries_for_tokens` already exists and works correctly

**Missing:** Connection between provider and the symmetry processing pipeline

## Files to Modify in Part 2

### 1. `boltr-io/src/featurizer/process_symmetry_features.rs`

**Current signature:**
```rust
pub fn process_symmetry_features_with_ligand_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    ligand_symmetry_map: Option<&HashMap<String, Vec<Vec<(usize, usize)>>>,
) -> SymmetryFeatures
```

**New signature needed:**
```rust
pub fn process_symmetry_features_with_ligand_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    ccd_provider: Option<&CcdMolProvider>,  // NEW
) -> SymmetryFeatures
```

**Implementation:**
```rust
pub fn process_symmetry_features_with_ligand_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    ccd_provider: Option<&CcdMolProvider>,
) -> SymmetryFeatures {
    let mut rng = StdRng::seed_from_u64(0);
    
    // Build symmetry map from CCD provider
    let symmetry_map = ccd_provider.map(|p| p.build_symmetry_map());
    
    let mut f = get_chain_symmetries(
        structure,
        tokens,
        100,
        symmetry_map.as_ref(),  // Pass to get_chain_symmetries
        &mut rng,
    );
    f
}
```

### 2. Update `get_chain_symmetries` signature

**Current signature:**
```rust
pub fn get_chain_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    max_n_symmetries: usize,
    rng: &mut impl Rng,
) -> SymmetryFeatures
```

**New signature needed:**
```rust
pub fn get_chain_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    max_n_symmetries: usize,
    ligand_symmetry_map: Option<&HashMap<String, Vec<Vec<(usize, usize)>>>,
    rng: &mut impl Rng,
) -> SymmetryFeatures
```

**Implementation change:**
```rust
// Replace:
ligand_symmetries: get_ligand_symmetries_empty(),

// With:
ligand_symmetries: ligand_symmetry_map
    .map(|m| get_ligand_symmetries_for_tokens(tokens, m))
    .unwrap_or_else(get_ligand_symmetries_empty),
```

## Testing Plan for Part 2

### Unit Test: `test_process_symmetry_features_with_ccd_provider`
- Create mock CCD provider with ligand symmetry data
- Create structure with NONPOLYMER tokens
- Verify ligand_symmetries field is populated correctly
- Check atom index mapping

### Integration Test: `test_end_to_end_ligand_symmetry`
- Load actual ligand-containing structure
- Create CcdMolProvider with ligand data
- Run process_symmetry_features_with_ligand_symmetries
- Verify symmetry groups match expected

### Golden Test (Future): `test_ligand_symmetry_python_parity`
- Export Python ligand symmetry tensors
- Compare with Rust implementation
- Ensure numerical parity

## Dependencies and Integration Points

### Current Call Sites
Need to update all callers of:
- `process_symmetry_features()` - remains unchanged (wrapper)
- `process_symmetry_features_with_ligand_symmetries()` - signature change

### New Call Sites (Future)
- `inference_dataset.rs` - pass CCD provider when available
- `featurizer/mod.rs` - update public API

## Success Criteria for Step 1.1 Completion

- [x] CCD symmetry extraction implemented
- [x] Provider symmetry map building implemented
- [x] Unit tests for symmetry extraction passing
- [ ] Updated `process_symmetry_features_with_ligand_symmetries` signature
- [ ] Updated `get_chain_symmetries` to use symmetry map
- [ ] Integration tests passing
- [ ] No regressions in existing tests
- [ ] Documentation updated

## Risk Assessment

**Low Risk:** The changes are additive and don't modify existing behavior when CCD provider is None.

**Backward Compatibility:** Fully maintained - existing code paths continue to work.

**Performance Impact:** Minimal - symmetry extraction is O(N) where N is number of bonds.

## Next Steps After Step 1.1

1. Complete Part 2 (wiring) - estimated 2-3 hours
2. Run full test suite - estimated 30 minutes
3. Move to Step 1.2: Multi-conformer ensemble sampling
4. Update documentation

## Notes

- The `get_ligand_symmetries_for_tokens` function already works correctly
- It's already tested in `ligand_symmetry_map_maps_atom_indices` test
- The main work is connecting the CCD provider to this function
- All infrastructure is in place, just needs wiring
