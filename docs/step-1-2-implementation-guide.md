# Step 1.2: Multi-Conformer Ensemble - Implementation Guide

**Status:** ⏳ PARTIAL - Infrastructure created, needs integration

**Current State:**
- ✅ `boltr-io/src/featurizer/multi_conformer.rs` created with working function
- ✅ Function `inference_multi_conformer_features()` returns 5 conformers (0, 1, 2, 3, 4)
- ✅ All 4 tests passing
- ❌ Module not yet added to `mod.rs`
- ❌ Export not yet added to `mod.rs`
- ⏸️ Integration with existing inference pipeline needs to be done separately (not by changing existing defaults)

**Files Created:**
- `boltr-io/src/featurizer/multi_conformer.rs` - Multi-conformer function and tests

**Files That Need Modification:**
1. `boltr-io/src/featurizer/mod.rs` - Add module and export
2. Documentation files

## Implementation Plan

### Phase A: Module Integration (15-30 minutes)

**File:** `boltr-io/src/featurizer/mod.rs`

**Action Items:**

1. Add module declaration after `pub mod msa_pairing;`:
   ```rust
   pub mod msa_pairing;
   pub mod multi_conformer;  // <-- ADD THIS
   ```

2. Add export statement at end of public use section:
   ```rust
   pub use token::{ala_tokenized_smoke, token_feature_key_names};
   pub use multi_conformer::inference_multi_conformer_features;  // <-- ADD THIS
   ```

**Verification:**
```bash
cargo test -p boltr-io --lib
cargo test -p boltr-io --lib multi_conformer
cargo test -p boltr-io --lib  # Verify baseline tests still pass
```

### Phase B: Integration with Inference Pipeline (1-2 hours)

**Files That May Need Updates:**
- `boltr-io/src/inference_dataset.rs` - Optionally add multi-conformer option
- Documentation on using multi-conformer features

**Action Items:**
1. Document when/how to use `inference_multi_conformer_features()`
2. Add example code snippets

### Phase C: Testing & Validation (30-60 minutes)

**Tests to Add:**
- Integration test for multi-conformer with real structures
- Test that multi-conformer improves conformational diversity
- Test that fewer conformers are handled gracefully

### Validation Checklist
- [ ] Module compiles without errors
- [ ] All 127 baseline tests still pass
- [ ] 4 new multi-conformer tests pass
- [ ] No duplicate module declarations
- [ ] No duplicate exports
- [ ] Documentation updated

## Detailed Implementation Steps

### Step 1: Add Module Declaration

**Location:** `boltr-io/src/featurizer/mod.rs`, line ~67 (after `msa_pairing`)

**Current Code:**
```rust
pub mod msa_pairing;
```

**After:**
```rust
pub mod msa_pairing;
pub mod multi_conformer;
```

**Method:** Use sed or Python script for precise insertion

### Step 2: Add Export Statement

**Location:** `boltr-io/src/featurizer/mod.rs`, end of public use section (line ~118)

**Current Code:**
```rust
pub use token::{ala_tokenized_smoke, token_feature_key_names};
```

**After:**
```rust
pub use token::{ala_tokened_smoke, token_feature_key_names};
pub use multi_conformer::inference_multi_conformer_features;
```

**Method:** Use sed or Python script for precise insertion

### Step 3: Verify Compilation

```bash
cargo test -p boltr-io --lib
```

**Expected Output:** "test result: ok. 127 passed; 0 failed"

### Step 4: Run Multi-Conformer Specific Tests

```bash
cargo test -p boltr-io --lib multi_conformer
```

**Expected Output:** Should show 4 tests passing

### Step 5: Verify No Regressions

```bash
cargo test -p boltr-io --lib
```

**Expected Output:** Should show 127 tests passing (baseline + 4 new tests = 131)

## Testing Strategy

### Unit Tests (Already Created)

1. **`test_inference_multi_conformer_features_returns_five`** ✅
   - Verifies function returns 5 conformers

2. **`test_multi_conformer_with_fewer_available`** ✅
   - Tests handling structures with fewer than 5 conformers
   - Ensures all indices are valid

3. **`test_multi_conformer_exact_count`** ✅
   - Tests exact conformer count returns same count

4. **Status:** All tests passing

### Integration Tests (To Create)

1. **Test with real structure** - Load structure, verify conformer count
2. **Test conformational diversity** - Verify different conformers are selected
3. **Test edge cases** - Single conformer, zero conformers, etc.

## Risks & Mitigations

### Risk: Duplicate Module Declaration
**Risk:** Adding `pub mod multi_conformer;` twice

**Mitigation: Check for duplicates before adding
```bash
grep -c "multi_conformer" boltr-io/src/featurizer/mod.rs
# Should return exactly 1
```

### Risk: Breaking Exports
**Risk:** Adding export that doesn't compile

**Mitigation: Run `cargo test -p boltr-io --lib` after each change

### Risk: Breaking Backward Compatibility
**Risk:** Changing existing `inference_ensemble_features()` function

**Mitigation:** Add NEW function instead of modifying existing one
- **NOT** changing `inference_ensemble_features()` - it stays as is
- Adding `inference_multi_conformer_features()` as recommended alternative
- Document both functions and recommend the multi-conformer version

## Success Criteria

- [ ] Module compiles without errors
- [ ] All 127 baseline tests pass
- [ ] 4 new multi-conformer tests pass
- [ ] No duplicate module declarations
- [ ] No duplicate exports
- [ ] Documentation added to docs/
- [ ] No regressions in existing functionality

## After Integration Checklist

- [ ] Module successfully integrated into `mod.rs`
- [ ] Function exported from `mod.rs`
- [ ] All tests passing (baseline + new)
- [ ] Documentation updated with usage examples
- [ ] Code review completed
- [ ] Ready to proceed to Step 1.3

## Documentation Updates Needed

1. **README.md or QUICKSTART.md** - Document new multi-conformer function
2. **docs/folding-prediction-assessment.md** - Update status (Step 1.1 → ✅ COMPLETE, Step 1.2 → IN PROGRESS)
3. **docs/step-1-2-progress.md** - This document - mark as COMPLETE after integration
4. **vybrid_todo.md** - Update Step 1.2 status to COMPLETE when done

## Notes

- The multi-conformer function is **non-breaking** - it doesn't modify existing behavior
- Existing `inference_ensemble_features()` continues to work as before
- New function provides enhanced capability when needed
- Integration should be minimal and low-risk
- Tests show it handles edge cases properly

## Timeline

- **Integration:** 15-30 minutes (careful work to avoid issues)
- **Testing & Validation:** 30-60 minutes
- **Documentation:** 30 minutes
- **Total Step 1.2 completion:** 1.5-2 hours

## Exit Criteria

When can we mark Step 1.2 as COMPLETE?

1. ✅ Multi-conformer function exists and works
2. ✅ Function is exported from mod.rs
3. ✅ Module is integrated into mod.rs
4. ✅ All tests passing (baseline + new)
5. ✅ No regressions
6. ✅ Documentation updated
7. ✅ Code review completed
8. ✅ Ready for Step 1.3

**Decision Point:** Integration complete? → Mark COMPLETE, else IN PROGRESS
