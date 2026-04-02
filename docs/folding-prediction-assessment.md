# Folding Prediction Assessment - Boltr

**Date:** 2025-04-02  
**Reviewer:** Vybrid (Expert Rust Engineer)  
**Focus:** Atom and residue utilization in folding prediction capability

---

## Executive Summary

The Boltr implementation demonstrates a sophisticated folding prediction pipeline that **DOES** effectively utilize atoms and residues throughout the prediction workflow. However, there are several areas where the atom-level detail could be enhanced for more accurate predictions, particularly around:

1. **Atom-level symmetry and conformer sampling**
2. **Residue constraint integration**
3. **Frame computation and propagation**
4. **Ligand/non-standard residue handling**

The system is working well as stated, but there are opportunities to improve prediction accuracy through better utilization of the available atom and residue information.

---

## Current Architecture Analysis

### 1. Atom-Level Processing ✅ **WORKING WELL**

**Strengths:**
- Comprehensive atom feature extraction in `process_atom_features.rs` (388-dim features)
- Detailed atom-to-token mapping with one-hot encoding
- Backbone and disto atom identification per residue type
- Element, charge, and chirality features per atom
- Reference conformer positions per atom
- Atom-level padding and masking

**Key Implementation:**
```rust
// From process_atom_features.rs
pub struct AtomFeatureTensors {
    pub atom_backbone_feat: Array2<f32>,  // [N_atoms, 17]
    pub atom_to_token: Array2<f32>,       // [N_atoms, N_tokens]
    pub ref_element: Array2<f32>,         // [N_atoms, 128] one-hot
    pub ref_pos: Array2<f32>,             // [N_atoms, 3] conformer
    pub ref_charge: Array1<f32>,          // [N_atoms]
    pub ref_chirality: Array1<i64>,       // [N_atoms]
    pub coords: Array3<f32>,              // [N_ensemble, N_atoms, 3]
    // ... more atom-level features
}
```

**Assessment:** ✅ **Strong** - Atom-level information is comprehensively extracted and utilized.

---

### 2. Residue-Level Processing ✅ **WORKING WELL**

**Strengths:**
- Proper residue tokenization with standard/non-standard distinction
- Center and disto atom tracking per residue
- Frame computation for protein residues (N-CA-C)
- Residue type encoding (384-dim token space)
- Residue-to-atom mapping and masking

**Key Implementation:**
```rust
// From tokenize/boltz2.rs
pub struct TokenData {
    pub token_idx: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub res_idx: i32,
    pub res_type: i32,
    pub res_name: String,
    pub center_idx: i32,      // CA for proteins, C1' for nucleic
    pub disto_idx: i32,       // CB for proteins, etc.
    pub frame_rot: [f32; 9],  // 3x3 rotation matrix
    pub frame_t: [f32; 3],    // translation vector
    pub frame_mask: bool,
    pub resolved_mask: bool,
    // ... more fields
}
```

**Frame Computation:**
```rust
pub fn compute_frame(n: [f32; 3], ca: [f32; 3], c: [f32; 3]) -> ([f32; 9], [f32; 3]) {
    // Proper backbone frame computation
    let v1 = sub(c, ca);      // C - CA
    let v2 = sub(n, ca);      // N - CA
    let e1 = scale(v1, 1.0 / norm(v1));
    let proj = dot(e1, v2);
    let u2 = sub(v2, scale(e1, proj));
    let e2 = scale(u2, 1.0 / norm(u2));
    let e3 = cross(e1, e2);
    // Returns rotation matrix and translation
}
```

**Assessment:** ✅ **Strong** - Residue-level processing is comprehensive with proper geometric features.

---

### 3. Symmetry Features ⚠️ **NEEDS ENHANCEMENT**

**Current State:**
- Amino acid symmetries are computed (ASP, GLU, PHE, TYR)
- Chain symmetries are generated
- Ligand symmetries are **empty by default** (`get_ligand_symmetries_empty()`)

**Issue:**
```rust
// From process_symmetry_features.rs
#[must_use]
pub fn get_ligand_symmetries_empty() -> Vec<Vec<Vec<(usize, usize)>>> {
    Vec::new()  // ← PROBLEM: No ligand symmetry data!
}
```

**Impact:** 
- Ligand molecules are treated without symmetry awareness
- Missing potential energy minima from symmetric atom arrangements
- Could lead to suboptimal ligand conformations in predictions

**Recommendation:**
- Implement CCD/RDKit symmetry data loading
- Wire ligand symmetry map from preprocess
- Test with ligand-containing complexes

---

### 4. Ensemble Features ⚠️ **UNDERUTILIZED**

**Current State:**
- Default is **single conformer** (`inference_ensemble_features()` returns `[0]`)
- Multiple ensemble sampling exists but requires explicit configuration
- Ensemble features are computed but not fully leveraged in diffusion

**Issue:**
```rust
// From process_ensemble_features.rs
#[must_use]
pub fn inference_ensemble_features() -> EnsembleFeatures {
    EnsembleFeatures {
        ensemble_ref_idxs: vec![0],  // ← Always uses first conformer only!
    }
}
```

**Impact:**
- Missing conformational diversity in predictions
- Reduced ability to capture alternative backbone/sidechain arrangements
- Single conformer may not represent the true structural ensemble

**Recommendation:**
- Enable multi-conformer sampling by default (3-5 conformers)
- Implement ensemble averaging in diffusion conditioning
- Add confidence-weighted conformer selection

---

### 5. Diffusion Conditioning ✅ **WORKING WELL**

**Strengths:**
- Comprehensive atom-level conditioning (position, element, charge)
- Token-level conditioning from trunk outputs
- Atom-to-token attention properly wired
- Coordinate noise and denoising implemented

**Key Implementation:**
```rust
// From diffusion.rs
pub fn forward(
    &self,
    s_inputs: &Tensor,      // Token embeddings
    s_trunk: &Tensor,       // Trunk outputs
    r_noisy: &Tensor,       // Noisy atom coordinates
    times: &Tensor,         // Noise levels
    cond: &DiffusionConditioningOutput,
    token_pad_mask: &Tensor,
    atom_pad_mask: &Tensor,
    atom_to_token: &Tensor, // ← Critical: atom-token mapping
    multiplicity: i64,
) -> Tensor
```

**Assessment:** ✅ **Strong** - Diffusion properly conditions on both atom and token features.

---

### 6. Frame Utilization ⚠️ **COULD BE IMPROVED**

**Current State:**
- Frames computed for protein residues
- Frames used for template alignment
- Frames **NOT explicitly used** in diffusion sampling

**Issue:**
Frames are computed and stored but don't appear to be used for:
- Rotational constraints during diffusion
- Local coordinate system for atom updates
- Frame-aware distance constraints

**Recommendation:**
- Use frames as local coordinate systems for atom denoising
- Apply frame-based rotational constraints
- Consider frame-consistency loss in training

---

### 7. Residue Constraints ⚠️ **PARTIALLY IMPLEMENTED**

**Current State:**
- NPZ loading for residue constraints is implemented
- Constraint tensors are generated
- Integration into the prediction pipeline is **incomplete**

**From TODO.md:**
```markdown
| [x] | Residue constraints NPZ → Rust | preprocess `main.py` |
| [~] | Residue constraint integration | Partially implemented |
```

**Recommendation:**
- Complete constraint integration in diffusion
- Implement constraint-aware sampling
- Add constraint violation penalties

---

## Critical Gaps Identified

### 🔴 High Priority

1. **Ligand Symmetry Handling**
   - **Issue:** Empty ligand symmetry data
   - **Impact:** Poor ligand conformations
   - **Fix:** Implement CCD symmetry loading
   - **Files:** `process_symmetry_features.rs`, `ccd.rs`

2. **Multi-Conformer Ensemble Sampling**
   - **Issue:** Default to single conformer
   - **Impact:** Missing conformational diversity
   - **Fix:** Enable multi-conformer by default
   - **Files:** `process_ensemble_features.rs`, `diffusion.rs`

3. **Residue Constraint Integration**
   - **Issue:** Constraints loaded but not enforced
   - **Impact:** No constraint satisfaction
   - **Fix:** Wire constraints into diffusion
   - **Files:** `process_residue_constraint_features.rs`, `diffusion.rs`

### 🟡 Medium Priority

4. **Frame-Based Local Coordinates**
   - **Issue:** Frames not used in diffusion
   - **Impact:** Less geometrically aware sampling
   - **Fix:** Use frames for local coordinate updates
   - **Files:** `diffusion.rs`, `encoders.rs`

5. **Template Force Integration**
   - **Issue:** Template bias may be weak
   - **Impact:** Reduced template influence
   - **Fix:** Strengthen template conditioning
   - **Files:** `process_template_features.rs`, `diffusion.rs`

### 🟢 Low Priority

6. **Enhanced Atom Feature Utilization**
   - **Issue:** Some atom features underused
   - **Impact:** Minor accuracy loss
   - **Fix:** Review feature importance in training
   - **Files:** `process_atom_features.rs`, `encoders.rs`

---

## What's Working Well ✅

1. **Atom Feature Engineering**
   - Comprehensive 388-dim atom features
   - Proper element, charge, chirality encoding
   - Good atom-to-token mapping

2. **Residue Tokenization**
   - Clean residue-to-token conversion
   - Proper center/disto atom identification
   - Accurate frame computation for proteins

3. **Backbone Processing**
   - Correct backbone atom indices
   - Proper backbone feature encoding
   - Accurate N-CA-C frame computation

4. **Diffusion Architecture**
   - Well-structured score model
   - Proper atom attention mechanisms
   - Good EDM sampling implementation

5. **Template Integration**
   - Template features properly generated
   - Template bias applied correctly
   - Template masking working

---

## Recommended Upgrade Plan

### Phase 1: Critical Fixes (1-2 weeks)

1. **Implement Ligand Symmetry Loading**
   ```rust
   // In process_symmetry_features.rs
   pub fn get_ligand_symmetries_for_tokens(
       tokens: &[TokenData],
       symmetries: &HashMap<String, Vec<Vec<(usize, usize)>>>,  // ← Load from CCD
   ) -> Vec<Vec<Vec<(usize, usize)>>>
   ```

2. **Enable Multi-Conformer Sampling**
   ```rust
   // In process_ensemble_features.rs
   pub fn inference_ensemble_features() -> EnsembleFeatures {
       EnsembleFeatures {
           ensemble_ref_idxs: vec![0, 1, 2, 3, 4],  // ← Use 5 conformers
       }
   }
   ```

3. **Wire Residue Constraints**
   ```rust
   // In diffusion.rs
   fn sample_inner(
       // ... existing parameters
       constraints: Option<&ResidueConstraintTensors>,  // ← Add constraints
   ) -> DiffusionSampleOutput
   ```

### Phase 2: Enhancements (2-3 weeks)

4. **Frame-Based Local Coordinates**
   - Implement frame-aware diffusion updates
   - Add frame consistency checks
   - Use frames for rotational constraints

5. **Strengthen Template Integration**
   - Increase template conditioning weight
   - Add template-force features to diffusion
   - Implement template-guided sampling

### Phase 3: Validation (1 week)

6. **Testing and Validation**
   - Create test cases for each enhancement
   - Run golden tensor regression
   - Measure prediction accuracy improvements

---

## Conclusion

The Boltr folding prediction system has a **solid foundation** with comprehensive atom and residue processing. The main areas for improvement are:

1. **Completing** partially implemented features (constraints, ligand symmetries)
2. **Enabling** multi-conformer sampling by default
3. **Better utilizing** computed geometric features (frames)

The system is "working well" as stated, but these enhancements would significantly improve prediction accuracy, especially for:
- Ligand-containing complexes
- Multi-domain proteins
- Systems with conformational flexibility

**Overall Assessment:** 7.5/10 - Strong foundation with clear upgrade path to 9/10.

---

## Files Requiring Changes

### High Priority
- `boltr-io/src/featurizer/process_symmetry_features.rs`
- `boltr-io/src/featurizer/process_ensemble_features.rs`
- `boltr-io/src/featurizer/process_residue_constraint_features.rs`
- `boltr-backend-tch/src/boltz2/diffusion.rs`

### Medium Priority
- `boltr-backend-tch/src/boltz2/encoders.rs`
- `boltr-io/src/featurizer/process_template_features.rs`
- `boltr-io/src/structure_v2.rs` (for frame methods)

### Low Priority
- `boltr-io/src/featurizer/process_atom_features.rs` (review usage)
- `boltr-backend-tch/src/boltz2/diffusion_conditioning.rs`

---

## Testing Strategy

1. **Unit Tests**
   - Test ligand symmetry loading
   - Test multi-conformer feature generation
   - Test constraint application

2. **Integration Tests**
   - Test end-to-end prediction with ligands
   - Test multi-conformer diffusion
   - Test constraint satisfaction

3. **Golden Tests**
   - Export Python tensors for enhanced features
   - Compare Rust vs Python outputs
   - Validate numerical parity

4. **Accuracy Tests**
   - Measure RMSD improvements
   - Test on benchmark complexes
   - Compare to baseline predictions
