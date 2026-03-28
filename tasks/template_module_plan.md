# TemplateModule Implementation Plan

## Overview
Implement `TemplateV2Module` for Boltz2 to add template-derived bias to pairwise embeddings (z).

## Current State
- ✅ Stub implementation exists (returns z unchanged)
- ❌ No real template processing
- ❌ Missing distogram computation
- ❌ Missing unit vector computation
- ❌ Missing pairformer stack for templates

## Python Reference
From `boltz-reference/src/boltz/model/modules/trunkv2.py`:

### TemplateV2Module Structure:
1. **Input Features:**
   - `template_restype`: Residue types per template
   - `template_frame_rot`: Rotation matrices
   - `template_frame_t`: Translation vectors
   - `template_mask_frame`: Valid frame masks
   - `template_cb`: CB coordinates
   - `template_ca`: CA coordinates
   - `template_mask_cb`: Valid CB masks
   - `visibility_ids`: For template pairing
   - `template_mask`: Overall mask

2. **Processing Steps:**
   - Compute distogram from CB-CB distances
   - Compute unit vectors from frames and CA coordinates
   - Concatenate: distogram + masks + unit vectors
   - Add residue type encodings
   - Project to template_dim
   - Add to z (broadcast over templates)
   - Process through PairformerNoSeqModule
   - Aggregate templates (weighted average)
   - Project back to token_z

3. **Parameters:**
   - token_z: Pairwise embedding dimension
   - template_dim: Internal template dimension
   - template_blocks: Number of pairformer blocks
   - dropout: Dropout rate
   - pairwise_head_width: Triangular attention head width
   - pairwise_num_heads: Number of triangular attention heads
   - post_layer_norm: Whether to use post layer norm
   - activation_checkpointing: Whether to use activation checkpointing
   - min_dist, max_dist, num_bins: Distogram binning

## Implementation Tasks

### Phase 1: Data Structures
- [ ] Create TemplateFeatures struct for input features
- [ ] Create TemplateV2Module struct with all components

### Phase 2: Component Initialization
- [ ] z_proj: Linear(token_z → template_dim)
- [ ] a_proj: Linear(feature_dim → template_dim)
- [ ] u_proj: Linear(template_dim → token_z)
- [ ] pairformer: PairformerNoSeqModule
- [ ] LayerNorm for z and v

### Phase 3: Core Operations
- [ ] compute_distogram: CB distance binning
- [ ] compute_unit_vectors: Frame rotation + translation
- [ ] build_template_features: Concatenation
- [ ] aggregate_templates: Weighted average

### Phase 4: Forward Pass
- [ ] Extract and validate features
- [ ] Compute masks (pairwise, template, asym)
- [ ] Compute template features
- [ ] Project and add to z
- [ ] Process through pairformer
- [ ] Aggregate and project output

### Phase 5: Testing
- [ ] Unit test with dummy data
- [ ] Test feature shapes
- [ ] Test forward pass
- [ ] Test with/without templates

## Key Challenges

1. **Tensor shapes**: Need careful handling of [B, N, N, D] with [B, T, N, ...]
2. **Broadcasting**: Proper expansion over templates dimension
3. **Masking**: Correct pairwise and template masks
4. **Autocast**: Disable for coordinate computations

## Dependencies
- ✅ PairformerNoSeqModule (already implemented)
- ✅ Linear projections (tch-rs)
- ✅ LayerNorm (tch-rs)
- ✅ Tensor operations (tch-rs)

## Success Criteria
- [ ] Compiles without errors
- [ ] Passes all unit tests
- [ ] Matches Python output shapes
- [ ] Integrates with TrunkV2
- [ ] No regressions in existing tests
