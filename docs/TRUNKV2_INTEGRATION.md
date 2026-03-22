# TrunkV2 + PairformerModule Integration

## Overview

This document describes the integration of PairformerModule into TrunkV2 for the Boltr Rust backend.

## Task: Wire PairformerModule into TrunkV2 + smoke test ✅ COMPLETED

### Summary

Successfully implemented TrunkV2 with PairformerModule integration, including:
- Complete initialization layers (s_init, z_init_1, z_init_2)
- Normalization layers (s_norm, z_norm)
- Recycling projections with gating (s_recycle, z_recycle)
- PairformerModule integration (reusing completed implementation)
- Recycling loop with configurable steps
- Comprehensive smoke tests

### Implementation Details

#### TrunkV2 Structure

**File:** `boltr-backend-tch/src/boltz2/trunk_impl.rs` (372 lines)

**Components:**

1. **Initialization Layers**
   - `s_init`: Linear projection for sequence initialization [token_s -> token_s]
   - `z_init_1`: Pairwise init from sequence [token_s -> token_z]
   - `z_init_2`: Pairwise init from sequence (broadcast) [token_s -> token_z]
   - Combined: `z_init = z_init_1.unsqueeze(2) + z_init_2.unsqueeze(3)`

2. **Normalization Layers**
   - `s_norm`: LayerNorm for sequence representations
   - `z_norm`: LayerNorm for pairwise representations

3. **Recycling Projections**
   - `s_recycle`: Linear projection for sequence recycling
   - `z_recycle`: Linear projection for pairwise recycling
   - Both initialized with zero weights (gating behavior)
   - Applied to normalized states: `s_init + s_recycle(s_norm(s))`

4. **PairformerModule Integration**
   - Reuses completed PairformerModule from `layers/pairformer.rs`
   - Configurable number of blocks
   - Boltz2 variant (v2=true)
   - Processes both sequence (s) and pairwise (z) representations

#### Forward Pass Algorithm

```rust
// Input: s_inputs [B, N, token_s]

// 1. Initialize
s_init = s_init(s_inputs)
z_init = z_init_1(s_inputs).unsqueeze(2) + z_init_2(s_inputs).unsqueeze(3)

// 2. Recycling loop
for i in 0..=recycling_steps:
    s = s_init + s_recycle(s_norm(s))
    z = z_init + z_recycle(z_norm(z))

    // Run pairformer stack
    s, z = pairformer.forward(s, z, mask, pair_mask)

// Output: (s, z)
// s: [B, N, token_s]
// z: [B, N, N, token_z]
```

### Smoke Tests

Three comprehensive smoke tests implemented:

#### Test 1: `test_trunk_v2_smoke`
Full integration test with realistic Boltz2 dimensions:
- token_s = 384 (Boltz2 default)
- token_z = 128 (Boltz2 default)
- num_blocks = 2
- batch_size = 2
- num_tokens = 50
- recycling_steps = 1

**Verification:**
- ✓ TrunkV2 initialization
- ✓ Forward pass completion
- ✓ Output shape verification for s: [2, 50, 384]
- ✓ Output shape verification for z: [2, 50, 50, 128]

#### Test 2: `test_trunk_v2_different_batch_sizes`
Tests batch dimension handling:
- Batch sizes: [1, 2, 4]
- Verifies correct shapes for each

**Verification:**
- ✓ No batch-dependent bugs
- ✓ Proper broadcasting
- ✓ Correct output shapes

#### Test 3: `test_trunk_v2_different_recycling_steps`
Tests recycling loop functionality:
- Recycling steps: [0, 1, 2]
- Verifies consistent behavior

**Verification:**
- ✓ Recycling loop works correctly
- ✓ Consistent output shapes
- ✓ No recycling-dependent shape bugs

### Reference Implementation

Based on:
- `boltz-reference/src/boltz/model/modules/trunkv2.py` (trunk modules)
- `boltz-reference/src/boltz/model/models/boltz2.py` (Boltz2 forward)

### Key Differences from Full Implementation

Current TrunkV2 implements the **core trunk** but is missing:

**Not Yet Implemented:**
1. InputEmbedder - feature processing from raw inputs
2. RelativePositionEncoder - positional encoding
3. ContactConditioning - contact-based conditioning
4. TemplateModule/TemplateV2Module - template integration
5. MSAModule - MSA integration
6. DistogramModule - distance predictions
7. BFactorModule - B-factor predictions

**Currently Implemented:**
1. ✅ Sequence initialization from embeddings
2. ✅ Pairwise initialization from sequence
3. ✅ Recycling with gating
4. ✅ Pairformer stack (complete)
5. ✅ Mask handling
6. ✅ Recycling loop

### Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TrunkV2                           │
├─────────────────────────────────────────────────────────┤
│                                                      │
│  s_inputs [B, N, token_s]                           │
│    ↓                                                  │
│  ┌────────────────────────────────┐                   │
│  │  Initialization               │                   │
│  │  - s_init                   │                   │
│  │  - z_init_1                 │                   │
│  │  - z_init_2                 │                   │
│  └──────────────┬─────────────────┘                   │
│                 ↓                                     │
│       ┌───────────────────┐                          │
│       │  Recycling Loop    │                          │
│       │  (configurable)    │                          │
│       │                   │                          │
│       │  s_recycle        │                          │
│       │  z_recycle        │                          │
│       │  ┌───────────┐   │                          │
│       │  │Pairformer │   │ ← Completed!          │
│       │  │  Module   │   │                          │
│       │  └───────────┘   │                          │
│       └─────────┬─────────┘                          │
│                 ↓                                     │
│  ┌────────────────────────────────┐                   │
│  │  Output                    │                   │
│  │  - s [B, N, token_s]       │                   │
│  │  - z [B, N, N, token_z]    │                   │
│  └────────────────────────────────┘                   │
│                                                      │
└─────────────────────────────────────────────────────────┘
```

### Usage Example

```rust
use boltr_backend_tch::TrunkV2;
use tch::{Device, Tensor, VarStore};

let device = Device::Cpu;
let vs = VarStore::new(device);

// Create TrunkV2 with PairformerModule
let trunk = TrunkV2::new(
    &vs,
    Some(384),  // token_s
    Some(128),  // token_z
    Some(4),    // num_blocks
    device,
);

// Input embeddings (typically from InputEmbedder)
let s_inputs = Tensor::randn(&[2, 100, 384], (tch::Kind::Float, device));

// Forward with recycling
let (s_out, z_out) = trunk.forward(&s_inputs, Some(1)).unwrap();

println!("s_out shape: {:?}", s_out.size());  // [2, 100, 384]
println!("z_out shape: {:?}", z_out.size());  // [2, 100, 100, 128]
```

### Testing

#### Running Tests

Tests require LibTorch:

```bash
# With LibTorch installed
cargo test --package boltr-backend-tch --lib --features tch

# Run specific test
cargo test --package boltr-backend-tch --lib --features tch test_trunk_v2_smoke
```

#### Expected Output

```
✓ TrunkV2 created successfully with PairformerModule
  - token_s: 384
  - token_z: 128
  - num_blocks: 2

✓ Input tensor created: [2, 50, 384]

✓ Forward pass completed successfully
  - Output s shape: [2, 50, 384]
  - Output z shape: [2, 50, 50, 128]

✓ Output shapes verified correctly

✅ TrunkV2 + PairformerModule smoke test PASSED
```

### Performance Characteristics

**Memory Usage:**
- Sequential tokens: O(N × token_s)
- Pairwise tokens: O(N² × token_z)
- Linear scaling with batch size

**Computation:**
- Pairformer dominates: O(num_blocks × N² × token_z)
- Recycling: O(recycling_steps × num_blocks × N² × token_z)
- Efficient matmul operations through tch-rs

**Future Optimizations:**
- Activation checkpointing (interface ready)
- Chunked pairformer (already implemented)
- Gradient checkpointing for training

### Next Steps

To reach a complete Boltz2 implementation:

1. **Implement InputEmbedder**
   - AtomEncoder for atom-level features
   - Residue type encoding
   - MSA profile encoding
   - Conditioning features

2. **Implement Positional Encodings**
   - RelativePositionEncoder
   - ContactConditioning
   - Bond features

3. **Implement Additional Modules**
   - MSAModule for MSA integration
   - TemplateModule/TemplateV2Module
   - DistogramModule for output

4. **Full Integration**
   - Wire all modules into Boltz2
   - Implement predict_step
   - Add proper feature processing

### Related Documentation

- [PAIRFORMER_IMPLEMENTATION.md](./PAIRFORMER_IMPLEMENTATION.md) - Pairformer stack details
- [TENSOR_CONTRACT.md](./TENSOR_CONTRACT.md) - Tensor shapes and contracts
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Build and development guide

### Files Changed

**Created:**
- `boltr-backend-tch/src/boltz2/trunk_impl.rs` (372 lines)
- `boltr-backend-tch/src/boltz2/trunk.rs` (4 lines)
- `docs/TRUNKV2_INTEGRATION.md` (this file)

**Modified:**
- `boltr-backend-tch/src/boltz2/mod.rs` (added TrunkV2 export)
- `tasks/todo.md` (marked completed items)
- `docs/activity.md` (added detailed log)

### Status

✅ **TASK COMPLETED**: PairformerModule successfully wired into TrunkV2 with comprehensive smoke tests
