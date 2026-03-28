# Pairformer Stack Implementation

## Overview

This document describes the implementation of the Pairformer stack in the Boltr Rust backend. The Pairformer is a core component of the Boltz2 model that processes both sequence (s) and pairwise (z) representations through multiple layers of attention and triangular operations.

## Implementation Status

### Completed Components ✅

All core Pairformer components have been implemented following the Python reference implementation:

1. **AttentionPairBiasV2** (`boltr-backend-tch/src/attention/pair_bias.rs`)
   - Multi-head attention with pairwise bias
   - Supports LayerNorm and gating
   - Implements Boltz2 variant
   - Reference: `boltz-reference/src/boltz/model/layers/attentionv2.py`

2. **TriangleMultiplicationOutgoing** (`boltr-backend-tch/src/layers/triangular_mult.rs`)
   - Triangle multiplication with outgoing edges
   - Uses einsum pattern: `bikd,bjkd->bijd`
   - Fallback PyTorch path (no cuequivariance)
   - Reference: `boltz-reference/src/boltz/model/layers/triangular_mult.py`

3. **TriangleMultiplicationIncoming** (`boltr-backend-tch/src/layers/triangular_mult.rs`)
   - Triangle multiplication with incoming edges
   - Uses einsum pattern: `bkid,bkjd->bijd`
   - Fallback PyTorch path (no cuequivariance)
   - Reference: `boltz-reference/src/boltz/model/layers/triangular_mult.py`

4. **TriangleAttentionStartingNode** (`boltr-backend-tch/src/layers/triangular_attention.rs`)
   - Multi-head attention on pairwise representation
   - Starting node mode (Algorithm 13)
   - Reference: `boltz-reference/src/boltz/model/layers/triangular_attention/attention.py`

5. **TriangleAttentionEndingNode** (`boltr-backend-tch/src/layers/triangular_attention.rs`)
   - Multi-head attention on pairwise representation
   - Ending node mode (Algorithm 14)
   - Reference: `boltz-reference/src/boltz/model/layers/triangular_attention/attention.py`

6. **Transition** (`boltr-backend-tch/src/layers/transition.rs`)
   - Two-layer MLP with SwiGLU activation
   - LayerNorm -> SiLU(fc1(x)) * fc2(x) -> fc3
   - Reference: `boltz-reference/src/boltz/model/layers/transition.py`

7. **OuterProductMean** (`boltr-backend-tch/src/layers/outer_product_mean.rs`)
   - Outer product computation for pairwise interactions
   - Reference: `boltz-reference/src/boltz/model/layers/outer_product_mean.py`

8. **PairformerLayer** (`boltr-backend-tch/src/layers/pairformer.rs`)
   - Complete single layer combining:
     - Pairwise stack: tri_mul_out → tri_mul_in → tri_att_start → tri_att_end → transition_z
     - Sequence stack: attention_pair_bias → transition_s → post_norm
   - Supports dropout and chunking
   - Reference: `boltz-reference/src/boltz/model/layers/pairformer.py`

9. **PairformerModule** (`boltr-backend-tch/src/layers/pairformer.rs`)
   - Stack of multiple PairformerLayers
   - Handles activation checkpointing interface
   - Dynamic chunk size selection
   - Reference: `boltz-reference/src/boltz/model/layers/pairformer.py`

## Key Design Decisions

### 1. Training Mode Support
The implementation includes proper training/evaluation mode handling:

- **Training mode** (`training=true`): Dropout is applied with scaling
- **Evaluation mode** (`training=false`): Dropout is disabled for deterministic outputs

This matches Python's `get_dropout_mask()` behavior from `boltz-reference/src/boltz/model/layers/dropout.py`:

```python
def get_dropout_mask(dropout, z, training, columnwise=False):
    dropout = dropout * training  # Zero during evaluation
    # ... mask generation
    return d
```

**Key Implementation Details:**

1. **Slice-based Mask Generation**: Uses small subsample `z[:, :, 0:1, 0:1]` (non-columnwise) or `z[:, 0:1, :, 0:1]` (columnwise), matching Python's approach for efficiency.

2. **Correct Comparison**: Uses `>=` comparison (not `>`) to match Python exactly.

3. **Chunking Logic**: During training, `chunk_size_tri_attn = None` (no chunking); during evaluation, uses threshold-based chunking (128 for seq_len > 256, else 512).

4. **API Support**: Both `PairformerModule` and `TrunkV2` provide `set_training()` methods for easy mode switching.

**Usage:**

```rust
// Inference (default)
let mut model = PairformerModule::new(/*...*/);
model.set_training(false);  // Already default
let (s, z) = model.forward(&s_in, &z_in, &mask, &pair_mask, false);
// Result: Deterministic (no dropout)

// Training
let mut model = PairformerModule::new(/*...*/);
model.set_training(true);  // Enable dropout
let (s, z) = model.forward(&s_in, &z_in, &mask, &pair_mask, false);
// Result: Stochastic (with dropout)
```

See [PAIRFORMER_DROPOUT_FIX.md](./PAIRFORMER_DROPOUT_FIX.md) for detailed implementation.

### 2. Feature-Gated Implementation
All tch-rs dependent code is behind the `tch-backend` feature flag:
- Allows building without LibTorch for CI/testing
- Maintains clean separation of concerns
- Matches workspace design

### 2. Fallback Path Only
We implement only the PyTorch fallback path (not cuequivariance kernels):
- **Reason**: tch-rs doesn't support cuEquivariance kernels
- **Equivalence**: Matches `use_kernels=False` in Python
- **Numerics**: Same math, just different implementation
- **Performance**: Still benefits from CUDA LibTorch for standard ops

### 3. Precision Handling
Explicit dtype handling matches Python's autocast behavior:
- Attention computations use float32 explicitly
- Restores original dtype after operations
- Matches `torch.autocast("cuda", enabled=False)` in Python

### 4. Module Structure
Organized into subdirectories for clarity:
```
boltr-backend-tch/src/
├── attention/
│   ├── mod.rs
│   └── pair_bias.rs
└── layers/
    ├── mod.rs
    ├── pairformer.rs
    ├── transition.rs
    ├── triangular_attention.rs
    ├── triangular_mult.rs
    └── outer_product_mean.rs
```

## Testing

### Unit Tests
Each component includes unit tests:
- Test forward pass with dummy inputs
- Verify output shapes
- Test with/without bias where applicable

### Running Tests
```bash
# Without LibTorch (compiles but tests don't run)
cargo test --package boltr-backend-tch

# With LibTorch (full test execution)
cargo test --package boltr-backend-tch --features tch
```

### Golden Testing (Future)
Planned integration with golden fixtures:
- Export tensors from Python for test inputs
- Compare outputs with numerical tolerance
- Validate numerical parity

## Usage Example

```rust
use boltr_backend_tch::{PairformerModule, Transition, AttentionPairBiasV2};
use tch::{Device, VarStore, Tensor};

let device = Device::Cpu;
let vs = VarStore::new(device);

// Create a PairformerModule
let pairformer = PairformerModule::new(
    &vs,
    384,  // token_s
    128,  // token_z
    4,    // num_blocks
    Some(16),  // num_heads
    None,  // dropout (default 0.25)
    None,  // pairwise_head_width (default 32)
    None,  // pairwise_num_heads (default 4)
    None,  // post_layer_norm (default false)
    None,  // activation_checkpointing (default false)
    Some(true),  // v2 (use Boltz2 variant)
    device,
);

// Forward pass
let s = Tensor::randn(&[2, 100, 384], (tch::Kind::Float, device));
let z = Tensor::randn(&[2, 100, 100, 128], (tch::Kind::Float, device));
let mask = Tensor::ones(&[2, 100, 100], (tch::Kind::Float, device));
let pair_mask = Tensor::ones(&[2, 100, 100], (tch::Kind::Float, device));

let (s_out, z_out) = pairformer.forward(&s, &z, &mask, &pair_mask, false);
```

## Next Steps

### Priority 1: Complete Boltz2 Model
- [ ] Implement InputEmbedder
- [ ] Implement RelativePositionEncoder
- [ ] Implement DiffusionConditioning
- [ ] Implement AtomDiffusion v2
- [ ] Implement ConfidenceModule v2
- [ ] Implement AffinityModule
- [ ] Wire full Boltz2 forward pass

### Priority 2: IO & Preprocessing
- [ ] Implement Tokenizer
- [ ] Implement Featurizer
- [ ] Implement collate and dataset loading
- [ ] Implement output writers

### Priority 3: Testing & Validation
- [ ] Create golden fixture repository
- [ ] Write Python export scripts
- [ ] Implement numerical parity tests
- [ ] End-to-end integration tests

## Dependencies

### Rust Crates
- `tch` (optional): PyTorch bindings for tensor operations
- `anyhow`: Error handling
- `thiserror`: Error types
- `tracing`: Logging

### Python Reference
The implementation is based on these Python files:
- `boltz-reference/src/boltz/model/layers/attentionv2.py`
- `boltz-reference/src/boltz/model/layers/pairformer.py`
- `boltz-reference/src/boltz/model/layers/triangular_mult.py`
- `boltz-reference/src/boltz/model/layers/triangular_attention/attention.py`
- `boltz-reference/src/boltz/model/layers/transition.py`
- `boltz-reference/src/boltz/model/layers/outer_product_mean.py`

## Performance Considerations

### Memory Usage
- Chunking support for large sequences (>256 tokens)
- Gradient checkpointing interface (to be implemented)
- Efficient tensor operations through tch-rs

### GPU Acceleration
- All operations support CUDA through LibTorch
- No custom kernels needed (uses standard PyTorch ops)
- Batch processing through matmul operations

### Future Optimizations
- Implement actual activation checkpointing
- Optimize einsum patterns for specific hardware
- Consider batch size tuning for different GPUs

## Known Limitations

1. **Chunked Attention**: Triangular attention chunking interface exists but uses simple implementation
2. **Dropout**: Dropout masks are basic, could be optimized
3. **Columnwise Dropout**: Simplified implementation for ending node
4. **Activation Checkpointing**: Interface exists but not yet implemented
5. **Autocast**: Manual dtype handling instead of true autocast

## References

- [Boltz2 Paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707)
- [Boltz Reference Implementation](https://github.com/jwohlwend/boltz)
- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
- [TENSOR_CONTRACT.md](./TENSOR_CONTRACT.md)
- [DEVELOPMENT.md](../DEVELOPMENT.md)
