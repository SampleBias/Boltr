# TrunkV2 - Owns PairformerModule, Exposes Clean API

## Overview

**Refactored TrunkV2** to own a PairformerModule and expose a clean API so other components (MSA, templates, embeddings) can connect easily without rewriting structure.

## Key Design Principles

### 1. TrunkV2 Owns PairformerModule
- TrunkV2 **owns** a PairformerModule (not a separate layer)
- PairformerModule is initialized as part of TrunkV2
- No need for external code to create/manage PairformerModule

### 2. Exposes `forward_pairformer(s, z, ...)` API
```rust
pub fn forward_pairformer(
    &self,
    s: &Tensor,
    z: &Tensor,
    mask: &Tensor,
    pair_mask: &Tensor,
) -> (Tensor, Tensor)
```
This is the **key API** that other components use.

### 3. Easy Component Connection
Other components (MSA, templates, embeddings) can:
1. Get (s, z) from `trunk.initialize(s_inputs)`
2. Add their contributions to z
3. Call `trunk.forward_pairformer(s, z, mask, pair_mask)`
4. Get updated (s, z) - **no structure rewriting needed!**

## Implementation

### File: `boltr-backend-tch/src/boltz2/trunk.rs` (450+ lines)

### Public API

```rust
impl TrunkV2 {
    // Create new TrunkV2 with owned PairformerModule
    pub fn new(vs, token_s, token_z, num_blocks, device) -> Self

    // Initialize s and z from input embeddings
    pub fn initialize(&self, s_inputs: &Tensor) -> (Tensor, Tensor)

    // Apply recycling projections
    pub fn apply_recycling(&self, s_init, z_init, s_prev, z_prev) -> (Tensor, Tensor)

    // ⭐ KEY API: Forward through owned PairformerModule
    pub fn forward_pairformer(&self, s, z, mask, pair_mask) -> (Tensor, Tensor)

    // Full forward with recycling loop
    pub fn forward(&self, s_inputs, recycling_steps) -> anyhow::Result<(Tensor, Tensor)>

    // Accessors for owned PairformerModule
    pub fn pairformer_mut(&mut self) -> &mut PairformerModule
    pub fn pairformer(&self) -> &PairformerModule

    // Dimension accessors
    pub fn token_s(&self) -> i64
    pub fn token_z(&self) -> i64
}
```

## Usage Example: Connecting MSA Module

```rust
// After implementing MSA module, here's how it connects:

let trunk = TrunkV2::new(&vs, Some(384), Some(128), Some(4), device);
let s_inputs = Tensor::randn(&[2, 100, 384], (Kind::Float, device));

// Initialize
let (s, z) = trunk.initialize(&s_inputs);

// ⭐ MSA module connects here easily:
let mask = Tensor::ones(&[2, 100, 100], (Kind::Float, device));
let pair_mask = Tensor::ones(&[2, 100, 100], (Kind::Float, device));

// MSA module adds to z without rewriting structure!
let z = z + msa_module.forward(&z, &s, &feats, &mask, &pair_mask);

// Run pairformer (owned by trunk)
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);

// MSA can connect again if needed
let z = z + msa_module.forward_2(&z, ...);
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);
```

## Same Pattern for Other Components

### Template Module
```rust
let (s, z) = trunk.initialize(&s_inputs);
z = z + template_module.forward(&z, &feats, &pair_mask);
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);
```

### Embedding Components
```rust
let (s, z) = trunk.initialize(&s_inputs);
z = z + relative_position_encoder.forward(&feats);
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);
```

## Tests

### Test 1: `test_trunk_owns_pairformer`
Verifies TrunkV2 owns PairformerModule and exposes proper API:
- ✓ TrunkV2 owns PairformerModule
- ✓ Can access pairformer via `pairformer()` method
- ✓ `initialize()` produces correct shapes
- ✓ `apply_recycling()` produces correct shapes
- ✓ `forward_pairformer(s, z, ...)` works correctly

### Test 2: `test_trunk_api_for_component_connection`
Demonstrates how other components connect easily:
```rust
let (s, z) = trunk.initialize(&s_inputs);

// Simulate MSA adding to z (easy connection!)
let msa_contribution = Tensor::randn(...);
let z = z + msa_contribution * 0.1;

// Call pairformer (clean API!)
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);
```

### Test 3: `test_trunk_full_forward`
Tests complete forward pass with recycling.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              TrunkV2 (owns PairformerModule)      │
├─────────────────────────────────────────────────────────┤
│                                                       │
│  s_inputs [B, N, token_s]                           │
│    ↓                                                  │
│  ┌────────────────────────────────┐                   │
│  │  initialize()               │                   │
│  │  ↓                         │                   │
│  │  (s_init, z_init)          │                   │
│  └──────────┬─────────────────────┘                   │
│             ↓                                        │
│  ┌────────────────────────────────┐                   │
│  │  Recycling Loop    │                   │
│  │  (configurable)    │                   │
│  │                   │                          │
│  │  apply_recycling()         │                   │
│  │       ↓            │                          │
│  │  (s, z)          │                   │
│  │       │            │                          │
│  │  ┌─────┴────┐    │                          │
│  │  │ Other     │    │                          │
│  │  │ Components│    │ ← Easy connection point!        │
│  │  │ (MSA,     │    │                          │
│  │  │ Templates)│    │                          │
│  │  │    ↓      │    │                          │
│  │  │  z += ... │    │                          │
│  │  └─────┬────┘    │                          │
│  │       ↓            │                          │
│  │  forward_pairformer(s, z, ...) ← KEY API!     │
│  │       ↓            │                          │
│  │  (s, z)          │                   │
│  └──────────┬─────────────┘                   │
│             ↓                                        │
│  (final_s, final_z)                                 │
│                                                       │
└─────────────────────────────────────────────────────────┘
```

## Benefits of This Design

### 1. Clear Ownership
- TrunkV2 owns PairformerModule
- Single VarStore hierarchy
- No circular dependencies

### 2. Clean API
- `forward_pairformer(s, z, ...)` is straightforward
- Other components don't need to understand Pairformer internals
- Just pass (s, z) and get updated (s, z)

### 3. Easy to Extend
- MSA module: `z += msa.forward(...); trunk.forward_pairformer(s, z, ...)`
- Template module: `z += template.forward(...); trunk.forward_pairformer(s, z, ...)`
- Positional encodings: `z += pos_enc.forward(...); trunk.forward_pairformer(s, z, ...)`

### 4. No Structure Rewriting
- Other components just add to z
- Don't need to modify TrunkV2 structure
- Don't need to recreate PairformerModule

### 5. Testable in Isolation
- `forward_pairformer` can be tested independently
- Component connection can be tested with mocks
- Clear separation of concerns

## Comparison: Old vs New Design

### Old Design (Standalone Layer)
```rust
// PairformerModule is separate
let pairformer = PairformerModule::new(...);
let trunk = TrunkV2::new(..., pairformer);  // Pass in pairformer

// Other components need to create/use pairformer too
// Messy ownership structure!
```

### New Design (Owned Component)
```rust
// TrunkV2 owns PairformerModule
let trunk = TrunkV2::new(&vs, ...);

// Other components use simple API
let (s, z) = trunk.initialize(&s_inputs);
z = z + msa_module.forward(...);  // Easy!
let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);  // Clean API!
```

## Implementation Details

### Owned PairformerModule
```rust
pub struct TrunkV2 {
    // ... other fields ...

    // ⭐ Owned PairformerModule (not separate)
    pairformer: PairformerModule,

    device: Device,
}

impl TrunkV2 {
    pub fn new(vs: &VarStore, ...) -> Self {
        let pairformer_vs = root.sub("pairformer");
        let pairformer = PairformerModule::new(
            &pairformer_vs.fork(),
            token_s,
            token_z,
            num_blocks,
            // ... config
        );

        Self {
            // ... other fields ...
            pairformer,  // ← Owned!
            device,
        }
    }
}
```

### Exposed API
```rust
impl TrunkV2 {
    // ⭐ KEY API for other components
    pub fn forward_pairformer(
        &self,
        s: &Tensor,
        z: &Tensor,
        mask: &Tensor,
        pair_mask: &Tensor,
    ) -> (Tensor, Tensor) {
        // Delegate to owned PairformerModule
        self.pairformer.forward(s, z, mask, pair_mask, false)
    }

    // Also provide access if needed
    pub fn pairformer(&self) -> &PairformerModule {
        &self.pairformer
    }

    pub fn pairformer_mut(&mut self) -> &mut PairformerModule {
        &mut self.pairformer
    }
}
```

## Related Documentation

- [PAIRFORMER_IMPLEMENTATION.md](./PAIRFORMER_IMPLEMENTATION.md) - PairformerModule details
- [TENSOR_CONTRACT.md](./TENSOR_CONTRACT.md) - Tensor shapes
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Build guide

## Status

✅ **TASK COMPLETED**: TrunkV2 refactored to own PairformerModule and expose clean API

**Key Achievement:** Other components (MSA, templates, embeddings) can now connect easily via `trunk.forward_pairformer(s, z, ...)` without rewriting structure.
