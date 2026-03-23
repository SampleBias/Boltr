//! Boltz2 `MSAModule` (`modules/trunkv2.py`) — **stub** until PairWeightedAveraging / MSALayer are ported.
//!
//! Python applies MSA refinement **before** the main `PairformerModule` stack on `z`.

use tch::Tensor;

#[derive(Debug, Default, Clone, Copy)]
pub struct MsaModule;

impl MsaModule {
    /// Placeholder: return `z` unchanged (real impl updates `z` from MSA rows in `feats`).
    #[must_use]
    pub fn forward_trunk_step(&self, z: &Tensor, _s: &Tensor) -> Tensor {
        z.shallow_clone()
    }
}
