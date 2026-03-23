//! Boltz2 `TemplateV2Module` (`modules/trunkv2.py`) — **stub** until template pairformer is ported.
//!
//! Python adds a template-derived bias to `z` before recycling / pairformer iterations.

use tch::Tensor;

#[derive(Debug, Default, Clone, Copy)]
pub struct TemplateModule;

impl TemplateModule {
    /// Placeholder: return `z` unchanged.
    #[must_use]
    pub fn forward_trunk_step(&self, z: &Tensor) -> Tensor {
        z.shallow_clone()
    }
}
