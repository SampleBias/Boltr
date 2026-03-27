//! Structure module: `AtomDiffusion` / score model from `boltz.model.modules.diffusionv2`.
//!
//! **Roadmap (TODO.md §5.6):** port `DiffusionConditioning`, `AtomDiffusion`, score/transformers v2,
//! distogram head, optional B-factor; golden a single reverse-diffusion step vs Python on a tiny batch.

/// Placeholder for the diffusion sampler and score network.
#[derive(Debug, Default)]
pub struct AtomDiffusionV2;

impl AtomDiffusionV2 {
    pub fn new() -> Self {
        Self
    }
}
