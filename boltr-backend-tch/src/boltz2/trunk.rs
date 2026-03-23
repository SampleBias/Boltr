//! Trunk slice for Boltz2: init + recycling + `PairformerModule`.
//!
//! **Python alignment:** `s_init`, `z_init_*`, norms, recycling, and `pairformer_module` live on the
//! **Boltz2 model root** in `boltz2.py` (not inside a `TrunkV2` class). This struct groups that
//! subgraph for Rust ergonomics; [`VarStore`](tch::nn::VarStore) names match Lightning `state_dict`
//! when using the same root (see `pairformer_module.layers.0.…`).
//!
//! Reference: `boltz-reference/src/boltz/model/models/boltz2.py` (trunk loop) and
//! `modules/trunkv2.py` (embedder/MSA/template building blocks).

use tch::nn::{linear, LayerNorm, LinearConfig, Module, VarStore};
use tch::{Device, Tensor};

use crate::layers::PairformerModule;

use super::msa_module::{MsaFeatures, MsaModule};
use super::template_module::TemplateModule;

/// TrunkV2 - Main trunk that owns PairformerModule
///
/// The trunk processes sequence (s) and pairwise (z) representations through:
/// - Initialization from input embeddings
/// - Recycling projections
/// - PairformerModule (owned internally)
///
/// Key design: Exposes `forward_pairformer(s, z, ...)` → (s, z) so other components
/// can connect (MSA, templates, embeddings) without rewriting structure.
pub struct TrunkV2 {
    token_s: i64,
    token_z: i64,

    // Initialization layers (for first iteration)
    s_init: tch::nn::Linear,
    z_init_1: tch::nn::Linear,
    z_init_2: tch::nn::Linear,

    // Normalization layers
    s_norm: tch::nn::LayerNorm,
    z_norm: tch::nn::LayerNorm,

    // Recycling projections
    s_recycle: tch::nn::Linear,
    z_recycle: tch::nn::Linear,

    // Owned PairformerModule
    pairformer: PairformerModule,

    /// Pre-pairformer MSA path (`msa_module` in Lightning).
    msa: MsaModule,
    /// Template bias on `z` (stub until `TemplateV2Module` port).
    template: TemplateModule,

    device: Device,
}

impl TrunkV2 {
    /// Create a new TrunkV2 that owns a PairformerModule
    ///
    /// # Arguments
    ///
    /// * `vs` - Variable store for parameter storage
    /// * `token_s` - Sequence embedding dimension (default 384)
    /// * `token_z` - Pairwise embedding dimension (default 128)
    /// * `num_blocks` - Number of pairformer blocks (default 4)
    /// * `device` - Computation device
    pub fn new(
        vs: &VarStore,
        token_s: Option<i64>,
        token_z: Option<i64>,
        num_blocks: Option<i64>,
        device: Device,
    ) -> Self {
        let token_s = token_s.unwrap_or(384);
        let token_z = token_z.unwrap_or(128);
        let num_blocks = num_blocks.unwrap_or(4);

        let root = vs.root();

        // Initialization layers (for first iteration)
        let s_init = linear(
            root.sub("s_init"),
            token_s,
            token_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let z_init_1 = linear(
            root.sub("z_init_1"),
            token_s,
            token_z,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let z_init_2 = linear(
            root.sub("z_init_2"),
            token_s,
            token_z,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        // Normalization layers
        let s_norm = LayerNorm::new(
            root.sub("s_norm"),
            vec![token_s],
            token_s as f64 * 1e-5,
            true,
        );
        let z_norm = LayerNorm::new(
            root.sub("z_norm"),
            vec![token_z],
            token_z as f64 * 1e-5,
            true,
        );

        // Recycling projections (with gating initialization)
        let s_recycle = linear(
            root.sub("s_recycle"),
            token_s,
            token_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let z_recycle = linear(
            root.sub("z_recycle"),
            token_z,
            token_z,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        // Initialize recycling weights for gating (zeros)
        s_recycle.ws.set_zero();
        z_recycle.ws.set_zero();

        // Owned PairformerModule — keys align with Lightning `pairformer_module.*`
        let pairformer = PairformerModule::new(
            root.sub("pairformer_module"),
            token_s,
            token_z,
            num_blocks,
            None,       // num_heads (default 16)
            None,       // dropout (default 0.25)
            None,       // pairwise_head_width (default 32)
            None,       // pairwise_num_heads (default 4)
            None,       // post_layer_norm (default false)
            None,       // activation_checkpointing (default false)
            Some(true), // v2 (use Boltz2 variant)
            device,
        );

        let msa = MsaModule::new(
            root.sub("msa_module"),
            token_s,
            token_z,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            device,
        );

        Self {
            token_s,
            token_z,
            s_init,
            z_init_1,
            z_init_2,
            s_norm,
            z_norm,
            s_recycle,
            z_recycle,
            pairformer,
            msa,
            template: TemplateModule,
            device,
        }
    }

    /// Initialize s and z from input embeddings
    ///
    /// This is typically called once at the start to create initial s, z from
    /// input embeddings (from InputEmbedder).
    ///
    /// # Arguments
    ///
    /// * `s_inputs` - Input sequence embeddings [B, N, token_s]
    ///
    /// # Returns
    ///
    /// Tuple of (s_init, z_init) where:
    /// - s_init: [B, N, token_s]
    /// - z_init: [B, N, N, token_z]
    pub fn initialize(&self, s_inputs: &Tensor) -> (Tensor, Tensor) {
        let batch_size = s_inputs.size()[0];
        let num_tokens = s_inputs.size()[1];

        // Initialize sequence embeddings
        let s_init = self.s_init.forward(s_inputs);

        // Initialize pairwise embeddings from sequence
        let z_init_1 = self.z_init_1.forward(s_inputs).unsqueeze(2); // [B, N, 1, token_z]
        let z_init_2 = self.z_init_2.forward(s_inputs).unsqueeze(3); // [B, 1, N, token_z]
        let z_init = z_init_1 + z_init_2; // [B, N, N, token_z]

        (s_init, z_init)
    }

    /// Apply recycling to s and z
    ///
    /// # Arguments
    ///
    /// * `s_init` - Initial sequence embeddings [B, N, token_s]
    /// * `z_init` - Initial pairwise embeddings [B, N, N, token_z]
    /// * `s_prev` - Previous sequence embeddings [B, N, token_s]
    /// * `z_prev` - Previous pairwise embeddings [B, N, N, token_z]
    ///
    /// # Returns
    ///
    /// Tuple of (s_recycled, z_recycled)
    pub fn apply_recycling(
        &self,
        s_init: &Tensor,
        z_init: &Tensor,
        s_prev: &Tensor,
        z_prev: &Tensor,
    ) -> (Tensor, Tensor) {
        let s_recycled = s_init + self.s_recycle.forward(&self.s_norm.forward(s_prev));
        let z_recycled = z_init + self.z_recycle.forward(&self.z_norm.forward(z_prev));

        (s_recycled, z_recycled)
    }

    /// Forward through pairformer stack (owned component)
    ///
    /// This is the **core method** that other components use to process (s, z).
    /// MSA and templates can add to z before/after calling this.
    ///
    /// # Arguments
    ///
    /// * `s` - Sequence embeddings [B, N, token_s]
    /// * `z` - Pairwise embeddings [B, N, N, token_z]
    /// * `mask` - Sequence mask [B, N, N]
    /// * `pair_mask` - Pairwise mask [B, N, N]
    ///
    /// # Returns
    ///
    /// Tuple of (s_out, z_out) where:
    /// - s_out: [B, N, token_s]
    /// - z_out: [B, N, N, token_z]
    pub fn forward_pairformer(
        &self,
        s: &Tensor,
        z: &Tensor,
        mask: &Tensor,
        pair_mask: &Tensor,
    ) -> (Tensor, Tensor) {
        self.pairformer.forward(s, z, mask, pair_mask, false)
    }

    /// Full forward pass through trunk with recycling
    ///
    /// This handles initialization and recycling loop, calling forward_pairformer
    /// internally. Other components can call forward_pairformer directly.
    ///
    /// # Arguments
    ///
    /// * `s_inputs` - Input sequence embeddings [B, N, token_s]
    /// * `recycling_steps` - Number of recycling iterations (default 0)
    ///
    /// # Returns
    ///
    /// Tuple of (final_s, final_z)
    pub fn forward(
        &self,
        s_inputs: &Tensor,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let recycling_steps = recycling_steps.unwrap_or(0);
        let batch_size = s_inputs.size()[0];
        let num_tokens = s_inputs.size()[1];

        // Initialize from input embeddings
        let (s_init, z_init) = self.initialize(s_inputs);

        // Compute masks
        let mask = Tensor::ones(&[batch_size, num_tokens], (tch::Kind::Float, self.device));
        let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2); // [B, N, N]

        // Initialize recycled embeddings as zeros
        let mut s = Tensor::zeros(
            &[batch_size, num_tokens, self.token_s],
            (s_init.kind(), self.device),
        );
        let mut z = Tensor::zeros(
            &[batch_size, num_tokens, num_tokens, self.token_z],
            (s_init.kind(), self.device),
        );

        // Recycling loop
        for _i in 0..=recycling_steps {
            // Apply recycling
            let (s_recycled, z_recycled) = self.apply_recycling(&s_init, &z_init, &s, &z);

            s = s_recycled;
            z = z_recycled;
            z = self
                .msa
                .forward_trunk_step(&z, &s, msa_feats, false, None, false);
            z = self.template.forward_trunk_step(&z);

            // Run owned pairformer module
            let (s_new, z_new) = self.forward_pairformer(&s, &z, &pair_mask, &pair_mask);

            s = s_new;
            z = z_new;
        }

        Ok((s, z))
    }

    /// Recycling + pairformer from precomputed `s_init` / `z_init` (e.g. `z_init` already includes
    /// relative position bias). Matches Python trunk loop after `z_init += rel_pos(...)`.
    pub fn forward_from_init(
        &self,
        s_init: &Tensor,
        z_init: &Tensor,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let recycling_steps = recycling_steps.unwrap_or(0);
        let batch_size = s_init.size()[0];
        let num_tokens = s_init.size()[1];

        let mask = Tensor::ones(&[batch_size, num_tokens], (tch::Kind::Float, self.device));
        let pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2);

        let mut s = Tensor::zeros(
            &[batch_size, num_tokens, self.token_s],
            (s_init.kind(), self.device),
        );
        let mut z = Tensor::zeros(
            &[batch_size, num_tokens, num_tokens, self.token_z],
            (z_init.kind(), self.device),
        );

        for _i in 0..=recycling_steps {
            let (s_recycled, z_recycled) = self.apply_recycling(s_init, z_init, &s, &z);
            s = s_recycled;
            z = z_recycled;
            z = self
                .msa
                .forward_trunk_step(&z, &s, msa_feats, false, None, false);
            z = self.template.forward_trunk_step(&z);
            let (s_new, z_new) = self.forward_pairformer(&s, &z, &pair_mask, &pair_mask);
            s = s_new;
            z = z_new;
        }

        Ok((s, z))
    }

    /// Get token_s dimension
    pub fn token_s(&self) -> i64 {
        self.token_s
    }

    /// Get token_z dimension
    pub fn token_z(&self) -> i64 {
        self.token_z
    }

    /// Get mutable reference to owned PairformerModule
    ///
    /// This allows external code to access pairformer if needed
    pub fn pairformer_mut(&mut self) -> &mut PairformerModule {
        &mut self.pairformer
    }

    /// Get reference to owned PairformerModule
    pub fn pairformer(&self) -> &PairformerModule {
        &self.pairformer
    }

    /// Pre-pairformer MSA stack (`msa_module` weights in Lightning).
    pub fn msa(&self) -> &MsaModule {
        &self.msa
    }

    /// Python `Boltz2.s_init`: linear on per-token features `[B, N, token_s]`.
    pub fn apply_s_init(&self, s_inputs: &Tensor) -> Tensor {
        self.s_init.forward(s_inputs)
    }

    /// Copy exported `s_init.weight` into this trunk (`[token_s, token_s]`).
    pub fn load_s_init_weight(&mut self, w: &Tensor) {
        self.s_init.ws.copy_(w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that TrunkV2 owns PairformerModule and exposes proper API
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_trunk_owns_pairformer() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 384;
        let token_z = 128;
        let num_blocks = 2;
        let batch_size = 2;
        let num_tokens = 50;

        let vs = VarStore::new(device);

        // Create TrunkV2 that owns a PairformerModule
        let trunk = TrunkV2::new(&vs, Some(token_s), Some(token_z), Some(num_blocks), device);

        assert_eq!(trunk.pairformer().num_blocks(), num_blocks);

        // Test initialize method
        let s_inputs = Tensor::randn(
            &[batch_size, num_tokens, token_s],
            (tch::Kind::Float, device),
        );

        let (s_init, z_init) = trunk.initialize(&s_inputs);

        assert_eq!(s_init.size(), vec![batch_size, num_tokens, token_s]);
        assert_eq!(
            z_init.size(),
            vec![batch_size, num_tokens, num_tokens, token_z]
        );

        // Test apply_recycling method
        let s_prev = Tensor::randn(
            &[batch_size, num_tokens, token_s],
            (tch::Kind::Float, device),
        );
        let z_prev = Tensor::randn(
            &[batch_size, num_tokens, num_tokens, token_z],
            (tch::Kind::Float, device),
        );

        let (s_recycled, z_recycled) = trunk.apply_recycling(&s_init, &z_init, &s_prev, &z_prev);

        assert_eq!(s_recycled.size(), vec![batch_size, num_tokens, token_s]);
        assert_eq!(
            z_recycled.size(),
            vec![batch_size, num_tokens, num_tokens, token_z]
        );

        // Test forward_pairformer method (the key API!)
        let mask = Tensor::ones(
            &[batch_size, num_tokens, num_tokens],
            (tch::Kind::Float, device),
        );
        let pair_mask = Tensor::ones(
            &[batch_size, num_tokens, num_tokens],
            (tch::Kind::Float, device),
        );

        let (s_out, z_out) = trunk.forward_pairformer(&s_init, &z_init, &mask, &pair_mask);

        assert_eq!(s_out.size(), vec![batch_size, num_tokens, token_s]);
        assert_eq!(
            z_out.size(),
            vec![batch_size, num_tokens, num_tokens, token_z]
        );
    }

    /// Test full forward with recycling
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_trunk_full_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 256;
        let token_z = 64;
        let num_blocks = 1;
        let batch_size = 2;
        let num_tokens = 30;

        let vs = VarStore::new(device);
        let trunk = TrunkV2::new(&vs, Some(token_s), Some(token_z), Some(num_blocks), device);

        let s_inputs = Tensor::randn(
            &[batch_size, num_tokens, token_s],
            (tch::Kind::Float, device),
        );

        let (s_out, z_out) = trunk.forward(&s_inputs, Some(1), None).unwrap();

        assert_eq!(s_out.size(), vec![batch_size, num_tokens, token_s]);
        assert_eq!(
            z_out.size(),
            vec![batch_size, num_tokens, num_tokens, token_z]
        );
    }

    /// Test API allows easy connection of other components
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_trunk_api_for_component_connection() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 128;
        let token_z = 32;
        let num_blocks = 1;
        let batch_size = 1;
        let num_tokens = 20;

        let vs = VarStore::new(device);
        let trunk = TrunkV2::new(&vs, Some(token_s), Some(token_z), Some(num_blocks), device);

        let s_inputs = Tensor::randn(
            &[batch_size, num_tokens, token_s],
            (tch::Kind::Float, device),
        );

        // Initialize
        let (s, z) = trunk.initialize(&s_inputs);

        let mask = Tensor::ones(
            &[batch_size, num_tokens, num_tokens],
            (tch::Kind::Float, device),
        );
        let pair_mask = Tensor::ones(
            &[batch_size, num_tokens, num_tokens],
            (tch::Kind::Float, device),
        );

        // Simulate MSA module adding to z
        let msa_contribution = Tensor::randn(
            &[batch_size, num_tokens, num_tokens, token_z],
            (tch::Kind::Float, device),
        );
        let z = z + msa_contribution * 0.1; // Simulate MSA with small weight

        // Call pairformer (easy API!)
        let (s, z) = trunk.forward_pairformer(&s, &z, &mask, &pair_mask);

        assert_eq!(s.size(), vec![batch_size, num_tokens, token_s]);
        assert_eq!(z.size(), vec![batch_size, num_tokens, num_tokens, token_z]);
    }
}
