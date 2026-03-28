//! Pairformer layers (PairformerLayer, PairformerModule)
//!
//! Reference: boltz-reference/src/boltz/model/layers/pairformer.py
//! Combines attention, triangular operations, and transitions

use super::transition::Transition;
use super::triangular_attention::TriangleAttention;
use super::triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};

use crate::attention::pair_bias::AttentionPairBiasV2;

use crate::tch_compat::layer_norm_1d;
use tch::nn::{Module, Path};
use tch::{Device, Kind, Tensor};

/// Pairformer Layer
///
/// A single layer in the pairformer stack that processes both sequence (s)
/// and pairwise (z) representations through:
/// 1. Pairwise stack: triangular mult -> triangular attention -> transition
/// 2. Sequence stack: attention pair bias -> transition
pub struct PairformerLayer {
    token_z: i64,
    dropout: f64,
    num_heads: i64,
    post_layer_norm: bool,

    // Sequence stack
    pre_norm_s: tch::nn::LayerNorm,
    attention: AttentionPairBiasV2,
    transition_s: Transition,
    s_post_norm: Option<tch::nn::LayerNorm>,

    // Pairwise stack
    tri_mul_out: TriangleMultiplicationOutgoing,
    tri_mul_in: TriangleMultiplicationIncoming,
    tri_att_start: TriangleAttention,
    tri_att_end: TriangleAttention,
    transition_z: Transition,

    device: Device,
}

impl PairformerLayer {
    /// Create a new PairformerLayer
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for this layer (e.g. `vs.root().sub("layers_0")`)
    /// * `token_s` - Sequence dimension
    /// * `token_z` - Pairwise dimension
    /// * `num_heads` - Number of attention heads for sequence attention
    /// * `dropout` - Dropout rate
    /// * `pairwise_head_width` - Hidden dimension for triangular attention
    /// * `pairwise_num_heads` - Number of heads for triangular attention
    /// * `post_layer_norm` - Whether to apply post layer norm on sequence
    /// * `v2` - Whether to use Boltz2 variant (true) or Boltz1 variant (false)
    /// * `device` - Computation device
    pub fn new<'a>(
        path: Path<'a>,
        token_s: i64,
        token_z: i64,
        num_heads: Option<i64>,
        dropout: Option<f64>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        post_layer_norm: Option<bool>,
        v2: Option<bool>,
        device: Device,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(16);
        let dropout = dropout.unwrap_or(0.25);
        let post_layer_norm = post_layer_norm.unwrap_or(false);
        let pairwise_head_width = pairwise_head_width.unwrap_or(32);
        let pairwise_num_heads = pairwise_num_heads.unwrap_or(4);

        // Sequence stack
        let pre_norm_s = layer_norm_1d(path.sub("pre_norm_s"), token_s);

        // Boltz1 vs Boltz2 branching not wired yet; keep API aligned with Python.
        let _ = v2.unwrap_or(true);
        let attention = AttentionPairBiasV2::new(
            path.sub("attention"),
            token_s,
            Some(token_z),
            Some(num_heads),
            None,
            device,
        );

        let transition_s = Transition::new(
            path.sub("transition_s"),
            token_s,
            Some(token_s * 4),
            None,
            device,
        );

        let s_post_norm = if post_layer_norm {
            Some(layer_norm_1d(path.sub("s_post_norm"), token_s))
        } else {
            None
        };

        // Pairwise stack
        let tri_mul_out =
            TriangleMultiplicationOutgoing::new(path.sub("tri_mul_out"), Some(token_z), device);

        let tri_mul_in =
            TriangleMultiplicationIncoming::new(path.sub("tri_mul_in"), Some(token_z), device);

        let tri_att_start = TriangleAttention::new(
            path.sub("tri_att_start"),
            token_z,
            Some(pairwise_head_width),
            Some(pairwise_num_heads),
            Some(true),
            Some(1e9),
            device,
        );

        let tri_att_end = TriangleAttention::new_ending_node(
            path.sub("tri_att_end"),
            token_z,
            Some(pairwise_head_width),
            Some(pairwise_num_heads),
            Some(1e9),
            device,
        );

        let transition_z = Transition::new(
            path.sub("transition_z"),
            token_z,
            Some(token_z * 4),
            None,
            device,
        );

        Self {
            token_z,
            dropout,
            num_heads,
            post_layer_norm,
            pre_norm_s,
            attention,
            transition_s,
            s_post_norm,
            tri_mul_out,
            tri_mul_in,
            tri_att_start,
            tri_att_end,
            transition_z,
            device,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `s` - Sequence tensor of shape [B, N, token_s]
    /// * `z` - Pairwise tensor of shape [B, N, N, token_z]
    /// * `mask` - Sequence mask of shape [B, N, N]
    /// * `pair_mask` - Pairwise mask of shape [B, N, N]
    /// * `chunk_size_tri_attn` - Optional chunk size for triangular attention
    /// * `_use_kernels` - Placeholder for kernel usage
    ///
    /// # Returns
    ///
    /// Tuple of (updated_s, updated_z)
    pub fn forward(
        &self,
        s: &Tensor,
        z: &Tensor,
        mask: &Tensor,
        pair_mask: &Tensor,
        chunk_size_tri_attn: Option<i64>,
        _use_kernels: bool,
    ) -> (Tensor, Tensor) {
        // Compute pairwise stack
        let mut z = z.shallow_clone();

        // Triangle multiplication outgoing
        let dropout_mask = if self.dropout > 0.0 {
            Some(self.create_dropout_mask(&z))
        } else {
            None
        };

        let z_out = self.tri_mul_out.forward(&z, pair_mask, false);
        z = if let Some(drop) = dropout_mask {
            z + drop * z_out
        } else {
            z + z_out
        };

        // Triangle multiplication incoming
        let dropout_mask = if self.dropout > 0.0 {
            Some(self.create_dropout_mask(&z))
        } else {
            None
        };

        let z_out = self.tri_mul_in.forward(&z, pair_mask, false);
        z = if let Some(drop) = dropout_mask {
            z + drop * z_out
        } else {
            z + z_out
        };

        // Triangle attention starting
        let dropout_mask = if self.dropout > 0.0 {
            Some(self.create_dropout_mask(&z))
        } else {
            None
        };

        let z_out = self
            .tri_att_start
            .forward(&z, Some(pair_mask), chunk_size_tri_attn, false);
        z = if let Some(drop) = dropout_mask {
            z + drop * z_out
        } else {
            z + z_out
        };

        // Triangle attention ending (columnwise dropout)
        let dropout_mask = if self.dropout > 0.0 {
            Some(self.create_dropout_mask_columnwise(&z))
        } else {
            None
        };

        let z_out = self
            .tri_att_end
            .forward(&z, Some(pair_mask), chunk_size_tri_attn, false);
        z = if let Some(drop) = dropout_mask {
            z + drop * z_out
        } else {
            z + z_out
        };

        // Transition z
        let z_out = self.transition_z.forward(&z, None);
        z = z + z_out;

        // Compute sequence stack
        let s_normed = self.pre_norm_s.forward(s);

        // Cast to float for attention (autocast disabled)
        let s_normed_float = s_normed.to_kind(Kind::Float);
        let z_float = z.to_kind(Kind::Float);
        let mask_float = mask.to_kind(Kind::Float);

        let s_out_float = self.attention.forward(
            &s_normed_float,
            &z_float,
            &mask_float,
            &s_normed_float,
            None,
        );

        let s_out = s_out_float.to_kind(s.kind());

        let s = s + s_out;

        // Transition s
        let s_out = self.transition_s.forward(&s, None);
        let mut s = s + s_out;

        // Apply post layer norm if enabled
        if let Some(ref post_norm) = self.s_post_norm {
            s = post_norm.forward(&s);
        }

        (s, z)
    }

    fn create_dropout_mask(&self, tensor: &Tensor) -> Tensor {
        let scale = 1.0 / (1.0 - self.dropout);
        let thr = Tensor::from(self.dropout).to_device(tensor.device());
        let mask = tensor.rand_like().gt_tensor(&thr);
        mask.to_kind(tensor.kind()).to_kind(Kind::Float) * scale
    }

    fn create_dropout_mask_columnwise(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.size();
        let _batch_size = shape[0];
        let dim = shape[3];

        // Create columnwise mask: broadcast over first two dimensions
        let mask = Tensor::empty(&[1i64, 1, 1, dim], (Kind::Float, self.device));
        let thr = Tensor::from(self.dropout).to_device(self.device);
        let mask =
            (mask.rand_like().gt_tensor(&thr)).to_kind(Kind::Float) * (1.0 / (1.0 - self.dropout));

        mask.expand(shape.as_slice(), false)
    }
}

/// Pairformer Module
///
/// A stack of PairformerLayers that processes the sequence and pairwise
/// representations through multiple layers of processing.
pub struct PairformerModule {
    token_z: i64,
    num_blocks: i64,
    dropout: f64,
    num_heads: i64,
    post_layer_norm: bool,
    activation_checkpointing: bool,

    layers: Vec<PairformerLayer>,
}

impl PairformerModule {
    /// Create a new PairformerModule
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for the stack (e.g. `vs.root().sub("pairformer_module")`)
    /// * `token_s` - Sequence dimension
    /// * `token_z` - Pairwise dimension
    /// * `num_blocks` - Number of PairformerLayers
    /// * `num_heads` - Number of attention heads for sequence attention
    /// * `dropout` - Dropout rate
    /// * `pairwise_head_width` - Hidden dimension for triangular attention
    /// * `pairwise_num_heads` - Number of heads for triangular attention
    /// * `post_layer_norm` - Whether to apply post layer norm on sequence
    /// * `activation_checkpointing` - Whether to use activation checkpointing (not yet implemented)
    /// * `v2` - Whether to use Boltz2 variant
    /// * `device` - Computation device
    pub fn new<'a>(
        path: Path<'a>,
        token_s: i64,
        token_z: i64,
        num_blocks: i64,
        num_heads: Option<i64>,
        dropout: Option<f64>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        post_layer_norm: Option<bool>,
        activation_checkpointing: Option<bool>,
        v2: Option<bool>,
        device: Device,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(16);
        let dropout = dropout.unwrap_or(0.25);
        let post_layer_norm = post_layer_norm.unwrap_or(false);
        let activation_checkpointing = activation_checkpointing.unwrap_or(false);

        // Match PyTorch `nn.ModuleList` names: `pairformer_module.layers.0.*`
        let layers_root = path.sub("layers");
        let mut layers = Vec::new();
        for i in 0..num_blocks {
            let layer = PairformerLayer::new(
                layers_root.sub(&i.to_string()),
                token_s,
                token_z,
                Some(num_heads),
                Some(dropout),
                pairwise_head_width,
                pairwise_num_heads,
                Some(post_layer_norm),
                v2,
                device,
            );
            layers.push(layer);
        }

        Self {
            token_z,
            num_blocks,
            dropout,
            num_heads,
            post_layer_norm,
            activation_checkpointing,
            layers,
        }
    }

    /// Number of pairformer blocks in this stack.
    pub fn num_blocks(&self) -> i64 {
        self.num_blocks
    }

    /// Forward pass through all layers
    ///
    /// # Arguments
    ///
    /// * `s` - Sequence tensor of shape [B, N, token_s]
    /// * `z` - Pairwise tensor of shape [B, N, N, token_z]
    /// * `mask` - Sequence mask of shape [B, N, N]
    /// * `pair_mask` - Pairwise mask of shape [B, N, N]
    /// * `_use_kernels` - Placeholder for kernel usage
    ///
    /// # Returns
    ///
    /// Tuple of (updated_s, updated_z)
    pub fn forward(
        &self,
        s: &Tensor,
        z: &Tensor,
        mask: &Tensor,
        pair_mask: &Tensor,
        _use_kernels: bool,
    ) -> (Tensor, Tensor) {
        let mut s = s.shallow_clone();
        let mut z = z.shallow_clone();

        // Determine chunk size based on sequence length (matching Python logic)
        let chunk_size_tri_attn = if z.size()[1] > 256 {
            // chunk_size_threshold from const.py
            Some(128)
        } else {
            Some(512)
        };

        for layer in &self.layers {
            // TODO: Implement activation checkpointing
            let (s_new, z_new) = layer.forward(&s, &z, mask, pair_mask, chunk_size_tri_attn, false);
            s = s_new;
            z = z_new;
        }

        (s, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_pairformer_layer_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = PairformerLayer::new(
            vs.root(),
            token_s,
            token_z,
            Some(num_heads),
            None,
            None,
            None,
            None,
            Some(true),
            device,
        );

        let s = Tensor::randn(&[batch_size, seq_len, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let (s_out, z_out) = layer.forward(&s, &z, &mask, &pair_mask, None, false);

        assert_eq!(s_out.size(), vec![batch_size, seq_len, token_s]);
        assert_eq!(z_out.size(), vec![batch_size, seq_len, seq_len, token_z]);
    }

    #[test]
    fn test_pairformer_module_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_blocks = 2;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let module = PairformerModule::new(
            vs.root(),
            token_s,
            token_z,
            num_blocks,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true),
            device,
        );

        let s = Tensor::randn(&[batch_size, seq_len, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let (s_out, z_out) = module.forward(&s, &z, &mask, &pair_mask, false);

        assert_eq!(s_out.size(), vec![batch_size, seq_len, token_s]);
        assert_eq!(z_out.size(), vec![batch_size, seq_len, seq_len, token_z]);
    }
}
