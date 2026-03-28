//! Pairformer layer/module without sequence track (`PairformerNoSeqLayer`, `PairformerNoSeqModule`).
//!
//! Reference: `boltz-reference/src/boltz/model/layers/pairformer.py`

use super::transition::Transition;
use super::triangular_attention::TriangleAttention;
use super::triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};

use tch::nn::Path;
use tch::{Device, Kind, Tensor};

/// Dropout mask matching `boltz.model.layers.dropout.get_dropout_mask` (non-columnwise).
fn dropout_mask_pair(dropout: f64, z: &Tensor, training: bool) -> Tensor {
    let dropout = if training { dropout } else { 0.0 };
    let v = z.narrow(2, 0, 1).narrow(3, 0, 1);
    let thr = Tensor::from(dropout).to_device(v.device());
    let d = v.rand_like().ge_tensor(&thr);
    d.to_kind(Kind::Float) * (1.0 / (1.0 - dropout).max(1e-12))
}

fn dropout_mask_columnwise(dropout: f64, z: &Tensor, training: bool) -> Tensor {
    let dropout = if training { dropout } else { 0.0 };
    let v = z.narrow(1, 0, 1).narrow(3, 0, 1);
    let thr = Tensor::from(dropout).to_device(v.device());
    let d = v.rand_like().ge_tensor(&thr);
    let d = d.to_kind(Kind::Float) * (1.0 / (1.0 - dropout).max(1e-12));
    let shape = z.size();
    d.expand(shape.as_slice(), false)
}

pub struct PairformerNoSeqLayer {
    dropout: f64,
    tri_mul_out: TriangleMultiplicationOutgoing,
    tri_mul_in: TriangleMultiplicationIncoming,
    tri_att_start: TriangleAttention,
    tri_att_end: TriangleAttention,
    transition_z: Transition,
}

impl PairformerNoSeqLayer {
    pub fn new<'a>(
        path: Path<'a>,
        token_z: i64,
        dropout: Option<f64>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        _post_layer_norm: Option<bool>,
        device: Device,
    ) -> Self {
        let dropout = dropout.unwrap_or(0.25);
        let pairwise_head_width = pairwise_head_width.unwrap_or(32);
        let pairwise_num_heads = pairwise_num_heads.unwrap_or(4);

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
            dropout,
            tri_mul_out,
            tri_mul_in,
            tri_att_start,
            tri_att_end,
            transition_z,
        }
    }

    pub fn forward(
        &self,
        z: &Tensor,
        pair_mask: &Tensor,
        chunk_size_tri_attn: Option<i64>,
        training: bool,
        use_kernels: bool,
    ) -> Tensor {
        let mut z = z.shallow_clone();

        let drop = dropout_mask_pair(self.dropout, &z, training);
        let z_out = self.tri_mul_out.forward(&z, pair_mask, use_kernels);
        z = z + drop * z_out;

        let drop = dropout_mask_pair(self.dropout, &z, training);
        let z_out = self.tri_mul_in.forward(&z, pair_mask, use_kernels);
        z = z + drop * z_out;

        let drop = dropout_mask_pair(self.dropout, &z, training);
        let z_out =
            self.tri_att_start
                .forward(&z, Some(pair_mask), chunk_size_tri_attn, use_kernels);
        z = z + drop * z_out;

        let drop = dropout_mask_columnwise(self.dropout, &z, training);
        let z_out = self
            .tri_att_end
            .forward(&z, Some(pair_mask), chunk_size_tri_attn, use_kernels);
        z = z + drop * z_out;

        {
            let z_t = self.transition_z.forward(&z, None);
            z + z_t
        }
    }
}

/// Stack of `PairformerNoSeqLayer` — pairwise-only pairformer used by the template module.
///
/// Reference: `PairformerNoSeqModule` in `boltz-reference/src/boltz/model/layers/pairformer.py`.
pub struct PairformerNoSeqModule {
    layers: Vec<PairformerNoSeqLayer>,
}

impl PairformerNoSeqModule {
    pub fn new<'a>(
        path: Path<'a>,
        token_z: i64,
        num_blocks: i64,
        dropout: Option<f64>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        post_layer_norm: Option<bool>,
        _activation_checkpointing: Option<bool>,
        device: Device,
    ) -> Self {
        let layers_root = path.sub("layers");
        let layers = (0..num_blocks)
            .map(|i| {
                PairformerNoSeqLayer::new(
                    layers_root.sub(&i.to_string()),
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                    device,
                )
            })
            .collect();
        Self { layers }
    }

    /// Forward pass through all layers (inference mode, training=false).
    pub fn forward(&self, z: &Tensor, pair_mask: &Tensor, use_kernels: bool) -> Tensor {
        let n = z.size()[1];
        let chunk_size = if n > 256 { Some(128) } else { Some(512) };
        let mut z = z.shallow_clone();
        for layer in &self.layers {
            z = layer.forward(&z, pair_mask, chunk_size, false, use_kernels);
        }
        z
    }

    pub fn num_blocks(&self) -> i64 {
        self.layers.len() as i64
    }
}
