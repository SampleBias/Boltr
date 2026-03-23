//! Attention pair bias layer (Boltz2)
//!
//! Reference: boltz-reference/src/boltz/model/layers/attentionv2.py

use crate::tch_compat::layer_norm_1d;
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

/// Attention pair bias layer (Boltz2 variant)
///
/// This implements attention with pairwise bias for sequence representations.
/// The bias comes from the pairwise representation tensor `z`.
///
/// # Arguments
///
/// * `vs` - Variable store for parameter storage
/// * `c_s` - Input/output sequence dimension
/// * `c_z` - Pairwise representation dimension
/// * `num_heads` - Number of attention heads
/// * `inf` - Large negative value for masking in attention scores
/// * `device` - Computation device
pub struct AttentionPairBiasV2 {
    c_s: i64,
    num_heads: i64,
    head_dim: i64,
    inf: f64,

    // Projections for query, key, value, and gate
    proj_q: tch::nn::Linear,
    proj_k: tch::nn::Linear,
    proj_v: tch::nn::Linear,
    proj_g: tch::nn::Linear,
    proj_o: tch::nn::Linear,

    // Pairwise bias projection (optional)
    proj_z_layer_norm: Option<tch::nn::LayerNorm>,
    proj_z: Option<tch::nn::Linear>,
    compute_pair_bias: bool,

    device: Device,
}

impl AttentionPairBiasV2 {
    /// Create a new AttentionPairBiasV2 layer
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for parameter names (e.g. `vs.root().sub("attention")`)
    /// * `c_s` - Input/output sequence dimension
    /// * `c_z` - Pairwise representation dimension
    /// * `num_heads` - Number of attention heads (must divide c_s evenly)
    /// * `inf` - Large negative value for masking in attention scores
    /// * `device` - Computation device
    ///
    /// # Panics
    ///
    /// Panics if `c_s` is not divisible by `num_heads`
    pub fn new<'a>(
        path: Path<'a>,
        c_s: i64,
        c_z: Option<i64>,
        num_heads: Option<i64>,
        inf: Option<f64>,
        device: Device,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(16);
        assert_eq!(
            c_s % num_heads,
            0,
            "c_s ({}) must be divisible by num_heads ({})",
            c_s,
            num_heads
        );

        let head_dim = c_s / num_heads;
        let compute_pair_bias = c_z.is_some();

        // Linear projections for sequence representations
        let proj_q = linear(
            path.sub("proj_q"),
            c_s,
            c_s,
            LinearConfig {
                bias: true,
                ..Default::default()
            },
        );

        let proj_k = linear(
            path.sub("proj_k"),
            c_s,
            c_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let proj_v = linear(
            path.sub("proj_v"),
            c_s,
            c_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let proj_g = linear(
            path.sub("proj_g"),
            c_s,
            c_s,
            LinearConfig {
                bias: true,
                ..Default::default()
            },
        );

        let proj_o = linear(
            path.sub("proj_o"),
            c_s,
            c_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        // Pairwise bias projection (LayerNorm -> Linear -> reshape)
        let (proj_z_layer_norm, proj_z) = if compute_pair_bias {
            let c_z = c_z.unwrap();
            let ln = layer_norm_1d(path.sub("proj_z_layer_norm"), c_z);

            let proj_z_linear = linear(
                path.sub("proj_z"),
                c_z,
                num_heads,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            );

            (Some(ln), Some(proj_z_linear))
        } else {
            (None, None)
        };

        Self {
            c_s,
            num_heads,
            head_dim,
            inf: inf.unwrap_or(1e6),
            proj_q,
            proj_k,
            proj_v,
            proj_g,
            proj_o,
            proj_z_layer_norm,
            proj_z,
            compute_pair_bias,
            device,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `s` - Sequence tensor of shape [B, N, c_s]
    /// * `z` - Pairwise tensor of shape [B, N, N, c_z] or [B, N, N] (bias only)
    /// * `mask` - Pairwise mask tensor of shape [B, N, N]
    /// * `k_in` - Key input tensor (can be same as s)
    /// * `multiplicity` - Repeat multiplicity for bias (default: 1)
    ///
    /// # Returns
    ///
    /// Output sequence tensor of shape [B, N, c_s]
    pub fn forward(
        &self,
        s: &Tensor,
        z: &Tensor,
        mask: &Tensor,
        k_in: &Tensor,
        multiplicity: Option<i64>,
    ) -> Tensor {
        let multiplicity = multiplicity.unwrap_or(1);
        let b = s.size()[0];

        // Compute projections
        let q = self
            .proj_q
            .forward(s)
            .view([b, -1, self.num_heads, self.head_dim]);
        let k = self
            .proj_k
            .forward(k_in)
            .view([b, -1, self.num_heads, self.head_dim]);
        let v = self
            .proj_v
            .forward(k_in)
            .view([b, -1, self.num_heads, self.head_dim]);

        // Compute pairwise bias
        let bias = if self.compute_pair_bias {
            let ln = self.proj_z_layer_norm.as_ref().unwrap();
            let lin = self.proj_z.as_ref().unwrap();

            // Apply LayerNorm -> Linear -> reshape
            let z_proj = ln.forward(z);
            let z_linear = lin.forward(&z_proj);
            z_linear
        } else {
            // If z is already bias, just reshape it
            if z.dim() == 3 {
                z.view([b, -1, self.num_heads])
            } else {
                z.shallow_clone()
            }
        };

        // Repeat bias if multiplicity > 1
        let bias = if multiplicity > 1 {
            bias.repeat_interleave_self_int(multiplicity, Some(0), None)
        } else {
            bias
        };

        // Compute gate
        let g = self.proj_g.forward(s).sigmoid();

        // Compute attention with autocast disabled (float32)
        // Note: tch-rs doesn't have autocast, so we explicitly cast
        let q_float = q.to_kind(Kind::Float);
        let k_float = k.to_kind(Kind::Float);
        let v_float = v.to_kind(Kind::Float);
        let bias_float = bias.to_kind(Kind::Float);
        let mask_float = mask.to_kind(Kind::Float);

        // Compute attention scores: einsum("bihd,bjhd->bhij", q, k)
        let attn = q_float.matmul(&k_float.transpose(1, 2)); // [B, H, N, N]

        // Scale by sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt();
        let attn = attn / scale;

        // Add pairwise bias
        let attn = attn + bias_float;

        // Apply mask: add -inf where mask is 0
        let mask_expanded = mask_float.unsqueeze(1); // [B, 1, N, N]
        let attn = attn
            + mask_expanded
                .ones_like()
                .g_sub(&mask_expanded)
                .g_mul_scalar(-self.inf);

        // Softmax over last dimension
        let attn = attn.softmax(-1, Kind::Float);

        // Compute output: einsum("bhij,bjhd->bihd", attn, v)
        let o = attn.matmul(&v_float); // [B, H, N, D]

        // Restore original dtype
        let o = o.to_kind(s.kind());

        // Reshape and apply output projection with gate
        let o = o.view([b, -1, self.c_s]);
        let o = o * g;
        let o = self.proj_o.forward(&o);

        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_attention_pair_bias_v2_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let c_s = 64;
        let c_z = 32;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer =
            AttentionPairBiasV2::new(vs.root(), c_s, Some(c_z), Some(num_heads), None, device);

        let s = Tensor::randn(&[batch_size, seq_len, c_s], (Kind::Float, device));
        let z = Tensor::randn(&[batch_size, seq_len, seq_len, c_z], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&s, &z, &mask, &s, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, c_s]);
    }

    #[test]
    fn test_attention_pair_bias_v2_no_bias() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let c_s = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = AttentionPairBiasV2::new(vs.root(), c_s, None, Some(num_heads), None, device);

        let s = Tensor::randn(&[batch_size, seq_len, c_s], (Kind::Float, device));
        let z = Tensor::zeros(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&s, &z, &mask, &s, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, c_s]);
    }
}
