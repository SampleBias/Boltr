//! Diffusion transformer stack: `AdaLN`, `ConditionedTransitionBlock`,
//! `DiffusionTransformerLayer`, `DiffusionTransformer`, `AtomTransformer`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/transformersv2.py`

use crate::attention::AttentionPairBiasV2;
use crate::tch_compat::{layer_norm_1d, linear_no_bias};
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

// ---------------------------------------------------------------------------
// AdaLN  (Algorithm 26)
// ---------------------------------------------------------------------------

pub struct AdaLN {
    a_norm: tch::nn::LayerNorm,
    s_norm: tch::nn::LayerNorm,
    s_scale: tch::nn::Linear,
    s_bias: tch::nn::Linear,
}

impl AdaLN {
    pub fn new(path: Path<'_>, dim: i64, dim_single_cond: i64) -> Self {
        let a_norm = tch::nn::layer_norm(
            path.sub("a_norm"),
            vec![dim],
            tch::nn::LayerNormConfig {
                elementwise_affine: false,
                ..Default::default()
            },
        );
        // Python uses `LayerNorm(dim, bias=False)` — weight only. tch 0.16 always
        // creates both weight+bias when affine; the extra zero bias is harmless at inference.
        let s_norm = layer_norm_1d(path.sub("s_norm"), dim_single_cond);
        let s_scale = linear(
            path.sub("s_scale"),
            dim_single_cond,
            dim,
            LinearConfig::default(),
        );
        let s_bias = linear_no_bias(path.sub("s_bias"), dim_single_cond, dim);
        Self {
            a_norm,
            s_norm,
            s_scale,
            s_bias,
        }
    }

    pub fn forward(&self, a: &Tensor, s: &Tensor) -> Tensor {
        let a = self.a_norm.forward(a);
        let s = self.s_norm.forward(s);
        self.s_scale.forward(&s).sigmoid() * a + self.s_bias.forward(&s)
    }
}

// ---------------------------------------------------------------------------
// ConditionedTransitionBlock  (Algorithm 25)
// ---------------------------------------------------------------------------

pub struct ConditionedTransitionBlock {
    adaln: AdaLN,
    swish_gate_linear: tch::nn::Linear,
    a_to_b: tch::nn::Linear,
    b_to_a: tch::nn::Linear,
    output_projection_linear: tch::nn::Linear,
}

impl ConditionedTransitionBlock {
    pub fn new(
        path: Path<'_>,
        dim_single: i64,
        dim_single_cond: i64,
        expansion_factor: Option<i64>,
    ) -> Self {
        let expansion_factor = expansion_factor.unwrap_or(2);
        let dim_inner = dim_single * expansion_factor;

        let adaln = AdaLN::new(path.sub("adaln"), dim_single, dim_single_cond);

        // swish_gate: LinearNoBias(dim_single, dim_inner * 2) → SwiGLU
        let swish_gate_linear =
            linear_no_bias(path.sub("swish_gate").sub("0"), dim_single, dim_inner * 2);
        let a_to_b = linear_no_bias(path.sub("a_to_b"), dim_single, dim_inner);
        let b_to_a = linear_no_bias(path.sub("b_to_a"), dim_inner, dim_single);

        // output_projection: Linear(dim_single_cond, dim_single, bias=True) → Sigmoid
        // init: zeros weight, bias=-2.0
        let output_projection_linear = linear(
            path.sub("output_projection").sub("0"),
            dim_single_cond,
            dim_single,
            LinearConfig::default(),
        );

        Self {
            adaln,
            swish_gate_linear,
            a_to_b,
            b_to_a,
            output_projection_linear,
        }
    }

    pub fn forward(&self, a: &Tensor, s: &Tensor) -> Tensor {
        let a = self.adaln.forward(a, s);
        // SwiGLU: split in half, silu(gates) * x
        let gate_out = self.swish_gate_linear.forward(&a);
        let chunks = gate_out.chunk(2, -1);
        let swiglu = chunks[1].silu() * &chunks[0];
        let b = swiglu * self.a_to_b.forward(&a);
        self.output_projection_linear.forward(s).sigmoid() * self.b_to_a.forward(&b)
    }
}

// ---------------------------------------------------------------------------
// DiffusionTransformerLayer  (Algorithm 23)
// ---------------------------------------------------------------------------

pub struct DiffusionTransformerLayer {
    adaln: AdaLN,
    pair_bias_attn: AttentionPairBiasV2,
    output_projection_linear: tch::nn::Linear,
    transition: ConditionedTransitionBlock,
    c_s: i64,
}

impl DiffusionTransformerLayer {
    pub fn new(path: Path<'_>, heads: i64, dim: i64, dim_single_cond: i64, device: Device) -> Self {
        let adaln = AdaLN::new(path.sub("adaln"), dim, dim_single_cond);
        // AttentionPairBias with compute_pair_bias=False → c_z=None
        let pair_bias_attn = AttentionPairBiasV2::new(
            path.sub("pair_bias_attn"),
            dim,
            None,
            Some(heads),
            None,
            device,
        );
        let output_projection_linear = linear(
            path.sub("output_projection_linear"),
            dim_single_cond,
            dim,
            LinearConfig::default(),
        );
        let transition =
            ConditionedTransitionBlock::new(path.sub("transition"), dim, dim_single_cond, None);
        Self {
            adaln,
            pair_bias_attn,
            output_projection_linear,
            transition,
            c_s: dim,
        }
    }

    pub fn forward(
        &self,
        a: &Tensor,
        s: &Tensor,
        bias: Option<&Tensor>,
        mask: &Tensor,
        multiplicity: i64,
    ) -> Tensor {
        let b_val = self.adaln.forward(a, s);

        let bias_t = match bias {
            Some(b) => b.shallow_clone(),
            None => Tensor::zeros(
                &[a.size()[0], a.size()[1], a.size()[1]],
                (a.kind(), a.device()),
            ),
        };

        // Rust AttentionPairBiasV2 expects a 3D mask [B, I, J].
        // Expand 2D key mask [B, N] → [B, 1, N] → broadcast inside attention.
        let mask_3d = if mask.dim() == 2 {
            mask.unsqueeze(1)
                .expand(&[mask.size()[0], a.size()[1], mask.size()[1]], false)
        } else {
            mask.shallow_clone()
        };

        let b_val =
            self.pair_bias_attn
                .forward(&b_val, &bias_t, &mask_3d, &b_val, Some(multiplicity));

        let b_val = self.output_projection_linear.forward(s).sigmoid() * b_val;

        let a_out = a + b_val;
        let t = self.transition.forward(&a_out, s);
        &a_out + t
    }
}

// ---------------------------------------------------------------------------
// DiffusionTransformer  (Algorithm 23 - full stack)
// ---------------------------------------------------------------------------

pub struct DiffusionTransformer {
    layers: Vec<DiffusionTransformerLayer>,
    num_layers: i64,
    pair_bias_attn: bool,
}

impl DiffusionTransformer {
    pub fn new(
        path: Path<'_>,
        depth: i64,
        heads: i64,
        dim: i64,
        dim_single_cond: Option<i64>,
        pair_bias_attn: bool,
        device: Device,
    ) -> Self {
        let dim_single_cond = dim_single_cond.unwrap_or(dim);
        let mut layers = Vec::with_capacity(depth as usize);
        for i in 0..depth {
            layers.push(DiffusionTransformerLayer::new(
                path.sub("layers").sub(format!("{i}")),
                heads,
                dim,
                dim_single_cond,
                device,
            ));
        }
        Self {
            layers,
            num_layers: depth,
            pair_bias_attn,
        }
    }

    pub fn forward(
        &self,
        a: &Tensor,
        s: &Tensor,
        bias: Option<&Tensor>,
        mask: &Tensor,
        multiplicity: i64,
    ) -> Tensor {
        let mut out = a.shallow_clone();

        if self.pair_bias_attn {
            if let Some(bias) = bias {
                let size = bias.size();
                // [B, N, M, D] → [B, N, M, L, D//L]
                let per_layer = size[3] / self.num_layers;
                let bias_reshaped =
                    bias.reshape(&[size[0], size[1], size[2], self.num_layers, per_layer]);

                for (i, layer) in self.layers.iter().enumerate() {
                    let bias_l = bias_reshaped.select(3, i as i64);
                    out = layer.forward(&out, s, Some(&bias_l), mask, multiplicity);
                }
            } else {
                for layer in &self.layers {
                    out = layer.forward(&out, s, None, mask, multiplicity);
                }
            }
        } else {
            for layer in &self.layers {
                out = layer.forward(&out, s, None, mask, multiplicity);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// AtomTransformer  (Algorithm 7 - windowed)
// ---------------------------------------------------------------------------

pub struct AtomTransformer {
    attn_window_queries: i64,
    attn_window_keys: i64,
    diffusion_transformer: DiffusionTransformer,
}

impl AtomTransformer {
    pub fn new(
        path: Path<'_>,
        attn_window_queries: i64,
        attn_window_keys: i64,
        depth: i64,
        heads: i64,
        dim: i64,
        dim_single_cond: Option<i64>,
        device: Device,
    ) -> Self {
        let diffusion_transformer = DiffusionTransformer::new(
            path.sub("diffusion_transformer"),
            depth,
            heads,
            dim,
            dim_single_cond,
            true,
            device,
        );
        Self {
            attn_window_queries,
            attn_window_keys,
            diffusion_transformer,
        }
    }

    /// Windowed atom-level attention.
    ///
    /// * `q`: `[B, M, D]` atom queries
    /// * `c`: `[B, M, D_cond]` atom conditioning
    /// * `bias`: `[B, K, W, H, D_heads]` pre-windowed pair bias (already expanded for multiplicity outside)
    /// * `mask`: `[B, M]` atom padding mask (float)
    /// * `multiplicity`: how many times the batch was repeated for diffusion sampling
    pub fn forward(
        &self,
        q: &Tensor,
        c: &Tensor,
        bias: &Tensor,
        mask: &Tensor,
        multiplicity: i64,
    ) -> Tensor {
        let w = self.attn_window_queries;
        let h = self.attn_window_keys;
        let size = q.size();
        let (b, n, d) = (size[0], size[1], size[2]);
        let nw = n / w;

        let q_w = q.reshape(&[b * nw, w, d]);
        let c_w = c.reshape(&[b * nw, w, c.size()[2]]);

        // bias: repeat_interleave(multiplicity, 0) then reshape to windowed
        let bias_exp = bias.repeat_interleave_self_int(multiplicity, Some(0), None);
        let bias_size = bias_exp.size();
        let bias_w = bias_exp.reshape(&[bias_size[0] * nw, w, h, bias_size[4]]);

        // Key mask: for windowed attention, use all-ones since atom masking is handled
        // by the windowed pair bias construction and the outer mask. Shape [b*nw, W, H]
        // for the pairwise-style mask expected by AttentionPairBiasV2.
        let mask_w = Tensor::ones(&[b * nw, w, h], (mask.kind(), mask.device()));

        let out = self.diffusion_transformer.forward(
            &q_w,
            &c_w,
            Some(&bias_w),
            &mask_w,
            1, // multiplicity already expanded in bias
        );

        out.reshape(&[b, nw * w, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn adaln_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let dim = 64;
        let dim_cond = 32;
        let adaln = AdaLN::new(vs.root(), dim, dim_cond);
        let a = Tensor::randn(&[2, 8, dim], (Kind::Float, device));
        let s = Tensor::randn(&[2, 8, dim_cond], (Kind::Float, device));
        let out = adaln.forward(&a, &s);
        assert_eq!(out.size(), vec![2, 8, dim]);
    }

    #[test]
    fn conditioned_transition_block_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let dim = 64;
        let ctb = ConditionedTransitionBlock::new(vs.root(), dim, dim, None);
        let a = Tensor::randn(&[2, 8, dim], (Kind::Float, device));
        let s = Tensor::randn(&[2, 8, dim], (Kind::Float, device));
        let out = ctb.forward(&a, &s);
        assert_eq!(out.size(), vec![2, 8, dim]);
    }

    #[test]
    fn diffusion_transformer_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let dim = 64;
        let depth = 2;
        let heads = 4;
        let b = 2_i64;
        let n = 8_i64;
        let dt = DiffusionTransformer::new(vs.root(), depth, heads, dim, Some(dim), true, device);
        let a = Tensor::randn(&[b, n, dim], (Kind::Float, device));
        let s = Tensor::randn(&[b, n, dim], (Kind::Float, device));
        let bias = Tensor::randn(&[b, n, n, heads * depth], (Kind::Float, device));
        let mask = Tensor::ones(&[b, n, n], (Kind::Float, device));
        let out = dt.forward(&a, &s, Some(&bias), &mask, 1);
        assert_eq!(out.size(), vec![b, n, dim]);
    }
}
