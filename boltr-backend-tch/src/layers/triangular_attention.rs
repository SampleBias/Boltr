//! Triangular attention layers (TriangleAttentionStartingNode/EndingNode)
//!
//! Reference: boltz-reference/src/boltz/model/layers/triangular_attention/attention.py
//! and primitives.py (`Attention`, `_attention`) with `use_kernels=False`.
//!
//! VarStore layout matches Lightning: `mha.linear_{q,k,v,o,g}` under each `tri_att_*`.

use crate::tch_compat::layer_norm_1d;
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

/// Triangle Attention layer (base implementation)
pub struct TriangleAttention {
    c_in: i64,
    c_hidden: i64,
    no_heads: i64,
    starting: bool,
    inf: f64,

    layer_norm: tch::nn::LayerNorm,
    linear: tch::nn::Linear,

    /// Boltz `Attention` projections (`pairwise_head_width * pairwise_num_heads` output dim).
    mha_linear_q: tch::nn::Linear,
    mha_linear_k: tch::nn::Linear,
    mha_linear_v: tch::nn::Linear,
    mha_linear_o: tch::nn::Linear,
    mha_linear_g: tch::nn::Linear,

    device: Device,
}

impl TriangleAttention {
    /// * `c_hidden` — per-head width (`pairwise_head_width`); total Q/K/V dim is `c_hidden * no_heads`.
    pub fn new<'a>(
        path: Path<'a>,
        c_in: i64,
        c_hidden: Option<i64>,
        no_heads: Option<i64>,
        starting: Option<bool>,
        inf: Option<f64>,
        device: Device,
    ) -> Self {
        let c_hidden = c_hidden.unwrap_or(32);
        let no_heads = no_heads.unwrap_or(4);
        let starting = starting.unwrap_or(true);
        let mha_dim = c_hidden * no_heads;

        let layer_norm = layer_norm_1d(path.sub("layer_norm"), c_in);

        let bias_linear = linear(
            path.sub("linear"),
            c_in,
            no_heads,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let mha = path.sub("mha");
        let mha_linear_q = linear(
            mha.sub("linear_q"),
            c_in,
            mha_dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let mha_linear_k = linear(
            mha.sub("linear_k"),
            c_in,
            mha_dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let mha_linear_v = linear(
            mha.sub("linear_v"),
            c_in,
            mha_dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let mha_linear_o = linear(
            mha.sub("linear_o"),
            mha_dim,
            c_in,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let mha_linear_g = linear(
            mha.sub("linear_g"),
            c_in,
            mha_dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            c_in,
            c_hidden,
            no_heads,
            starting,
            inf: inf.unwrap_or(1e9),
            layer_norm,
            linear: bias_linear,
            mha_linear_q,
            mha_linear_k,
            mha_linear_v,
            mha_linear_o,
            mha_linear_g,
            device,
        }
    }

    /// Forward pass (`chunk_size` reserved for parity with Python `chunk_layer`; full attention for now).
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        chunk_size: Option<i64>,
        _use_kernels: bool,
    ) -> Tensor {
        let _ = chunk_size;

        let mask_tensor = if let Some(m) = mask {
            m.shallow_clone()
        } else {
            Tensor::ones(x.size().as_slice(), (Kind::Float, self.device))
        };

        let mut x = x.shallow_clone();
        let mut mask_tensor = mask_tensor;

        if !self.starting {
            x = x.transpose(-2, -3);
            mask_tensor = mask_tensor.transpose(-1, -2);
        }

        x = self.layer_norm.forward(&x);

        // [*, I, 1, 1, J] — same as `mask[..., :, None, None, :]` in Python.
        let mask_expanded = mask_tensor.unsqueeze(-2).unsqueeze(-3);
        let mask_bias = mask_expanded * self.inf - self.inf;

        // [*, I, J, H] -> `permute_final_dims(..., (2,0,1))` -> [*, H, I, J]
        let triangle_bias = self.linear.forward(&x);
        let sh = triangle_bias.size();
        let z = sh.len() - 3;
        let mut perm: Vec<i64> = (0..z as i64).collect();
        perm.push((z + 2) as i64);
        perm.push(z as i64);
        perm.push((z + 1) as i64);
        let triangle_bias = triangle_bias.permute(perm.as_slice());
        let triangle_bias = triangle_bias.unsqueeze(-4);

        let output = self.mha_with_bias(&x, &mask_bias, &triangle_bias);

        if !self.starting {
            output.transpose(-2, -3)
        } else {
            output
        }
    }

    /// `_attention` in primitives.py: softmax over keys, biases broadcast onto `[*, H, J, J]`.
    fn mha_with_bias(&self, x: &Tensor, mask_bias: &Tensor, triangle_bias: &Tensor) -> Tensor {
        let b = x.size()[0];
        let i = x.size()[1];
        let j = x.size()[2];
        let h = self.no_heads;
        let d = self.c_hidden;

        let q = self.mha_linear_q.forward(x);
        let k = self.mha_linear_k.forward(x);
        let v = self.mha_linear_v.forward(x);

        let q = q
            .reshape(&[b, i, j, h, d])
            .transpose(2, 3)
            .to_kind(Kind::Float)
            / (d as f64).sqrt();
        let k = k
            .reshape(&[b, i, j, h, d])
            .transpose(2, 3)
            .to_kind(Kind::Float);
        let v = v
            .reshape(&[b, i, j, h, d])
            .transpose(2, 3)
            .to_kind(Kind::Float);

        let kt = k.transpose(-1, -2);
        let mut a = q.matmul(&kt);
        a = a + mask_bias.to_kind(Kind::Float);
        a = a + triangle_bias.to_kind(Kind::Float);
        a = a.softmax(-1, Kind::Float);

        let mut o = a.matmul(&v);
        o = o.transpose(2, 3);

        let g = self.mha_linear_g.forward(x).sigmoid();
        let g = g.reshape(&[b, i, j, h, d]).to_kind(o.kind());
        o = o * g;

        let o = o.reshape(&[b, i, j, h * d]);
        self.mha_linear_o.forward(&o)
    }
}

pub type TriangleAttentionStartingNode = TriangleAttention;

impl TriangleAttention {
    pub fn new_ending_node<'a>(
        path: Path<'a>,
        c_in: i64,
        c_hidden: Option<i64>,
        no_heads: Option<i64>,
        inf: Option<f64>,
        device: Device,
    ) -> Self {
        Self::new(path, c_in, c_hidden, no_heads, Some(false), inf, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_triangle_attention_starting_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let c_in = 128;
        let c_hidden = 64;
        let no_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = TriangleAttention::new(
            vs.root(),
            c_in,
            Some(c_hidden),
            Some(no_heads),
            Some(true),
            None,
            device,
        );

        let x = Tensor::randn(&[batch_size, seq_len, seq_len, c_in], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&x, Some(&mask), None, false);

        assert_eq!(output.size(), vec![batch_size, seq_len, seq_len, c_in]);
    }

    #[test]
    fn test_triangle_attention_ending_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let c_in = 128;
        let c_hidden = 64;
        let no_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = TriangleAttention::new_ending_node(
            vs.root(),
            c_in,
            Some(c_hidden),
            Some(no_heads),
            None,
            device,
        );

        let x = Tensor::randn(&[batch_size, seq_len, seq_len, c_in], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&x, Some(&mask), None, false);

        assert_eq!(output.size(), vec![batch_size, seq_len, seq_len, c_in]);
    }
}
