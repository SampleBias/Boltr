//! Outer product mean used inside Boltz2 `MSALayer` (distinct from `outer_product_mean.rs`).
//!
//! Reference: `boltz-reference/src/boltz/model/layers/outer_product_mean.py`

use tch::nn::{linear, LayerNorm, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

pub struct OuterProductMeanMsa {
    c_hidden: i64,
    norm: tch::nn::LayerNorm,
    proj_a: tch::nn::Linear,
    proj_b: tch::nn::Linear,
    proj_o: tch::nn::Linear,
}

impl OuterProductMeanMsa {
    pub fn new<'a>(path: Path<'a>, c_in: i64, c_hidden: i64, c_out: i64, device: Device) -> Self {
        let norm = LayerNorm::new(path.sub("norm"), vec![c_in], c_in as f64 * 1e-5, true);
        let proj_a = linear(
            path.sub("proj_a"),
            c_in,
            c_hidden,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let proj_b = linear(
            path.sub("proj_b"),
            c_in,
            c_hidden,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let proj_o = linear(
            path.sub("proj_o"),
            c_hidden * c_hidden,
            c_out,
            LinearConfig {
                bias: true,
                ..Default::default()
            },
        );
        let _ = device;
        Self {
            c_hidden,
            norm,
            proj_a,
            proj_b,
            proj_o,
        }
    }

    /// `m`: `[B, S, N, c_in]`, `msa_mask`: `[B, S, N]` (numeric mask, same dtype as m after norm).
    pub fn forward(&self, m: &Tensor, msa_mask: &Tensor) -> Tensor {
        let mask_4 = msa_mask.unsqueeze(-1).to_kind(m.kind());
        let m = self.norm.forward(m);
        let a = self.proj_a.forward(&m) * &mask_4;
        let b = self.proj_b.forward(&m) * mask_4;

        let pair_mask = mask_4.unsqueeze(2) * mask_4.unsqueeze(3);
        let num_mask = pair_mask.sum_dim_intlist(&[1], false, Kind::Float).clamp_min(1.0);

        let a_f = a.to_kind(Kind::Float);
        let b_f = b.to_kind(Kind::Float);
        let z = Tensor::einsum("bsic,bsjd->bijcd", &[&a_f, &b_f], None::<i64>);
        let z = z.reshape(&[z.size()[0], z.size()[1], z.size()[2], self.c_hidden * self.c_hidden]);
        let z = z / &num_mask;
        self.proj_o.forward(&z.to_kind(m.kind()))
    }
}
