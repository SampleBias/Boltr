//! Pair-weighted averaging (`PairWeightedAveraging` in Boltz).
//!
//! Reference: `boltz-reference/src/boltz/model/layers/pair_averaging.py`

use tch::nn::{linear, LayerNorm, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

pub struct PairWeightedAveraging {
    c_m: i64,
    c_h: i64,
    num_heads: i64,
    inf: f64,
    norm_m: tch::nn::LayerNorm,
    norm_z: tch::nn::LayerNorm,
    proj_m: tch::nn::Linear,
    proj_g: tch::nn::Linear,
    proj_z: tch::nn::Linear,
    proj_o: tch::nn::Linear,
}

impl PairWeightedAveraging {
    pub fn new<'a>(
        path: Path<'a>,
        c_m: i64,
        c_z: i64,
        c_h: i64,
        num_heads: i64,
        inf: Option<f64>,
        device: Device,
    ) -> Self {
        let inf = inf.unwrap_or(1e6);
        let norm_m = LayerNorm::new(path.sub("norm_m"), vec![c_m], c_m as f64 * 1e-5, true);
        let norm_z = LayerNorm::new(path.sub("norm_z"), vec![c_z], c_z as f64 * 1e-5, true);
        let proj_m = linear(
            path.sub("proj_m"),
            c_m,
            c_h * num_heads,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let proj_g = linear(
            path.sub("proj_g"),
            c_m,
            c_h * num_heads,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let proj_z = linear(
            path.sub("proj_z"),
            c_z,
            num_heads,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let proj_o = linear(
            path.sub("proj_o"),
            c_h * num_heads,
            c_m,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let _ = device;
        Self {
            c_m,
            c_h,
            num_heads,
            inf,
            norm_m,
            norm_z,
            proj_m,
            proj_g,
            proj_z,
            proj_o,
        }
    }

    /// `m`: `[B, S, N, c_m]`, `z`: `[B, N, N, c_z]`, `pair_mask`: `[B, N, N]` (float 0/1).
    pub fn forward(&self, m: &Tensor, z: &Tensor, pair_mask: &Tensor) -> Tensor {
        let m = self.norm_m.forward(m);
        let z = self.norm_z.forward(z);

        let v = self.proj_m.forward(&m);
        let v = v.reshape(&[
            v.size()[0],
            v.size()[1],
            v.size()[2],
            self.num_heads,
            self.c_h,
        ]);
        let v = v.permute(&[0, 3, 1, 2, 4]);

        let mut b = self.proj_z.forward(&z);
        b = b.permute(&[0, 3, 1, 2]);
        let mask_exp = pair_mask.unsqueeze(1);
        b = b + (1.0 - &mask_exp) * (-self.inf);
        let w = b.softmax(-1, Kind::Float);

        let g = self.proj_g.forward(&m).sigmoid();

        let o = Tensor::einsum("bhij,bhsjd->bhsid", &[&w, &v], None::<i64>);
        let o = o.permute(&[0, 2, 3, 1, 4]);
        let o = o.reshape(&[
            o.size()[0],
            o.size()[1],
            o.size()[2],
            self.num_heads * self.c_h,
        ]);
        self.proj_o.forward(&(g * o))
    }
}
