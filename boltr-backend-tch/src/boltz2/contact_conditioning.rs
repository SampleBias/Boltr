//! Contact / pocket conditioning on the pairwise stream (`z`), from Boltz `ContactConditioning`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py`, `const.contact_conditioning_info`.

use std::f64::consts::PI;

use tch::nn::{linear, Init, LinearConfig, Module, Path};
use tch::{Kind, Tensor};

/// `len(const.contact_conditioning_info)` in Boltz (`UNSPECIFIED` … `CONTACT` = 5 channels).
pub const CONTACT_CONDITIONING_CHANNELS: i64 = 5;

/// Algorithm 22-style Fourier embed of scalar positions (frozen random projection).
///
/// Python: `FourierEmbedding` in `encodersv2.py` — `Linear(1, dim)`, N(0,1) init, `requires_grad=False`.
pub struct FourierEmbedding {
    proj: tch::nn::Linear,
}

impl FourierEmbedding {
    pub fn new(path: Path<'_>, dim: i64) -> Self {
        let cfg = LinearConfig {
            ws_init: Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
            bs_init: Some(Init::Randn {
                mean: 0.,
                stdev: 1.,
            }),
            bias: true,
        };
        let proj = linear(path.sub("proj"), 1, dim, cfg);
        let _ = proj.ws.set_requires_grad(false);
        if let Some(ref b) = proj.bs {
            let _ = b.set_requires_grad(false);
        }
        Self { proj }
    }

    /// `times`: `[*, 1]` float; returns `[*, dim]`.
    pub fn forward(&self, times: &Tensor) -> Tensor {
        let x = self.proj.forward(times);
        x.g_mul_scalar(2.0 * PI).cos()
    }
}

/// Inputs matching `feats["contact_conditioning"]` and `feats["contact_threshold"]`.
pub struct ContactFeatures<'a> {
    /// `[B, N, N, 5]` float (Boltz one-hot / indicator layout).
    pub contact_conditioning: &'a Tensor,
    /// `[B, N, N]` float (Ångström cutoff, etc.).
    pub contact_threshold: &'a Tensor,
}

pub struct ContactConditioning {
    fourier_embedding: FourierEmbedding,
    encoder: tch::nn::Linear,
    encoding_unspecified: Tensor,
    encoding_unselected: Tensor,
    cutoff_min: f64,
    cutoff_max: f64,
    token_z: i64,
}

impl ContactConditioning {
    pub fn new(path: Path<'_>, token_z: i64, cutoff_min: f64, cutoff_max: f64) -> Self {
        let enc_in = token_z + CONTACT_CONDITIONING_CHANNELS - 1;
        let fourier_embedding = FourierEmbedding::new(path.sub("fourier_embedding"), token_z);
        let encoder = linear(
            path.sub("encoder"),
            enc_in,
            token_z,
            LinearConfig::default(),
        );
        let encoding_unspecified = path.var("encoding_unspecified", &[token_z], Init::Const(0.));
        let encoding_unselected = path.var("encoding_unselected", &[token_z], Init::Const(0.));
        Self {
            fourier_embedding,
            encoder,
            encoding_unspecified,
            encoding_unselected,
            cutoff_min,
            cutoff_max,
            token_z,
        }
    }

    /// Pairwise bias `[B, N, N, token_z]`.
    pub fn forward(&self, feats: &ContactFeatures<'_>) -> Tensor {
        let cc_full = feats.contact_conditioning;
        let cc_tail = cc_full.narrow(3, 2, CONTACT_CONDITIONING_CHANNELS - 2);

        let thresh = feats.contact_threshold;
        let denom = self.cutoff_max - self.cutoff_min;
        let contact_threshold_normalized = thresh.g_sub_scalar(self.cutoff_min).g_div_scalar(denom);

        let sz = contact_threshold_normalized.size();
        let flat = contact_threshold_normalized.flatten(0, 2);
        let t_in = flat.unsqueeze(1);
        let fourier_flat = self.fourier_embedding.forward(&t_in);
        let contact_threshold_fourier = fourier_flat.view([sz[0], sz[1], sz[2], self.token_z]);

        let norm_exp = contact_threshold_normalized.unsqueeze(-1);
        let pieces = [
            cc_tail.to_kind(Kind::Float),
            norm_exp,
            contact_threshold_fourier,
        ];
        let cat = Tensor::cat(&pieces, -1);
        let encoded = self.encoder.forward(&cat);

        let head = cc_full.narrow(3, 0, 2);
        let sum01 = head.sum_dim_intlist(&[-1i64][..], true, Kind::Float);
        let one = sum01.ones_like();
        let mult = &one - &sum01;
        let mut out = encoded * mult;

        let ch0 = cc_full.narrow(3, 0, 1).to_kind(Kind::Float);
        let ch1 = cc_full.narrow(3, 1, 1).to_kind(Kind::Float);
        let eu = self.encoding_unspecified.view([1, 1, 1, self.token_z]);
        let es = self.encoding_unselected.view([1, 1, 1, self.token_z]);
        out = out + eu * ch0 + es * ch1;
        out
    }

    pub fn token_z(&self) -> i64 {
        self.token_z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::Device;

    #[test]
    fn contact_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 7_i64;
        let vs = VarStore::new(device);
        let m = ContactConditioning::new(vs.root(), token_z, 4.0, 20.0);
        let cc = Tensor::zeros(
            &[b, n, n, CONTACT_CONDITIONING_CHANNELS],
            (Kind::Float, device),
        );
        let ct = Tensor::ones(&[b, n, n], (Kind::Float, device)) * 10.0;
        let feats = ContactFeatures {
            contact_conditioning: &cc,
            contact_threshold: &ct,
        };
        let z = m.forward(&feats);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }
}
