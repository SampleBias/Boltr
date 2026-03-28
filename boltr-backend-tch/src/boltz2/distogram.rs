//! Trunk output heads: `DistogramModule` and `BFactorModule`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py`

use tch::nn::{linear, LinearConfig, Module, Path};
use tch::Tensor;

/// Predicted distogram logits from the pair representation `z`.
///
/// Python: `DistogramModule(token_z, num_bins, num_distograms=1)`.
pub struct DistogramModule {
    distogram: tch::nn::Linear,
    num_distograms: i64,
    num_bins: i64,
}

impl DistogramModule {
    pub fn new(path: Path<'_>, token_z: i64, num_bins: i64, num_distograms: Option<i64>) -> Self {
        let num_distograms = num_distograms.unwrap_or(1);
        let distogram = linear(
            path.sub("distogram"),
            token_z,
            num_distograms * num_bins,
            LinearConfig::default(),
        );
        Self {
            distogram,
            num_distograms,
            num_bins,
        }
    }

    /// `z + z^T` → linear → reshape `[B, N, N, num_distograms, num_bins]`.
    pub fn forward(&self, z: &Tensor) -> Tensor {
        let z_sym = z + z.transpose(1, 2);
        let out = self.distogram.forward(&z_sym);
        let size = z_sym.size();
        out.reshape(&[size[0], size[1], size[2], self.num_distograms, self.num_bins])
    }

    pub fn num_bins(&self) -> i64 {
        self.num_bins
    }

    pub fn num_distograms(&self) -> i64 {
        self.num_distograms
    }
}

/// Predicted B-factor histogram from the single representation `s`.
///
/// Python: `BFactorModule(token_s, num_bins)`.
pub struct BFactorModule {
    bfactor: tch::nn::Linear,
    num_bins: i64,
}

impl BFactorModule {
    pub fn new(path: Path<'_>, token_s: i64, num_bins: i64) -> Self {
        let bfactor = linear(
            path.sub("bfactor"),
            token_s,
            num_bins,
            LinearConfig::default(),
        );
        Self { bfactor, num_bins }
    }

    /// `s → linear → [B, N, num_bins]`.
    pub fn forward(&self, s: &Tensor) -> Tensor {
        self.bfactor.forward(s)
    }

    pub fn num_bins(&self) -> i64 {
        self.num_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::{Device, Kind};

    #[test]
    fn distogram_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let token_z = 32_i64;
        let num_bins = 64_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = DistogramModule::new(vs.root().sub("distogram_module"), token_z, num_bins, None);
        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let out = m.forward(&z);
        assert_eq!(out.size(), vec![b, n, n, 1, num_bins]);
    }

    #[test]
    fn distogram_multi_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let token_z = 32_i64;
        let num_bins = 64_i64;
        let num_distograms = 3_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = DistogramModule::new(
            vs.root().sub("distogram_module"),
            token_z,
            num_bins,
            Some(num_distograms),
        );
        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let out = m.forward(&z);
        assert_eq!(out.size(), vec![b, n, n, num_distograms, num_bins]);
    }

    #[test]
    fn bfactor_forward_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let token_s = 64_i64;
        let num_bins = 64_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = BFactorModule::new(vs.root().sub("bfactor_module"), token_s, num_bins);
        let s = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let out = m.forward(&s);
        assert_eq!(out.size(), vec![b, n, num_bins]);
    }
}
