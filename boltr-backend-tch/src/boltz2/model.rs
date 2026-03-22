//! Boltz2 high-level model: VarStore-backed layers aligned with PyTorch module names.

use std::path::Path;

use anyhow::Result;
use tch::nn::{linear, LinearConfig, Module, VarStore};
use tch::{Device, Tensor};

use crate::checkpoint::load_tensor_from_safetensors;

/// Boltz2 inference skeleton. Default `token_s` matches common Boltz2 checkpoints (384).
pub struct Boltz2Model {
    device: Device,
    var_store: VarStore,
    s_init: tch::nn::Linear,
    token_s: i64,
}

impl Boltz2Model {
    /// `token_s` must match the checkpoint (Boltz2 trunk width).
    pub fn new(device: Device, token_s: i64) -> Result<Self> {
        let var_store = VarStore::new(device);
        let root = var_store.root();
        let s_init = linear(
            root.sub("s_init"),
            token_s,
            token_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        Ok(Self {
            device,
            var_store,
            s_init,
            token_s,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    /// Load `s_init.weight` from a safetensors file (exported from Lightning `state_dict`).
    pub fn load_s_init_from_safetensors(&mut self, path: &Path) -> Result<()> {
        let w = load_tensor_from_safetensors(path, "s_init.weight", self.device)?;
        self.s_init.ws.copy_(&w);
        Ok(())
    }

    /// Apply the same transform as Python `self.s_init` on per-token features `[N, token_s]`.
    pub fn forward_s_init(&self, s: &Tensor) -> Tensor {
        self.s_init.forward(s)
    }

    /// Full structure forward is not wired until featurization + trunk + diffusion are ported.
    pub fn forward_structure(&self, _feats: &Tensor) -> Result<Tensor> {
        anyhow::bail!(
            "Boltz2 full structure forward not yet implemented; complete trunk + AtomDiffusion port"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s_init_linear_runs() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let ts = 384_i64;
        let mut m = Boltz2Model::new(device, ts).unwrap();
        let x = Tensor::randn(&[7, ts], (tch::Kind::Float, device));
        let y = m.forward_s_init(&x);
        assert_eq!(y.size(), vec![7, ts]);
    }
}
