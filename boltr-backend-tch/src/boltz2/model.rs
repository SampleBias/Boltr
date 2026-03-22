//! Boltz2 high-level model: one `VarStore` aligned with Lightning `Boltz2` root names.
//!
//! Python layout reference: `boltz-reference/src/boltz/model/models/boltz2.py`
//! — `s_init`, `z_init_*`, recycling, and `pairformer_module` live on the **model root**
//! (not inside a nested `trunk` submodule).

use std::path::Path;

use anyhow::Result;
use tch::nn::VarStore;
use tch::{Device, Tensor};

use super::trunk::TrunkV2;
use crate::checkpoint::load_tensor_from_safetensors;

/// Boltz2 inference skeleton: owns trunk + pairformer under a single [`VarStore`].
pub struct Boltz2Model {
    device: Device,
    var_store: VarStore,
    trunk: TrunkV2,
    token_s: i64,
    token_z: i64,
}

impl Boltz2Model {
    /// Build with default pairformer depth (`4` blocks) and `token_z = 128`.
    pub fn new(device: Device, token_s: i64) -> Self {
        Self::with_options(device, token_s, 128, None)
    }

    /// Full constructor: `num_pairformer_blocks` defaults to `4` when `None`.
    pub fn with_options(
        device: Device,
        token_s: i64,
        token_z: i64,
        num_pairformer_blocks: Option<i64>,
    ) -> Self {
        let var_store = VarStore::new(device);
        let trunk = TrunkV2::new(
            &var_store,
            Some(token_s),
            Some(token_z),
            num_pairformer_blocks,
            device,
        );
        Self {
            device,
            var_store,
            trunk,
            token_s,
            token_z,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn token_s(&self) -> i64 {
        self.token_s
    }

    pub fn token_z(&self) -> i64 {
        self.token_z
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn trunk(&self) -> &TrunkV2 {
        &self.trunk
    }

    pub fn trunk_mut(&mut self) -> &mut TrunkV2 {
        &mut self.trunk
    }

    /// Load `s_init.weight` from a safetensors file (exported from Lightning `state_dict`).
    pub fn load_s_init_from_safetensors(&mut self, path: &Path) -> Result<()> {
        let w = load_tensor_from_safetensors(path, "s_init.weight", self.device)?;
        self.trunk.load_s_init_weight(&w);
        Ok(())
    }

    /// Same transform as Python `self.s_init(s_inputs)` on `[B, N, token_s]`.
    pub fn forward_s_init(&self, s: &Tensor) -> Tensor {
        self.trunk.apply_s_init(s)
    }

    /// Trunk + pairformer + recycling, matching Python’s inner loop for a **pre-embedded** batch.
    ///
    /// `s_inputs` is the output of Python `input_embedder(feats)` — shape `[B, N, token_s]`.
    pub fn forward_trunk(
        &self,
        s_inputs: &Tensor,
        recycling_steps: Option<i64>,
    ) -> Result<(Tensor, Tensor)> {
        self.trunk.forward(s_inputs, recycling_steps)
    }

    /// Full structure forward is not wired until diffusion + featurizer match Python.
    pub fn forward_structure(&self, _feats: &Tensor) -> Result<Tensor> {
        anyhow::bail!(
            "Boltz2 full structure forward not yet implemented; use forward_trunk(s_inputs, ...) with embedder outputs"
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
        let mut m = Boltz2Model::new(device, ts);
        let x = Tensor::randn(&[7, ts], (tch::Kind::Float, device));
        let y = m.forward_s_init(&x);
        assert_eq!(y.size(), vec![7, ts]);
    }

    #[test]
    fn trunk_forward_runs_small() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));
        let s_in = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let (s, z) = m.forward_trunk(&s_in, Some(0)).unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }
}
