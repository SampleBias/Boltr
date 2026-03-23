//! Boltz `InputEmbedder` — **partial** Rust port.
//!
//! Implements the **token / MSA linear branch** (`res_type_encoding`, `msa_profile_encoding`) that
//! Python adds to the atom-attention output `a`. Full `AtomEncoder` + `AtomAttentionEncoder` are
//! still TBD; callers pass `a` with shape `[B, N, token_s]`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py` (`InputEmbedder`).
//! VarStore prefix: **`input_embedder`** (keys `input_embedder.res_type_encoding.weight`, etc.).

use tch::nn::{linear, LinearConfig, Module, Path};
use tch::Tensor;

/// `len(const.tokens)` in Boltz (`boltz-reference/src/boltz/data/const.py`).
pub const BOLTZ_NUM_TOKENS: i64 = 33;

/// `num_tokens + 1` for `torch.cat([profile, deletion_mean.unsqueeze(-1)], dim=-1)`.
pub const BOLTZ_MSA_PROFILE_IN: i64 = BOLTZ_NUM_TOKENS + 1;

/// `res_type_encoding` + `msa_profile_encoding` under `input_embedder/`.
pub struct InputEmbedder {
    res_type_encoding: tch::nn::Linear,
    msa_profile_encoding: tch::nn::Linear,
    token_s: i64,
}

impl InputEmbedder {
    pub fn new(path: Path<'_>, token_s: i64) -> Self {
        let res_type_encoding = linear(
            path.sub("res_type_encoding"),
            BOLTZ_NUM_TOKENS,
            token_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let msa_profile_encoding = linear(
            path.sub("msa_profile_encoding"),
            BOLTZ_MSA_PROFILE_IN,
            token_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        Self {
            res_type_encoding,
            msa_profile_encoding,
            token_s,
        }
    }

    pub fn token_s(&self) -> i64 {
        self.token_s
    }

    /// Python `s = a + res_type_encoding(res_type) + msa_profile_encoding(cat(profile, deletion_mean))`.
    ///
    /// * `atom_attn_out` (`a`): `[B, N, token_s]`
    /// * `res_type`: `[B, N, num_tokens]` float (one-hot or soft)
    /// * `profile`: `[B, N, num_tokens]` float
    /// * `deletion_mean`: `[B, N]` float
    pub fn forward_with_atom_repr(
        &self,
        atom_attn_out: &Tensor,
        res_type: &Tensor,
        profile: &Tensor,
        deletion_mean: &Tensor,
    ) -> Tensor {
        let dm = deletion_mean.unsqueeze(-1);
        let msa_in = Tensor::cat(&[profile.shallow_clone(), dm], -1);
        atom_attn_out
            + self.res_type_encoding.forward(res_type)
            + self.msa_profile_encoding.forward(&msa_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::{Device, Kind};

    #[test]
    fn forward_with_atom_repr_shapes() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 384_i64;
        let b = 2_i64;
        let n = 9_i64;
        let vs = VarStore::new(device);
        let emb = InputEmbedder::new(vs.root().sub("input_embedder"), token_s);
        let a = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let res = Tensor::randn(&[b, n, BOLTZ_NUM_TOKENS], (Kind::Float, device));
        let prof = Tensor::randn(&[b, n, BOLTZ_NUM_TOKENS], (Kind::Float, device));
        let del = Tensor::randn(&[b, n], (Kind::Float, device));
        let s = emb.forward_with_atom_repr(&a, &res, &prof, &del);
        assert_eq!(s.size(), vec![b, n, token_s]);
    }
}
