//! Boltz `InputEmbedder` — trunk-side atom → token embedding.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py` (`InputEmbedder`).
//! VarStore prefix: **`input_embedder`** (`input_embedder.atom_encoder`, `atom_enc_proj_z`, …).
//!
//! **Implementation status:** [`InputEmbedder::new`] wires the full Python path
//! `AtomEncoder` → LayerNorm+`atom_enc_proj_z` → `AtomAttentionEncoder` →
//! `res_type_encoding` + `msa_profile_encoding`. Use [`InputEmbedder::new_tail_only`] only for
//! narrow linear tests. Python parity: opt-in `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1` +
//! [`tests/input_embedder_golden.rs`](../../tests/input_embedder_golden.rs).
//!
//! Use [`AtomEncoderPlaceholder`] only for shape tests when bypassing the atom stack.

use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

use crate::tch_compat::{layer_norm_1d, linear_no_bias};

use super::encoders::{AtomAttentionEncoder, AtomEncoder, AtomEncoderBatchFeats, AtomEncoderFlags};

/// `len(const.tokens)` in Boltz (`boltz-reference/src/boltz/data/const.py`).
/// Must match `boltr_io::boltz_const::NUM_TOKENS` / Boltz `const.num_tokens`.
pub const BOLTZ_NUM_TOKENS: i64 = 33;

/// `num_tokens + 1` for `torch.cat([profile, deletion_mean.unsqueeze(-1)], dim=-1)`.
pub const BOLTZ_MSA_PROFILE_IN: i64 = BOLTZ_NUM_TOKENS + 1;

/// `res_type_encoding` + `msa_profile_encoding` under `input_embedder/`.
/// Zero `a` tensor for tests / scaffolding (`trunkv2.py` atom attention output).
#[derive(Debug, Default, Clone, Copy)]
pub struct AtomEncoderPlaceholder;

impl AtomEncoderPlaceholder {
    #[must_use]
    pub fn zeros_a(batch: i64, n_tokens: i64, token_s: i64, device: Device) -> Tensor {
        Tensor::zeros(&[batch, n_tokens, token_s], (Kind::Float, device))
    }
}

/// Collate-aligned tensors for [`InputEmbedder::forward`].
pub struct InputEmbedderFeats<'a> {
    pub ref_pos: &'a Tensor,
    pub ref_charge: &'a Tensor,
    pub ref_element: &'a Tensor,
    pub atom_pad_mask: &'a Tensor,
    pub ref_space_uid: &'a Tensor,
    pub atom_to_token: &'a Tensor,
    pub res_type: &'a Tensor,
    pub profile: &'a Tensor,
    pub deletion_mean: &'a Tensor,
    pub profile_affinity: Option<&'a Tensor>,
    pub deletion_mean_affinity: Option<&'a Tensor>,
    pub atom_encoder_batch: Option<&'a AtomEncoderBatchFeats<'a>>,
}

pub struct InputEmbedder {
    atom_encoder: Option<AtomEncoder>,
    atom_enc_proj_z_ln: Option<tch::nn::LayerNorm>,
    atom_enc_proj_z_lin: Option<tch::nn::Linear>,
    atom_attention_encoder: Option<AtomAttentionEncoder>,
    res_type_encoding: tch::nn::Linear,
    msa_profile_encoding: tch::nn::Linear,
    token_s: i64,
    device: Device,
}

impl InputEmbedder {
    /// Full trunk embedder: `AtomEncoder` → `atom_enc_proj_z` → `AtomAttentionEncoder` → token linears.
    ///
    /// `structure_prediction=false` for both atom encoder and atom attention (matches Python trunk).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        token_s: i64,
        token_z: i64,
        atom_s: i64,
        atom_z: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        atom_feature_dim: i64,
        atom_encoder_depth: i64,
        atom_encoder_heads: i64,
        flags: AtomEncoderFlags,
        device: Device,
    ) -> Self {
        let atom_encoder = Some(AtomEncoder::new(
            path.sub("atom_encoder"),
            atom_s,
            atom_z,
            token_s,
            token_z,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_feature_dim,
            false,
            flags,
        ));
        let atom_enc_proj_z_ln = Some(layer_norm_1d(path.sub("atom_enc_proj_z").sub("0"), atom_z));
        let atom_enc_proj_z_lin = Some(linear_no_bias(
            path.sub("atom_enc_proj_z").sub("1"),
            atom_z,
            atom_encoder_depth * atom_encoder_heads,
        ));
        let atom_attention_encoder = Some(AtomAttentionEncoder::new(
            path.sub("atom_attention_encoder"),
            atom_s,
            token_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_encoder_depth,
            atom_encoder_heads,
            false,
            device,
        ));
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
            atom_encoder,
            atom_enc_proj_z_ln,
            atom_enc_proj_z_lin,
            atom_attention_encoder,
            res_type_encoding,
            msa_profile_encoding,
            token_s,
            device,
        }
    }

    /// Token linears only (no atom stack). For minimal unit tests.
    pub fn new_tail_only(path: Path<'_>, token_s: i64, device: Device) -> Self {
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
            atom_encoder: None,
            atom_enc_proj_z_ln: None,
            atom_enc_proj_z_lin: None,
            atom_attention_encoder: None,
            res_type_encoding,
            msa_profile_encoding,
            token_s,
            device,
        }
    }

    pub fn token_s(&self) -> i64 {
        self.token_s
    }

    pub fn has_atom_stack(&self) -> bool {
        self.atom_encoder.is_some()
    }

    /// Full forward: Python `InputEmbedder.forward(feats, affinity=…)`.
    pub fn forward(&self, feats: &InputEmbedderFeats<'_>, affinity: bool) -> Tensor {
        let atom_encoder = self
            .atom_encoder
            .as_ref()
            .expect("InputEmbedder::forward requires full embedder (use InputEmbedder::new, not new_tail_only)");
        let ln = self
            .atom_enc_proj_z_ln
            .as_ref()
            .expect("atom_enc_proj_z_ln");
        let proj = self
            .atom_enc_proj_z_lin
            .as_ref()
            .expect("atom_enc_proj_z_lin");
        let attn = self
            .atom_attention_encoder
            .as_ref()
            .expect("atom_attention_encoder");

        let (q, c, p, indexing_matrix) = atom_encoder.forward(
            feats.ref_pos,
            feats.ref_charge,
            feats.ref_element,
            feats.atom_pad_mask,
            feats.ref_space_uid,
            feats.atom_to_token,
            None,
            None,
            feats.atom_encoder_batch,
        );
        let atom_enc_bias = proj.forward(&ln.forward(&p));
        let b = feats.ref_pos.size()[0];
        let m = feats.ref_pos.size()[1];
        let r_dummy = Tensor::zeros(&[b, m, 3], (Kind::Float, self.device));
        let (a, _, _) = attn.forward(
            &q,
            &c,
            &atom_enc_bias,
            feats.atom_pad_mask,
            feats.atom_to_token,
            &r_dummy,
            1,
            &indexing_matrix,
        );
        let (profile, deletion_mean) = if affinity {
            (
                feats
                    .profile_affinity
                    .expect("profile_affinity when affinity=true"),
                feats
                    .deletion_mean_affinity
                    .expect("deletion_mean_affinity when affinity=true"),
            )
        } else {
            (feats.profile, feats.deletion_mean)
        };
        let dm = deletion_mean.unsqueeze(-1);
        let msa_in = Tensor::cat(&[profile.shallow_clone(), dm], -1);
        a + self.res_type_encoding.forward(feats.res_type)
            + self.msa_profile_encoding.forward(&msa_in)
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
        let emb = InputEmbedder::new_tail_only(vs.root().sub("input_embedder"), token_s, device);
        let a = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let res = Tensor::randn(&[b, n, BOLTZ_NUM_TOKENS], (Kind::Float, device));
        let prof = Tensor::randn(&[b, n, BOLTZ_NUM_TOKENS], (Kind::Float, device));
        let del = Tensor::randn(&[b, n], (Kind::Float, device));
        let s = emb.forward_with_atom_repr(&a, &res, &prof, &del);
        assert_eq!(s.size(), vec![b, n, token_s]);
    }
}
