//! Featurizer batch → trunk entry-point shapes (§5.10).
//!
//! Full Boltz2 `s_inputs` are `atom_encoder` + `atom_attention_encoder` + [`super::InputEmbedder`].
//! Until the atom stack is ported, tests use [`AtomEncoderPlaceholder::zeros_a`] and
//! [`Boltz2Model::forward_input_embedder`](super::Boltz2Model::forward_input_embedder) (or the
//! affinity profile variant when `profile_affinity` / `deletion_mean_affinity` are present in feats).

use tch::{Device, Kind, Tensor};

use super::input_embedder::AtomEncoderPlaceholder;

/// Zero `a` tensor with shape `[B, N, token_s]` (atom-attention output placeholder).
#[must_use]
pub fn zeros_atom_attention_out(batch: i64, n_tokens: i64, token_s: i64, device: Device) -> Tensor {
    AtomEncoderPlaceholder::zeros_a(batch, n_tokens, token_s, device)
}
