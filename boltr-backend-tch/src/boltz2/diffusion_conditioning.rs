//! `DiffusionConditioning`: pre-computes the atom encoder output, pair biases,
//! and token-transformer bias that the score model consumes at every denoising step.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/diffusion_conditioning.py`

use tch::nn::{Module, Path};
use tch::{Device, Tensor};

use crate::tch_compat::{layer_norm_1d, linear_no_bias};

use super::encoders::{AtomEncoder, PairwiseConditioning};

/// Pre-computed conditioning tensors consumed by the score model at each step.
pub struct DiffusionConditioningOutput {
    /// Atom query features `[B, M, atom_s]`.
    pub q: Tensor,
    /// Atom conditioning features `[B, M, atom_s]`.
    pub c: Tensor,
    /// Windowed pair features `[B, K, W, H, atom_z]`.
    pub p: Tensor,
    /// Indexing matrix for windowed key construction.
    pub indexing_matrix: Tensor,
    /// Atom encoder bias (all depths concatenated) `[B, K, W, H, total_enc_heads]`.
    pub atom_enc_bias: Tensor,
    /// Atom decoder bias (all depths concatenated) `[B, K, W, H, total_dec_heads]`.
    pub atom_dec_bias: Tensor,
    /// Token transformer bias (all depths concatenated) `[B, N, N, total_trans_heads]`.
    pub token_trans_bias: Tensor,
}

/// `DiffusionConditioning` module.
///
/// Runs once per forward / sample call (not per diffusion step).
pub struct DiffusionConditioning {
    pairwise_conditioner: PairwiseConditioning,
    atom_encoder: AtomEncoder,
    /// Per-layer `LayerNorm → Linear(atom_z, heads)` for atom encoder depth.
    atom_enc_proj_z: Vec<(tch::nn::LayerNorm, tch::nn::Linear)>,
    /// Per-layer for atom decoder depth.
    atom_dec_proj_z: Vec<(tch::nn::LayerNorm, tch::nn::Linear)>,
    /// Per-layer for token transformer depth.
    token_trans_proj_z: Vec<(tch::nn::LayerNorm, tch::nn::Linear)>,
}

impl DiffusionConditioning {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        token_s: i64,
        token_z: i64,
        atom_s: i64,
        atom_z: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        atom_encoder_depth: i64,
        atom_encoder_heads: i64,
        token_transformer_depth: i64,
        token_transformer_heads: i64,
        atom_decoder_depth: i64,
        atom_decoder_heads: i64,
        atom_feature_dim: i64,
        conditioning_transition_layers: i64,
        device: Device,
    ) -> Self {
        let pairwise_conditioner = PairwiseConditioning::new(
            path.sub("pairwise_conditioner"),
            token_z,
            token_z,
            conditioning_transition_layers,
            2,
            device,
        );

        let atom_encoder = AtomEncoder::new(
            path.sub("atom_encoder"),
            atom_s,
            atom_z,
            token_s,
            token_z,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_feature_dim,
            true, // structure_prediction
        );

        let mut atom_enc_proj_z = Vec::new();
        for i in 0..atom_encoder_depth {
            let ln = layer_norm_1d(
                path.sub("atom_enc_proj_z").sub(format!("{i}")).sub("0"),
                atom_z,
            );
            let l = linear_no_bias(
                path.sub("atom_enc_proj_z").sub(format!("{i}")).sub("1"),
                atom_z,
                atom_encoder_heads,
            );
            atom_enc_proj_z.push((ln, l));
        }

        let mut atom_dec_proj_z = Vec::new();
        for i in 0..atom_decoder_depth {
            let ln = layer_norm_1d(
                path.sub("atom_dec_proj_z").sub(format!("{i}")).sub("0"),
                atom_z,
            );
            let l = linear_no_bias(
                path.sub("atom_dec_proj_z").sub(format!("{i}")).sub("1"),
                atom_z,
                atom_decoder_heads,
            );
            atom_dec_proj_z.push((ln, l));
        }

        let mut token_trans_proj_z = Vec::new();
        for i in 0..token_transformer_depth {
            let ln = layer_norm_1d(
                path.sub("token_trans_proj_z").sub(format!("{i}")).sub("0"),
                token_z,
            );
            let l = linear_no_bias(
                path.sub("token_trans_proj_z").sub(format!("{i}")).sub("1"),
                token_z,
                token_transformer_heads,
            );
            token_trans_proj_z.push((ln, l));
        }

        Self {
            pairwise_conditioner,
            atom_encoder,
            atom_enc_proj_z,
            atom_dec_proj_z,
            token_trans_proj_z,
        }
    }

    /// Run conditioning. Atom feature tensors are passed individually.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_trunk: &Tensor,
        z_trunk: &Tensor,
        relative_position_encoding: &Tensor,
        ref_pos: &Tensor,
        ref_charge: &Tensor,
        ref_element: &Tensor,
        atom_pad_mask: &Tensor,
        ref_space_uid: &Tensor,
        atom_to_token: &Tensor,
    ) -> DiffusionConditioningOutput {
        let z = self
            .pairwise_conditioner
            .forward(z_trunk, relative_position_encoding);

        let (q, c, p, indexing_matrix) = self.atom_encoder.forward(
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
            Some(s_trunk),
            Some(&z),
        );

        // Compute biases by concatenating per-layer projections
        let atom_enc_bias = {
            let parts: Vec<Tensor> = self
                .atom_enc_proj_z
                .iter()
                .map(|(ln, l)| l.forward(&ln.forward(&p)))
                .collect();
            Tensor::cat(&parts, -1)
        };

        let atom_dec_bias = {
            let parts: Vec<Tensor> = self
                .atom_dec_proj_z
                .iter()
                .map(|(ln, l)| l.forward(&ln.forward(&p)))
                .collect();
            Tensor::cat(&parts, -1)
        };

        let token_trans_bias = {
            let parts: Vec<Tensor> = self
                .token_trans_proj_z
                .iter()
                .map(|(ln, l)| l.forward(&ln.forward(&z)))
                .collect();
            Tensor::cat(&parts, -1)
        };

        DiffusionConditioningOutput {
            q,
            c,
            p,
            indexing_matrix,
            atom_enc_bias,
            atom_dec_bias,
            token_trans_bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::Kind;

    #[test]
    fn diffusion_conditioning_output_shapes() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);

        let token_s = 32_i64;
        let token_z = 16_i64;
        let atom_s = 16_i64;
        let atom_z = 8_i64;
        let w = 4_i64;
        let h = 8_i64;
        let enc_depth = 2_i64;
        let enc_heads = 2_i64;
        let dec_depth = 2_i64;
        let dec_heads = 2_i64;
        let trans_depth = 2_i64;
        let trans_heads = 4_i64;
        let atom_feat_dim = 3 + 1 + 4; // ref_pos(3) + charge(1) + element(4)

        let dc = DiffusionConditioning::new(
            vs.root().sub("diffusion_conditioning"),
            token_s,
            token_z,
            atom_s,
            atom_z,
            w,
            h,
            enc_depth,
            enc_heads,
            trans_depth,
            trans_heads,
            dec_depth,
            dec_heads,
            atom_feat_dim,
            2,
            device,
        );

        let b = 1_i64;
        let n_tokens = 4_i64;
        let n_atoms = 8_i64; // must be divisible by W

        let s_trunk = Tensor::randn(&[b, n_tokens, token_s], (Kind::Float, device));
        let z_trunk = Tensor::randn(&[b, n_tokens, n_tokens, token_z], (Kind::Float, device));
        let rel_pos = Tensor::randn(&[b, n_tokens, n_tokens, token_z], (Kind::Float, device));
        let ref_pos = Tensor::randn(&[b, n_atoms, 3], (Kind::Float, device));
        let ref_charge = Tensor::randn(&[b, n_atoms], (Kind::Float, device));
        let ref_element = Tensor::randn(&[b, n_atoms, 4], (Kind::Float, device));
        let atom_pad_mask = Tensor::ones(&[b, n_atoms], (Kind::Float, device));
        let ref_space_uid = Tensor::zeros(&[b, n_atoms], (Kind::Int64, device));
        let atom_to_token = Tensor::zeros(&[b, n_atoms, n_tokens], (Kind::Float, device));

        let out = dc.forward(
            &s_trunk,
            &z_trunk,
            &rel_pos,
            &ref_pos,
            &ref_charge,
            &ref_element,
            &atom_pad_mask,
            &ref_space_uid,
            &atom_to_token,
        );

        assert_eq!(out.q.size()[0], b);
        assert_eq!(out.q.size()[1], n_atoms);
        assert_eq!(out.q.size()[2], atom_s);

        assert_eq!(out.c.size()[0], b);
        assert_eq!(out.c.size()[1], n_atoms);

        let k = n_atoms / w;
        assert_eq!(
            out.atom_enc_bias.size(),
            vec![b, k, w, h, enc_depth * enc_heads]
        );
        assert_eq!(
            out.atom_dec_bias.size(),
            vec![b, k, w, h, dec_depth * dec_heads]
        );
        assert_eq!(
            out.token_trans_bias.size(),
            vec![b, n_tokens, n_tokens, trans_depth * trans_heads]
        );
    }
}
