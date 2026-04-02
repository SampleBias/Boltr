//! Encoder modules for diffusion conditioning and the score model.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/encodersv2.py`

use std::f64::consts::PI;

use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

use crate::layers::Transition;
use crate::tch_compat::{layer_norm_1d, linear_no_bias};

use super::transformers::{AtomTransformer, DiffusionTransformer};

// ---------------------------------------------------------------------------
// FourierEmbedding  (Algorithm 22)
// ---------------------------------------------------------------------------

/// Frozen random Fourier feature embedding for noise level σ.
pub struct FourierEmbedding {
    proj_weight: Tensor,
    proj_bias: Tensor,
}

impl FourierEmbedding {
    pub fn new(path: Path<'_>, dim: i64) -> Self {
        let proj = linear(path.sub("proj"), 1, dim, LinearConfig::default());
        let proj_weight = proj.ws.shallow_clone().set_requires_grad(false);
        let proj_bias = proj
            .bs
            .as_ref()
            .unwrap()
            .shallow_clone()
            .set_requires_grad(false);
        Self {
            proj_weight,
            proj_bias,
        }
    }

    /// `times: [B]` → `[B, dim]`.
    pub fn forward(&self, times: &Tensor) -> Tensor {
        let t = times.unsqueeze(-1); // [B, 1]
        let rand_proj = t.matmul(&self.proj_weight.tr()) + &self.proj_bias;
        (rand_proj * (2.0 * PI)).cos()
    }
}

// ---------------------------------------------------------------------------
// SingleConditioning  (Algorithm 21)
// ---------------------------------------------------------------------------

/// Conditions the score model on the trunk single representation + noise level.
pub struct SingleConditioning {
    norm_single: tch::nn::LayerNorm,
    single_embed: tch::nn::Linear,
    fourier_embed: FourierEmbedding,
    norm_fourier: tch::nn::LayerNorm,
    fourier_to_single: tch::nn::Linear,
    transitions: Vec<Transition>,
}

impl SingleConditioning {
    pub fn new(
        path: Path<'_>,
        sigma_data: f64,
        token_s: i64,
        dim_fourier: i64,
        num_transitions: i64,
        transition_expansion_factor: i64,
        device: Device,
    ) -> Self {
        let two_ts = 2 * token_s;
        let norm_single = layer_norm_1d(path.sub("norm_single"), two_ts);
        let single_embed = linear(
            path.sub("single_embed"),
            two_ts,
            two_ts,
            LinearConfig::default(),
        );
        let fourier_embed = FourierEmbedding::new(path.sub("fourier_embed"), dim_fourier);
        let norm_fourier = layer_norm_1d(path.sub("norm_fourier"), dim_fourier);
        let fourier_to_single = linear_no_bias(path.sub("fourier_to_single"), dim_fourier, two_ts);

        let mut transitions = Vec::new();
        for i in 0..num_transitions {
            transitions.push(Transition::new(
                path.sub("transitions").sub(format!("{i}")),
                two_ts,
                Some(transition_expansion_factor * two_ts),
                None,
                device,
            ));
        }

        Self {
            norm_single,
            single_embed,
            fourier_embed,
            norm_fourier,
            fourier_to_single,
            transitions,
        }
    }

    /// Returns `(s_conditioned, normed_fourier)`.
    ///
    /// * `times`: `[B]` noise level (already `c_noise`-scaled)
    /// * `s_trunk`: `[B, N, token_s]`
    /// * `s_inputs`: `[B, N, token_s]`
    pub fn forward(&self, times: &Tensor, s_trunk: &Tensor, s_inputs: &Tensor) -> (Tensor, Tensor) {
        let s = Tensor::cat(&[s_trunk, s_inputs], -1);
        let mut s = self.single_embed.forward(&self.norm_single.forward(&s));

        let fourier_embed = self.fourier_embed.forward(times);
        let normed_fourier = self.norm_fourier.forward(&fourier_embed);
        let fourier_to_single = self.fourier_to_single.forward(&normed_fourier);
        s = fourier_to_single.unsqueeze(1) + s;

        for transition in &self.transitions {
            let t = transition.forward(&s, None);
            s = t + s;
        }
        (s, normed_fourier)
    }
}

// ---------------------------------------------------------------------------
// PairwiseConditioning  (Algorithm 21)
// ---------------------------------------------------------------------------

/// Conditions the pairwise representation for diffusion.
pub struct PairwiseConditioning {
    dim_pairwise_init_proj_norm: tch::nn::LayerNorm,
    dim_pairwise_init_proj_linear: tch::nn::Linear,
    transitions: Vec<Transition>,
}

impl PairwiseConditioning {
    pub fn new(
        path: Path<'_>,
        token_z: i64,
        dim_token_rel_pos_feats: i64,
        num_transitions: i64,
        transition_expansion_factor: i64,
        device: Device,
    ) -> Self {
        let combined = token_z + dim_token_rel_pos_feats;
        let dim_pairwise_init_proj_norm =
            layer_norm_1d(path.sub("dim_pairwise_init_proj").sub("0"), combined);
        let dim_pairwise_init_proj_linear = linear_no_bias(
            path.sub("dim_pairwise_init_proj").sub("1"),
            combined,
            token_z,
        );

        let mut transitions = Vec::new();
        for i in 0..num_transitions {
            transitions.push(Transition::new(
                path.sub("transitions").sub(format!("{i}")),
                token_z,
                Some(transition_expansion_factor * token_z),
                None,
                device,
            ));
        }

        Self {
            dim_pairwise_init_proj_norm,
            dim_pairwise_init_proj_linear,
            transitions,
        }
    }

    /// `z_trunk [B, N, N, tz]` + `rel_pos [B, N, N, tz]` → conditioned `z`.
    pub fn forward(&self, z_trunk: &Tensor, token_rel_pos_feats: &Tensor) -> Tensor {
        let z = Tensor::cat(&[z_trunk, token_rel_pos_feats], -1);
        let mut z = self
            .dim_pairwise_init_proj_linear
            .forward(&self.dim_pairwise_init_proj_norm.forward(&z));

        for transition in &self.transitions {
            let t = transition.forward(&z, None);
            z = t + z;
        }
        z
    }
}

// ---------------------------------------------------------------------------
// Windowed key helpers
// ---------------------------------------------------------------------------

/// Build the static indexing matrix for windowed attention keys.
/// Equivalent to Python `get_indexing_matrix(K, W, H, device)`.
fn get_indexing_matrix(k: i64, w: i64, h: i64, device: Device) -> Tensor {
    assert!(w % 2 == 0, "W must be even");
    let half_w = w / 2;
    assert!(h % half_w == 0, "H must be divisible by W/2");
    let h_ratio = h / half_w;
    assert!(h_ratio % 2 == 0, "h ratio must be even");

    let arange = Tensor::arange(2 * k, (Kind::Int64, device));
    let diff = arange.unsqueeze(0) - arange.unsqueeze(1); // [2K, 2K]
    let index = (diff + h_ratio / 2).clamp(0, h_ratio + 1);
    // index: [2K, 2K], take every other row: index[::2] effectively = index.view(K, 2, 2K)[:, 0, :]
    let index = index.reshape(&[k, 2, 2 * k]).select(1, 0); // [K, 2K]
    let onehot = index.one_hot(h_ratio + 2); // [K, 2K, h_ratio+2]
                                             // Slice off first and last class
    let onehot = onehot.slice(2, 1, h_ratio + 1, 1); // [K, 2K, h_ratio]
    let onehot = onehot.transpose(0, 1); // [2K, K, h_ratio]
    onehot.reshape(&[2 * k, h_ratio * k]).to_kind(Kind::Float)
}

/// Map single representation from query windows to key windows.
/// `single [B, N, D]` → `[B, K, H, D]`.
fn single_to_keys(single: &Tensor, indexing_matrix: &Tensor, w: i64, h: i64) -> Tensor {
    let size = single.size();
    let (b, n, d) = (size[0], size[1], size[2]);
    let k = n / w;
    let single_r = single.reshape(&[b, 2 * k, w / 2, d]);
    // einsum "b j i d, j k -> b k i d"
    let out = Tensor::einsum("bjid,jk->bkid", &[&single_r, indexing_matrix], None::<i64>);
    out.reshape(&[b, k, h, d])
}

/// `z_to_p` einsum output must match `p` as `[B, K, W, H, atom_z]`. Some LibTorch builds return the
/// window axes swapped (`[B, K, H, W, …]`) when `W != H`, which breaks `p + z_to_p_out`.
fn align_z_to_p_with_p(p: &Tensor, z_to_p_out: Tensor) -> Tensor {
    let ps = p.size();
    let zs = z_to_p_out.size();
    if ps.len() == 5 && zs.len() == 5 && ps[0] == zs[0] && ps[4] == zs[4] {
        if ps[2] == zs[2] && ps[3] == zs[3] {
            return z_to_p_out;
        }
        if ps[2] == zs[3] && ps[3] == zs[2] {
            return z_to_p_out.transpose(2, 3);
        }
    }
    z_to_p_out
}

// ---------------------------------------------------------------------------
// AtomEncoder — flags / extra feats (encodersv2.AtomEncoder)
// ---------------------------------------------------------------------------

/// Hyperparameters controlling which atom feature tensors are concatenated before `embed_atom_features`.
///
/// Reference: `boltz-reference/.../encodersv2.py::AtomEncoder.__init__`.
#[derive(Clone, Debug)]
pub struct AtomEncoderFlags {
    /// One-hot width of `ref_element` (Boltz `num_elements`, often 128).
    pub num_elements: i64,
    pub use_no_atom_char: bool,
    pub use_atom_backbone_feat: bool,
    pub use_residue_feats_atoms: bool,
    /// When `use_atom_backbone_feat`, last dim of `atom_backbone_feat` (Boltz uses 17 classes).
    pub backbone_feat_dim: i64,
    /// Token vocabulary size (`const.num_tokens`, 33).
    pub num_tokens: i64,
}

impl Default for AtomEncoderFlags {
    fn default() -> Self {
        Self {
            // 3 + 1 + 128 = 132 with `use_no_atom_char=true` (Boltz `num_elements` / boltr-io `NUM_ELEMENTS`)
            num_elements: 128,
            use_no_atom_char: true,
            use_atom_backbone_feat: false,
            use_residue_feats_atoms: false,
            backbone_feat_dim: 17,
            num_tokens: 33,
        }
    }
}

impl AtomEncoderFlags {
    /// Expected `embed_atom_features` in-features for these flags (must match checkpoint `atom_feature_dim`).
    #[must_use]
    pub fn expected_atom_feature_dim(&self) -> i64 {
        let mut d = 3 + 1 + self.num_elements;
        if !self.use_no_atom_char {
            d += 4 * 64;
        }
        if self.use_atom_backbone_feat {
            d += self.backbone_feat_dim;
        }
        if self.use_residue_feats_atoms {
            d += self.num_tokens + 1 + 4;
        }
        d
    }
}

/// Optional per-forward tensors for extended atom encodings (name chars, backbone, residue broadcast).
pub struct AtomEncoderBatchFeats<'a> {
    pub ref_atom_name_chars: Option<&'a Tensor>,
    pub atom_backbone_feat: Option<&'a Tensor>,
    pub res_type: Option<&'a Tensor>,
    pub modified: Option<&'a Tensor>,
    pub mol_type: Option<&'a Tensor>,
}

// ---------------------------------------------------------------------------
// AtomEncoder
// ---------------------------------------------------------------------------

/// Encodes atom features into the diffusion conditioning tensors.
///
/// Reference: `encodersv2.py::AtomEncoder`
pub struct AtomEncoder {
    embed_atom_features: tch::nn::Linear,
    embed_atompair_ref_pos: tch::nn::Linear,
    embed_atompair_ref_dist: tch::nn::Linear,
    embed_atompair_mask: tch::nn::Linear,
    atoms_per_window_queries: i64,
    atoms_per_window_keys: i64,
    structure_prediction: bool,
    flags: AtomEncoderFlags,
    s_to_c_trans_norm: Option<tch::nn::LayerNorm>,
    s_to_c_trans_linear: Option<tch::nn::Linear>,
    z_to_p_trans_norm: Option<tch::nn::LayerNorm>,
    z_to_p_trans_linear: Option<tch::nn::Linear>,
    c_to_p_trans_k: tch::nn::Linear, // ReLU → LinearNoBias
    c_to_p_trans_q: tch::nn::Linear,
    p_mlp_1: tch::nn::Linear,
    p_mlp_3: tch::nn::Linear,
    p_mlp_5: tch::nn::Linear,
    atom_s: i64,
    atom_z: i64,
}

impl AtomEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        atom_s: i64,
        atom_z: i64,
        token_s: i64,
        token_z: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        atom_feature_dim: i64,
        structure_prediction: bool,
        flags: AtomEncoderFlags,
    ) -> Self {
        let expected = flags.expected_atom_feature_dim();
        assert_eq!(
            atom_feature_dim, expected,
            "atom_feature_dim {atom_feature_dim} != expected {expected} for AtomEncoderFlags {:?}",
            flags
        );
        let embed_atom_features = linear(
            path.sub("embed_atom_features"),
            atom_feature_dim,
            atom_s,
            LinearConfig::default(),
        );
        let embed_atompair_ref_pos = linear_no_bias(path.sub("embed_atompair_ref_pos"), 3, atom_z);
        let embed_atompair_ref_dist =
            linear_no_bias(path.sub("embed_atompair_ref_dist"), 1, atom_z);
        let embed_atompair_mask = linear_no_bias(path.sub("embed_atompair_mask"), 1, atom_z);

        let (s_to_c_trans_norm, s_to_c_trans_linear, z_to_p_trans_norm, z_to_p_trans_linear) =
            if structure_prediction {
                let s_n = layer_norm_1d(path.sub("s_to_c_trans").sub("0"), token_s);
                let s_l = linear_no_bias(path.sub("s_to_c_trans").sub("1"), token_s, atom_s);
                let z_n = layer_norm_1d(path.sub("z_to_p_trans").sub("0"), token_z);
                let z_l = linear_no_bias(path.sub("z_to_p_trans").sub("1"), token_z, atom_z);
                (Some(s_n), Some(s_l), Some(z_n), Some(z_l))
            } else {
                (None, None, None, None)
            };

        // c_to_p_trans_k: Sequential(ReLU, LinearNoBias(atom_s, atom_z))
        let c_to_p_trans_k = linear_no_bias(path.sub("c_to_p_trans_k").sub("1"), atom_s, atom_z);
        let c_to_p_trans_q = linear_no_bias(path.sub("c_to_p_trans_q").sub("1"), atom_s, atom_z);

        // p_mlp: Sequential(ReLU, LinearNoBias, ReLU, LinearNoBias, ReLU, LinearNoBias)
        let p_mlp_1 = linear_no_bias(path.sub("p_mlp").sub("1"), atom_z, atom_z);
        let p_mlp_3 = linear_no_bias(path.sub("p_mlp").sub("3"), atom_z, atom_z);
        let p_mlp_5 = linear_no_bias(path.sub("p_mlp").sub("5"), atom_z, atom_z);

        Self {
            embed_atom_features,
            embed_atompair_ref_pos,
            embed_atompair_ref_dist,
            embed_atompair_mask,
            atoms_per_window_queries,
            atoms_per_window_keys,
            structure_prediction,
            flags,
            s_to_c_trans_norm,
            s_to_c_trans_linear,
            z_to_p_trans_norm,
            z_to_p_trans_linear,
            c_to_p_trans_k,
            c_to_p_trans_q,
            p_mlp_1,
            p_mlp_3,
            p_mlp_5,
            atom_s,
            atom_z,
        }
    }

    pub fn flags(&self) -> &AtomEncoderFlags {
        &self.flags
    }

    fn concat_atom_feats(
        &self,
        ref_pos: &Tensor,
        ref_charge: &Tensor,
        ref_element: &Tensor,
        atom_to_token: &Tensor,
        batch: Option<&AtomEncoderBatchFeats<'_>>,
    ) -> Tensor {
        let f = &self.flags;
        debug_assert_eq!(
            ref_element.size()[2],
            f.num_elements,
            "ref_element last dim must match AtomEncoderFlags::num_elements"
        );
        let mut pieces: Vec<Tensor> = vec![
            ref_pos.shallow_clone(),
            ref_charge.unsqueeze(-1),
            ref_element.shallow_clone(),
        ];
        if !f.use_no_atom_char {
            let chars = batch
                .and_then(|b| b.ref_atom_name_chars)
                .expect("ref_atom_name_chars required when use_no_atom_char is false");
            let sz = chars.size();
            let flat = chars.reshape(&[sz[0], sz[1], 4 * 64]);
            pieces.push(flat);
        }
        if f.use_atom_backbone_feat {
            let bb = batch
                .and_then(|b| b.atom_backbone_feat)
                .expect("atom_backbone_feat required when use_atom_backbone_feat");
            pieces.push(bb.shallow_clone());
        }
        if f.use_residue_feats_atoms {
            let b = batch.expect("batch required when use_residue_feats_atoms");
            let res_type = b
                .res_type
                .expect("res_type required when use_residue_feats_atoms");
            let modified = b
                .modified
                .expect("modified required when use_residue_feats_atoms");
            let mol_type = b
                .mol_type
                .expect("mol_type required when use_residue_feats_atoms");
            let mol_oh = mol_type
                .to_kind(Kind::Int64)
                .one_hot(4)
                .to_kind(Kind::Float);
            let res_feats = Tensor::cat(
                &[
                    res_type.shallow_clone(),
                    modified.unsqueeze(-1).to_kind(Kind::Float),
                    mol_oh,
                ],
                -1,
            );
            let atom_res = atom_to_token.to_kind(Kind::Float).bmm(&res_feats);
            pieces.push(atom_res);
        }
        Tensor::cat(&pieces, -1)
    }

    /// Returns `(q, c, p, indexing_matrix)`.
    ///
    /// * `batch`: optional extra tensors matching [`AtomEncoderFlags`]; required when flags request them.
    /// * `s_trunk`: `[B, N, token_s]` (only when `structure_prediction`)
    /// * `z`: `[B, N, N, token_z]` conditioned pair repr (only when `structure_prediction`)
    pub fn forward(
        &self,
        ref_pos: &Tensor,
        ref_charge: &Tensor,
        ref_element: &Tensor,
        atom_pad_mask: &Tensor,
        ref_space_uid: &Tensor,
        atom_to_token: &Tensor,
        s_trunk: Option<&Tensor>,
        z: Option<&Tensor>,
        batch: Option<&AtomEncoderBatchFeats<'_>>,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let size = ref_pos.size();
        let (b, n) = (size[0], size[1]);
        let w = self.atoms_per_window_queries;
        let h = self.atoms_per_window_keys;
        let k = n / w;

        let atom_feats =
            self.concat_atom_feats(ref_pos, ref_charge, ref_element, atom_to_token, batch);
        let c = self.embed_atom_features.forward(&atom_feats);

        let indexing_matrix = get_indexing_matrix(k, w, h, ref_pos.device());

        // Build windowed pair features
        let atom_ref_pos_queries = ref_pos.reshape(&[b, k, w, 1, 3]);
        let atom_ref_pos_keys =
            single_to_keys(ref_pos, &indexing_matrix, w, h).reshape(&[b, k, 1, h, 3]);

        let d = &atom_ref_pos_keys - &atom_ref_pos_queries;
        let d_norm = d
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float);
        let d_norm = 1.0 / (1.0 + d_norm);

        let atom_mask = atom_pad_mask.to_kind(Kind::Bool);
        let atom_mask_queries = atom_mask.reshape(&[b, k, w, 1]);
        let atom_mask_keys = single_to_keys(
            &atom_mask.unsqueeze(-1).to_kind(Kind::Float),
            &indexing_matrix,
            w,
            h,
        )
        .reshape(&[b, k, 1, h])
        .to_kind(Kind::Bool);
        let atom_uid_queries = ref_space_uid.reshape(&[b, k, w, 1]);
        let atom_uid_keys = single_to_keys(
            &ref_space_uid.unsqueeze(-1).to_kind(Kind::Float),
            &indexing_matrix,
            w,
            h,
        )
        .reshape(&[b, k, 1, h])
        .to_kind(Kind::Int64);

        let v = (atom_mask_queries
            .logical_and(&atom_mask_keys)
            .logical_and(&atom_uid_queries.eq_tensor(&atom_uid_keys)))
        .to_kind(Kind::Float)
        .unsqueeze(-1);

        let mut p = self.embed_atompair_ref_pos.forward(&d) * &v;
        p = p + self.embed_atompair_ref_dist.forward(&d_norm) * &v;
        p = p + self.embed_atompair_mask.forward(&v) * &v;

        let q = c.shallow_clone();

        // Structure prediction conditioning from trunk
        let mut c = c;
        if self.structure_prediction {
            if let (Some(s_trunk), Some(z)) = (s_trunk, z) {
                let s_to_c = self
                    .s_to_c_trans_linear
                    .as_ref()
                    .expect("AtomEncoder: s_to_c_trans_linear missing despite structure_prediction")
                    .forward(
                        &self
                            .s_to_c_trans_norm
                            .as_ref()
                            .expect("AtomEncoder: s_to_c_trans_norm missing despite structure_prediction")
                            .forward(&s_trunk.to_kind(Kind::Float)),
                    );
                let s_to_c = atom_to_token.to_kind(Kind::Float).bmm(&s_to_c);
                c = &c + &s_to_c.to_kind(c.kind());

                let z_to_p = self
                    .z_to_p_trans_linear
                    .as_ref()
                    .expect("AtomEncoder: z_to_p_trans_linear missing despite structure_prediction")
                    .forward(
                        &self
                            .z_to_p_trans_norm
                            .as_ref()
                            .expect("AtomEncoder: z_to_p_trans_norm missing despite structure_prediction")
                            .forward(&z.to_kind(Kind::Float)),
                    );
                let atom_to_token_queries =
                    atom_to_token.to_kind(Kind::Float).reshape(&[b, k, w, -1]);
                let atom_to_token_keys =
                    single_to_keys(&atom_to_token.to_kind(Kind::Float), &indexing_matrix, w, h);
                let z_to_p_out = Tensor::einsum(
                    "bijd,bwki,bwlj->bwkld",
                    &[&z_to_p, &atom_to_token_queries, &atom_to_token_keys],
                    None::<i64>,
                );
                let z_to_p_out = align_z_to_p_with_p(&p, z_to_p_out);
                let p_sz = p.size();
                let z_sz = z_to_p_out.size();
                if p_sz != z_sz {
                    panic!(
                        "AtomEncoder: z_to_p_out shape {:?} does not match p {:?} after W/H alignment. \
                         Set boltz2_hparams atoms_per_window_queries/keys to match the checkpoint; \
                         run with RUST_BACKTRACE=1. If atom_to_token token count differs from trunk N, check preprocess collate.",
                        z_sz, p_sz
                    );
                }
                let p_kind = p.kind();
                p = p + z_to_p_out.to_kind(p_kind);
            }
        }

        // Add c contributions to p
        let c_q = c.reshape(&[b, k, w, 1, c.size()[2]]);
        let c_keys = single_to_keys(&c, &indexing_matrix, w, h).reshape(&[b, k, 1, h, c.size()[2]]);
        p = &p + &self.c_to_p_trans_q.forward(&c_q.relu());
        p = &p + &self.c_to_p_trans_k.forward(&c_keys.relu());

        // p_mlp: ReLU → Linear → ReLU → Linear → ReLU → Linear
        let p_mlp = self.p_mlp_5.forward(
            &self
                .p_mlp_3
                .forward(&self.p_mlp_1.forward(&p.relu()).relu())
                .relu(),
        );
        p = p + p_mlp;

        (q, c, p, indexing_matrix)
    }
}

// ---------------------------------------------------------------------------
// AtomAttentionEncoder
// ---------------------------------------------------------------------------

/// Atom attention encoder for the score model.
pub struct AtomAttentionEncoder {
    structure_prediction: bool,
    r_to_q_trans: Option<tch::nn::Linear>,
    atom_encoder: AtomTransformer,
    atom_to_token_trans_linear: tch::nn::Linear,
    token_s_out: i64,
}

impl AtomAttentionEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        atom_s: i64,
        token_s: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        atom_encoder_depth: i64,
        atom_encoder_heads: i64,
        structure_prediction: bool,
        device: Device,
    ) -> Self {
        let r_to_q_trans = if structure_prediction {
            Some(linear_no_bias(path.sub("r_to_q_trans"), 3, atom_s))
        } else {
            None
        };

        let atom_encoder = AtomTransformer::new(
            path.sub("atom_encoder"),
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_encoder_depth,
            atom_encoder_heads,
            atom_s,
            Some(atom_s),
            device,
        );

        let token_s_out = if structure_prediction {
            2 * token_s
        } else {
            token_s
        };
        let atom_to_token_trans_linear = linear_no_bias(
            path.sub("atom_to_token_trans").sub("0"),
            atom_s,
            token_s_out,
        );

        Self {
            structure_prediction,
            r_to_q_trans,
            atom_encoder,
            atom_to_token_trans_linear,
            token_s_out,
        }
    }

    /// Returns `(a, q_skip, c_skip)`.
    ///
    /// * `q`: `[B, M, atom_s]` from atom encoder
    /// * `c`: `[B, M, atom_s]` conditioning
    /// * `atom_enc_bias`: pre-computed atom encoder bias (windowed)
    /// * `atom_pad_mask`: `[B, M]`
    /// * `atom_to_token`: `[B, M, N]`
    /// * `r`: `[B*mult, M, 3]` noisy atom positions
    /// * `multiplicity`: sampling multiplicity
    /// * `indexing_matrix`: from AtomEncoder
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &Tensor,
        c: &Tensor,
        atom_enc_bias: &Tensor,
        atom_pad_mask: &Tensor,
        atom_to_token: &Tensor,
        r: &Tensor,
        multiplicity: i64,
        indexing_matrix: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let mut q = if self.structure_prediction {
            let q_exp = q.repeat_interleave_self_int(multiplicity, Some(0), None);
            let r_to_q = self
                .r_to_q_trans
                .as_ref()
                .expect("AtomDecoder: r_to_q_trans missing despite structure_prediction")
                .forward(r);
            q_exp + r_to_q
        } else {
            q.shallow_clone()
        };

        let c = c.repeat_interleave_self_int(multiplicity, Some(0), None);
        let mask = atom_pad_mask.repeat_interleave_self_int(multiplicity, Some(0), None);

        q = self
            .atom_encoder
            .forward(&q, &c, atom_enc_bias, &mask, multiplicity);

        let q_skip = q.shallow_clone();
        let c_skip = c.shallow_clone();

        // Aggregate atom → token
        let q_to_a = self
            .atom_to_token_trans_linear
            .forward(&q)
            .relu()
            .to_kind(Kind::Float);
        let a2t = atom_to_token
            .to_kind(Kind::Float)
            .repeat_interleave_self_int(multiplicity, Some(0), None);
        let a2t_mean = &a2t / (a2t.sum_dim_intlist(&[1i64][..], true, Kind::Float) + 1e-6);
        let a = a2t_mean.transpose(1, 2).bmm(&q_to_a).to_kind(q.kind());

        (a, q_skip, c_skip)
    }
}

// ---------------------------------------------------------------------------
// AtomAttentionDecoder  (Algorithm 6)
// ---------------------------------------------------------------------------

/// Atom attention decoder: broadcasts token activations back to atoms.
pub struct AtomAttentionDecoder {
    a_to_q_trans: tch::nn::Linear,
    atom_decoder: AtomTransformer,
    atom_feat_to_atom_pos_update_norm: tch::nn::LayerNorm,
    atom_feat_to_atom_pos_update_linear: tch::nn::Linear,
    atom_s: i64,
}

impl AtomAttentionDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        atom_s: i64,
        token_s: i64,
        attn_window_queries: i64,
        attn_window_keys: i64,
        atom_decoder_depth: i64,
        atom_decoder_heads: i64,
        device: Device,
    ) -> Self {
        let a_to_q_trans = linear_no_bias(path.sub("a_to_q_trans"), 2 * token_s, atom_s);

        let atom_decoder = AtomTransformer::new(
            path.sub("atom_decoder"),
            attn_window_queries,
            attn_window_keys,
            atom_decoder_depth,
            atom_decoder_heads,
            atom_s,
            Some(atom_s),
            device,
        );

        let atom_feat_norm =
            layer_norm_1d(path.sub("atom_feat_to_atom_pos_update").sub("0"), atom_s);
        let atom_feat_linear =
            linear_no_bias(path.sub("atom_feat_to_atom_pos_update").sub("1"), atom_s, 3);

        Self {
            a_to_q_trans,
            atom_decoder,
            atom_feat_to_atom_pos_update_norm: atom_feat_norm,
            atom_feat_to_atom_pos_update_linear: atom_feat_linear,
            atom_s,
        }
    }

    /// Returns `r_update [B*mult, M, 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        a: &Tensor,
        q: &Tensor,
        c: &Tensor,
        atom_dec_bias: &Tensor,
        atom_pad_mask: &Tensor,
        atom_to_token: &Tensor,
        multiplicity: i64,
        indexing_matrix: &Tensor,
    ) -> Tensor {
        let a2t = atom_to_token
            .to_kind(Kind::Float)
            .repeat_interleave_self_int(multiplicity, Some(0), None);
        let a_to_q = self.a_to_q_trans.forward(&a.to_kind(Kind::Float));
        let a_to_q = a2t.bmm(&a_to_q);

        let mut q = q + a_to_q.to_kind(q.kind());
        let mask = atom_pad_mask.repeat_interleave_self_int(multiplicity, Some(0), None);

        q = self
            .atom_decoder
            .forward(&q, c, atom_dec_bias, &mask, multiplicity);

        self.atom_feat_to_atom_pos_update_linear
            .forward(&self.atom_feat_to_atom_pos_update_norm.forward(&q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn align_z_to_p_with_p_fixes_swapped_window_axes() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let p = Tensor::zeros(&[1, 2, 32, 128, 16], (Kind::Float, device));
        let swapped = Tensor::zeros(&[1, 2, 128, 32, 16], (Kind::Float, device));
        let fixed = align_z_to_p_with_p(&p, swapped);
        assert_eq!(fixed.size(), p.size());
    }

    #[test]
    fn fourier_embedding_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let fe = FourierEmbedding::new(vs.root(), 256);
        let t = Tensor::randn(&[4], (Kind::Float, device));
        let out = fe.forward(&t);
        assert_eq!(out.size(), vec![4, 256]);
    }

    #[test]
    fn single_conditioning_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let ts = 32_i64;
        let sc = SingleConditioning::new(vs.root(), 16.0, ts, 64, 2, 2, device);
        let times = Tensor::randn(&[2], (Kind::Float, device));
        let s_trunk = Tensor::randn(&[2, 8, ts], (Kind::Float, device));
        let s_inputs = Tensor::randn(&[2, 8, ts], (Kind::Float, device));
        let (s, nf) = sc.forward(&times, &s_trunk, &s_inputs);
        assert_eq!(s.size(), vec![2, 8, 2 * ts]);
        assert_eq!(nf.size(), vec![2, 64]);
    }

    #[test]
    fn pairwise_conditioning_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let tz = 32_i64;
        let pc = PairwiseConditioning::new(vs.root(), tz, tz, 2, 2, device);
        let z = Tensor::randn(&[2, 8, 8, tz], (Kind::Float, device));
        let rel = Tensor::randn(&[2, 8, 8, tz], (Kind::Float, device));
        let out = pc.forward(&z, &rel);
        assert_eq!(out.size(), vec![2, 8, 8, tz]);
    }
}
