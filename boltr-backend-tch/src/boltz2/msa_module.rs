//! Boltz2 `MSAModule` (`modules/trunkv2.py`).
//!
//! VarStore root: `msa_module` (sibling of `pairformer_module` on the Lightning model).

use crate::layers::{OuterProductMeanMsa, PairWeightedAveraging, PairformerNoSeqLayer, Transition};

use super::input_embedder::BOLTZ_NUM_TOKENS;
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

/// Collated MSA-related tensors (Boltz `featurizerv2` / collate contract).
#[derive(Debug, Clone, Copy)]
pub struct MsaFeatures<'a> {
    /// `[B, S, N]` int64 residue indices (0..num_tokens).
    pub msa: &'a Tensor,
    /// `[B, S, N]` mask (int or float; treated as multiplicative mask).
    pub msa_mask: &'a Tensor,
    /// `[B, S, N]` int64 (0/1), broadcast like Python `unsqueeze(-1)`.
    pub has_deletion: &'a Tensor,
    /// `[B, S, N]` float.
    pub deletion_value: &'a Tensor,
    /// `[B, S, N]` int64 (0/1) when `use_paired_feature`.
    pub msa_paired: &'a Tensor,
    /// `[B, N]` float token pad mask.
    pub token_pad_mask: &'a Tensor,
}

struct MsaLayerBlock {
    pair_weighted_averaging: PairWeightedAveraging,
    msa_transition: Transition,
    outer_product_mean: OuterProductMeanMsa,
    pairformer_layer: PairformerNoSeqLayer,
    msa_dropout: f64,
}

impl MsaLayerBlock {
    fn new<'a>(
        path: Path<'a>,
        msa_s: i64,
        token_z: i64,
        msa_dropout: f64,
        z_dropout: f64,
        pairwise_head_width: i64,
        pairwise_num_heads: i64,
        device: Device,
    ) -> Self {
        let pair_weighted_averaging = PairWeightedAveraging::new(
            path.sub("pair_weighted_averaging"),
            msa_s,
            token_z,
            32,
            8,
            None,
            device,
        );
        let msa_transition = Transition::new(
            path.sub("msa_transition"),
            msa_s,
            Some(msa_s * 4),
            None,
            device,
        );
        let outer_product_mean =
            OuterProductMeanMsa::new(path.sub("outer_product_mean"), msa_s, 32, token_z, device);
        let pairformer_layer = PairformerNoSeqLayer::new(
            path.sub("pairformer_layer"),
            token_z,
            Some(z_dropout),
            Some(pairwise_head_width),
            Some(pairwise_num_heads),
            Some(false),
            device,
        );
        Self {
            pair_weighted_averaging,
            msa_transition,
            outer_product_mean,
            pairformer_layer,
            msa_dropout,
        }
    }

    fn forward(
        &self,
        z: &Tensor,
        m: &Tensor,
        pair_mask: &Tensor,
        msa_mask_float: &Tensor,
        training: bool,
        chunk_size_tri_attn: Option<i64>,
        use_kernels: bool,
    ) -> (Tensor, Tensor) {
        let msa_drop = msa_dropout_mask(self.msa_dropout, m, training);
        let pwa = self.pair_weighted_averaging.forward(m, z, pair_mask);
        let m = {
            let m1 = m + msa_drop * pwa;
            let m2 = self.msa_transition.forward(&m1, None);
            m1 + m2
        };

        let mut z = z + self.outer_product_mean.forward(&m, msa_mask_float);
        z = self.pairformer_layer.forward(
            &z,
            pair_mask,
            chunk_size_tri_attn,
            training,
            use_kernels,
        );
        (z, m)
    }
}

/// Matches `get_dropout_mask(..., columnwise=false)` for a 4D MSA tensor `[B, S, N, D]`.
fn msa_dropout_mask(dropout: f64, m: &Tensor, training: bool) -> Tensor {
    let dropout = if training { dropout } else { 0.0 };
    let v = m.narrow(2, 0, 1).narrow(3, 0, 1);
    let thr = Tensor::from(dropout).to_device(v.device());
    let d = v.rand_like().ge_tensor(&thr);
    d.to_kind(Kind::Float) * (1.0 / (1.0 - dropout).max(1e-12))
}

pub struct MsaModule {
    token_s: i64,
    token_z: i64,
    msa_s: i64,
    use_paired_feature: bool,
    s_proj: tch::nn::Linear,
    msa_proj: tch::nn::Linear,
    layers: Vec<MsaLayerBlock>,
}

impl MsaModule {
    /// `path`: `VarStore` subpath `msa_module` (e.g. `root.sub("msa_module")`).
    pub fn new<'a>(
        path: Path<'a>,
        token_s: i64,
        token_z: i64,
        msa_s: Option<i64>,
        msa_blocks: Option<i64>,
        msa_dropout: Option<f64>,
        z_dropout: Option<f64>,
        use_paired_feature: Option<bool>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        device: Device,
    ) -> Self {
        let msa_s = msa_s.unwrap_or(64);
        let msa_blocks = msa_blocks.unwrap_or(4);
        let msa_dropout = msa_dropout.unwrap_or(0.0);
        let z_dropout = z_dropout.unwrap_or(0.0);
        let use_paired_feature = use_paired_feature.unwrap_or(true);
        let pairwise_head_width = pairwise_head_width.unwrap_or(32);
        let pairwise_num_heads = pairwise_num_heads.unwrap_or(4);

        let msa_in = BOLTZ_NUM_TOKENS + 2 + if use_paired_feature { 1 } else { 0 };

        let s_proj = linear(
            path.sub("s_proj"),
            token_s,
            msa_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let msa_proj = linear(
            path.sub("msa_proj"),
            msa_in,
            msa_s,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let layers_root = path.sub("layers");
        let mut layers = Vec::new();
        for i in 0..msa_blocks {
            layers.push(MsaLayerBlock::new(
                layers_root.sub(&i.to_string()),
                msa_s,
                token_z,
                msa_dropout,
                z_dropout,
                pairwise_head_width,
                pairwise_num_heads,
                device,
            ));
        }

        Self {
            token_s,
            token_z,
            msa_s,
            use_paired_feature,
            s_proj,
            msa_proj,
            layers,
        }
    }

    /// When `feats` is `None`, returns `z` unchanged (stub behavior for callers without MSA).
    pub fn forward_trunk_step(
        &self,
        z: &Tensor,
        s: &Tensor,
        feats: Option<&MsaFeatures<'_>>,
        training: bool,
        chunk_size_tri_attn: Option<i64>,
        use_kernels: bool,
    ) -> Tensor {
        let Some(feats) = feats else {
            return z.shallow_clone();
        };

        let msa_oh = feats.msa.one_hot(BOLTZ_NUM_TOKENS);
        let msa_oh = msa_oh.to_kind(Kind::Float);

        let hd = feats.has_deletion.unsqueeze(-1).to_kind(Kind::Float);
        let dv = feats.deletion_value.unsqueeze(-1);
        let mut pieces = vec![msa_oh, hd, dv];
        if self.use_paired_feature {
            pieces.push(feats.msa_paired.unsqueeze(-1).to_kind(Kind::Float));
        }
        let m_cat = Tensor::cat(&pieces, -1);
        let m_lin = self.msa_proj.forward(&m_cat);
        let s_lin = self.s_proj.forward(s).unsqueeze(1);
        let mut m = m_lin + s_lin;

        let msa_mask_f = feats.msa_mask.to_kind(Kind::Float);
        let tm = feats.token_pad_mask;
        let pair_mask = tm.unsqueeze(2) * tm.unsqueeze(1);

        let mut z = z.shallow_clone();
        for layer in &self.layers {
            let (zn, mn) = layer.forward(
                &z,
                &m,
                &pair_mask,
                &msa_mask_f,
                training,
                chunk_size_tri_attn,
                use_kernels,
            );
            z = zn;
            m = mn;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msa_module_forward_shapes() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let b = 1_i64;
        let s = 4_i64;
        let n = 6_i64;
        let token_s = 32_i64;
        let token_z = 24_i64;

        let vs = tch::nn::VarStore::new(device);
        let m = MsaModule::new(
            vs.root().sub("msa_module"),
            token_s,
            token_z,
            Some(16),
            Some(2),
            Some(0.0),
            Some(0.0),
            Some(true),
            None,
            None,
            device,
        );

        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let emb = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let msa = Tensor::zeros(&[b, s, n], (Kind::Int64, device));
        let msa_mask = Tensor::ones(&[b, s, n], (Kind::Float, device));
        let has_deletion = Tensor::zeros(&[b, s, n], (Kind::Int64, device));
        let deletion_value = Tensor::zeros(&[b, s, n], (Kind::Float, device));
        let msa_paired = Tensor::zeros(&[b, s, n], (Kind::Int64, device));
        let token_pad_mask = Tensor::ones(&[b, n], (Kind::Float, device));

        let feats = MsaFeatures {
            msa: &msa,
            msa_mask: &msa_mask,
            has_deletion: &has_deletion,
            deletion_value: &deletion_value,
            msa_paired: &msa_paired,
            token_pad_mask: &token_pad_mask,
        };

        let out = m.forward_trunk_step(&z, &emb, Some(&feats), false, None, false);
        assert_eq!(out.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn msa_module_none_feats_identity() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = tch::nn::VarStore::new(device);
        let m = MsaModule::new(
            vs.root().sub("msa_module"),
            32,
            16,
            None,
            Some(1),
            None,
            None,
            None,
            None,
            None,
            device,
        );
        let z = Tensor::ones(&[1, 3, 3, 16], (Kind::Float, device));
        let s = Tensor::zeros(&[1, 3, 32], (Kind::Float, device));
        let out = m.forward_trunk_step(&z, &s, None, false, None, false);
        let diff = (out - z).abs().max();
        assert!(diff.double_value(&[]) < 1e-6);
    }
}
