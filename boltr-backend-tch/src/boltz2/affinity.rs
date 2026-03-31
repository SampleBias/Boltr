//! Affinity head — `boltz.model.modules.affinity` (`AffinityModule`, `AffinityHeadsTransformer`).
//!
//! Python reference: `boltz-reference/src/boltz/model/modules/affinity.py`.
//! MW calibration (post-ensemble): `boltz-reference/src/boltz/model/models/boltz2.py` — see
//! [`apply_affinity_mw_correction`] and [`AFFINITY_MW_MODEL_COEF`].

use tch::nn::{embedding, linear, EmbeddingConfig, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

use super::encoders::PairwiseConditioning;
use crate::layers::PairformerNoSeqModule;
use crate::tch_compat::{layer_norm_1d, linear_no_bias};

// ---------------------------------------------------------------------------
// MW correction (Boltz2 ensemble path)
// ---------------------------------------------------------------------------

/// `model_coef` in Python `boltz2.py` (`affinity_mw_correction` block).
pub const AFFINITY_MW_MODEL_COEF: f64 = 1.03525938;
/// `mw_coef` (multiplies `affinity_mw ** 0.3`).
pub const AFFINITY_MW_COEF: f64 = -0.59992683;
/// Scalar bias added after the linear mix.
pub const AFFINITY_MW_BIAS: f64 = 2.83288489;

/// Apply the post-hoc molecular-weight calibration used when Boltz2 runs with
/// `affinity_mw_correction=True` (Python applies this after **ensemble** averaging).
///
/// `affinity_pred_value` and `affinity_mw` must be broadcast-compatible; typical shapes are
/// `[B, 1]` and `[B]` or `[B, 1]`.
#[must_use]
pub fn apply_affinity_mw_correction(affinity_pred_value: &Tensor, affinity_mw: &Tensor) -> Tensor {
    let mw_pow = affinity_mw.pow_tensor_scalar(0.3);
    let mw_pow = if mw_pow.dim() == 1 {
        mw_pow.unsqueeze(-1)
    } else {
        mw_pow
    };
    affinity_pred_value.g_mul_scalar(AFFINITY_MW_MODEL_COEF)
        + mw_pow.g_mul_scalar(AFFINITY_MW_COEF)
        + Tensor::ones_like(affinity_pred_value).g_mul_scalar(AFFINITY_MW_BIAS)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Subset of Python `AffinityModule` kwargs + `pairformer_args` / `transformer_args`.
#[derive(Debug, Clone)]
pub struct AffinityModuleConfig {
    pub num_dist_bins: i64,
    pub max_dist: f64,
    pub pairformer_num_blocks: i64,
    pub pairformer_dropout: f64,
    pub pairformer_pairwise_head_width: i64,
    pub pairformer_pairwise_num_heads: i64,
    pub pairformer_post_layer_norm: bool,
    pub pairformer_activation_checkpointing: bool,
    /// `transformer_args["token_s"]` — hidden width for affinity head MLPs (`input_token_s` in Python).
    pub head_token_s: i64,
}

impl Default for AffinityModuleConfig {
    fn default() -> Self {
        Self {
            num_dist_bins: 64,
            max_dist: 22.0,
            pairformer_num_blocks: 4,
            pairformer_dropout: 0.25,
            pairformer_pairwise_head_width: 32,
            pairformer_pairwise_num_heads: 4,
            pairformer_post_layer_norm: false,
            pairformer_activation_checkpointing: false,
            head_token_s: 384,
        }
    }
}

impl AffinityModuleConfig {
    /// Build from Lightning `affinity_model_args` JSON (see [`crate::boltz_hparams::Boltz2Hparams::affinity_model_args`]).
    /// Top-level keys: `pairformer_args`, `transformer_args`, `num_dist_bins`, `max_dist`, …
    #[must_use]
    pub fn from_affinity_model_args(v: Option<&serde_json::Value>, token_s: i64) -> Self {
        let mut cfg = Self::default();
        cfg.head_token_s = token_s;
        let Some(v) = v else {
            return cfg;
        };
        if let Some(n) = v.get("num_dist_bins").and_then(serde_json::Value::as_i64) {
            cfg.num_dist_bins = n;
        }
        if let Some(x) = v.get("max_dist").and_then(|x| x.as_f64()) {
            cfg.max_dist = x;
        }
        if let Some(p) = v.get("pairformer_args").and_then(|x| x.as_object()) {
            if let Some(n) = p.get("num_blocks").and_then(serde_json::Value::as_i64) {
                cfg.pairformer_num_blocks = n;
            }
            if let Some(d) = p.get("dropout").and_then(|x| x.as_f64()) {
                cfg.pairformer_dropout = d;
            }
            if let Some(w) = p
                .get("pairwise_head_width")
                .and_then(serde_json::Value::as_i64)
            {
                cfg.pairformer_pairwise_head_width = w;
            }
            if let Some(h) = p
                .get("pairwise_num_heads")
                .and_then(serde_json::Value::as_i64)
            {
                cfg.pairformer_pairwise_num_heads = h;
            }
            if let Some(b) = p
                .get("post_layer_norm")
                .and_then(serde_json::Value::as_bool)
            {
                cfg.pairformer_post_layer_norm = b;
            }
            if let Some(b) = p
                .get("activation_checkpointing")
                .and_then(serde_json::Value::as_bool)
            {
                cfg.pairformer_activation_checkpointing = b;
            }
        }
        if let Some(t) = v.get("transformer_args").and_then(|x| x.as_object()) {
            if let Some(ts) = t.get("token_s").and_then(serde_json::Value::as_i64) {
                cfg.head_token_s = ts;
            }
        }
        cfg
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// Dict subset from Python `AffinityModule` / `AffinityHeadsTransformer.forward`.
#[derive(Debug)]
pub struct AffinityOutput {
    pub affinity_pred_value: Tensor,
    pub affinity_logits_binary: Tensor,
}

// ---------------------------------------------------------------------------
// AffinityHeadsTransformer
// ---------------------------------------------------------------------------

struct AffinityHeads {
    /// `affinity_out_mlp`: Linear → ReLU → Linear → ReLU
    out_lin1: tch::nn::Linear,
    out_lin2: tch::nn::Linear,
    pred_value_lin1: tch::nn::Linear,
    pred_value_lin2: tch::nn::Linear,
    pred_value_lin3: tch::nn::Linear,
    pred_score_lin1: tch::nn::Linear,
    pred_score_lin2: tch::nn::Linear,
    pred_score_lin3: tch::nn::Linear,
    logits_binary: tch::nn::Linear,
    token_z: i64,
    head_s: i64,
}

impl AffinityHeads {
    fn new<'a>(path: Path<'a>, token_z: i64, head_s: i64) -> Self {
        let p = path.sub("affinity_out_mlp");
        let out_lin1 = linear(p.sub("0"), token_z, token_z, LinearConfig::default());
        let out_lin2 = linear(p.sub("2"), token_z, head_s, LinearConfig::default());

        let pv = path.sub("to_affinity_pred_value");
        let pred_value_lin1 = linear(pv.sub("0"), head_s, head_s, LinearConfig::default());
        let pred_value_lin2 = linear(pv.sub("2"), head_s, head_s, LinearConfig::default());
        let pred_value_lin3 = linear(pv.sub("4"), head_s, 1, LinearConfig::default());

        let ps = path.sub("to_affinity_pred_score");
        let pred_score_lin1 = linear(ps.sub("0"), head_s, head_s, LinearConfig::default());
        let pred_score_lin2 = linear(ps.sub("2"), head_s, head_s, LinearConfig::default());
        let pred_score_lin3 = linear(ps.sub("4"), head_s, 1, LinearConfig::default());

        let logits_binary = linear(
            path.sub("to_affinity_logits_binary"),
            1,
            1,
            LinearConfig::default(),
        );

        Self {
            out_lin1,
            out_lin2,
            pred_value_lin1,
            pred_value_lin2,
            pred_value_lin3,
            pred_score_lin1,
            pred_score_lin2,
            pred_score_lin3,
            logits_binary,
            token_z,
            head_s,
        }
    }

    fn forward(
        &self,
        z: &Tensor,
        token_pad_mask: &Tensor,
        mol_type: &Tensor,
        affinity_token_mask: &Tensor,
        multiplicity: i64,
    ) -> AffinityOutput {
        let mut z = z.shallow_clone();
        let pad_token_mask = token_pad_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .unsqueeze(-1)
            .to_kind(Kind::Float);
        let mol_b = mol_type.repeat_interleave_self_int(multiplicity, Some(0), None);
        // `[B,N] * [B,N,1]` would broadcast to `[B,N,N]`; match Python `[B,N,1]` masks.
        let rec_mask = mol_b
            .eq_tensor(&Tensor::zeros_like(&mol_b))
            .to_kind(Kind::Float)
            .unsqueeze(-1)
            * &pad_token_mask;
        let lig_mask = affinity_token_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .to_kind(Kind::Float)
            .unsqueeze(-1)
            * &pad_token_mask;

        let mut cross_pair_mask = lig_mask.unsqueeze(2) * rec_mask.unsqueeze(1)
            + rec_mask.unsqueeze(2) * lig_mask.unsqueeze(1)
            + lig_mask.unsqueeze(2) * lig_mask.unsqueeze(1);

        let n = lig_mask.size()[1];
        let device = lig_mask.device();
        let eye = Tensor::eye(n, (Kind::Float, device)).unsqueeze(0);
        // Outer products are `[B, N, N, 1]`. Squeeze before `(1 - eye)`: `[B,N,N,1] * [1,N,N]`
        // would broadcast to `[B,N,N,N]` in libtorch.
        cross_pair_mask = cross_pair_mask.squeeze_dim(-1);
        cross_pair_mask = cross_pair_mask * (Tensor::ones_like(&eye) - eye);

        // `PairformerNoSeq` can mis-shape batch as an extra leading `N` (symmetric `[N,N,N,C]`);
        // collapse to `[1,N,N,C]` so pooling matches Python `[B,N,N,C]`.
        if multiplicity == 1 && z.dim() == 4 {
            let b0 = z.size()[0];
            let n1 = z.size()[1];
            let n2 = z.size()[2];
            if b0 > 1 && b0 == n1 && n1 == n2 {
                z = z.narrow(0, 0, 1);
            }
        }

        // Pool over (i,j) pairs — match Python `sum(dim=(1,2))` on `[B, N, N, C]`.
        let numer = (z * cross_pair_mask.unsqueeze(-1))
            .flatten(1, 2)
            .sum_dim_intlist(&[1i64][..], false, Kind::Float);
        let denom = cross_pair_mask
            .flatten(1, 2)
            .sum_dim_intlist(&[1i64][..], false, Kind::Float)
            + 1e-7;
        let g = numer / denom.unsqueeze(-1);

        let g = self
            .out_lin2
            .forward(&self.out_lin1.forward(&g).relu())
            .relu();

        let affinity_pred_value = self
            .pred_value_lin3
            .forward(
                &self
                    .pred_value_lin2
                    .forward(&self.pred_value_lin1.forward(&g).relu())
                    .relu(),
            )
            .reshape(&[-1, 1]);
        let affinity_pred_score = self
            .pred_score_lin3
            .forward(
                &self
                    .pred_score_lin2
                    .forward(&self.pred_score_lin1.forward(&g).relu())
                    .relu(),
            )
            .reshape(&[-1, 1]);
        let affinity_logits_binary = self
            .logits_binary
            .forward(&affinity_pred_score)
            .reshape(&[-1, 1]);

        AffinityOutput {
            affinity_pred_value,
            affinity_logits_binary,
        }
    }
}

// ---------------------------------------------------------------------------
// AffinityModule
// ---------------------------------------------------------------------------

/// Boltz2 affinity stack (`boltz.model.modules.affinity.AffinityModule`).
pub struct AffinityModule {
    device: Device,
    boundaries: Tensor,
    dist_bin_pairwise_embed: tch::nn::Embedding,
    s_to_z_prod_in1: tch::nn::Linear,
    s_to_z_prod_in2: tch::nn::Linear,
    z_norm: tch::nn::LayerNorm,
    z_linear: tch::nn::Linear,
    pairwise_conditioner: PairwiseConditioning,
    pairformer_stack: PairformerNoSeqModule,
    affinity_heads: AffinityHeads,
    token_z: i64,
}

impl AffinityModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        path: Path<'a>,
        device: Device,
        token_s: i64,
        token_z: i64,
        cfg: &AffinityModuleConfig,
    ) -> Self {
        let num_dist = cfg.num_dist_bins;
        let boundaries = Tensor::linspace(2.0, cfg.max_dist, num_dist - 1, (Kind::Float, device));

        let dist_bin_pairwise_embed = embedding(
            path.sub("dist_bin_pairwise_embed"),
            num_dist,
            token_z,
            EmbeddingConfig::default(),
        );

        let s_to_z_prod_in1 = linear_no_bias(path.sub("s_to_z_prod_in1"), token_s, token_z);
        let s_to_z_prod_in2 = linear_no_bias(path.sub("s_to_z_prod_in2"), token_s, token_z);

        let z_norm = layer_norm_1d(path.sub("z_norm"), token_z);
        let z_linear = linear_no_bias(path.sub("z_linear"), token_z, token_z);

        let pairwise_conditioner = PairwiseConditioning::new(
            path.sub("pairwise_conditioner"),
            token_z,
            token_z,
            2,
            2,
            device,
        );

        let pairformer_stack = PairformerNoSeqModule::new(
            path.sub("pairformer_stack"),
            token_z,
            cfg.pairformer_num_blocks,
            Some(cfg.pairformer_dropout),
            Some(cfg.pairformer_pairwise_head_width),
            Some(cfg.pairformer_pairwise_num_heads),
            Some(cfg.pairformer_post_layer_norm),
            Some(cfg.pairformer_activation_checkpointing),
            device,
        );

        let affinity_heads =
            AffinityHeads::new(path.sub("affinity_heads"), token_z, cfg.head_token_s);

        Self {
            device,
            boundaries,
            dist_bin_pairwise_embed,
            s_to_z_prod_in1,
            s_to_z_prod_in2,
            z_norm,
            z_linear,
            pairwise_conditioner,
            pairformer_stack,
            affinity_heads,
            token_z,
        }
    }

    /// Algorithm 31 — `AffinityModule.forward` (`affinity.py`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: &Tensor,
        z: &Tensor,
        x_pred: &Tensor,
        token_pad_mask: &Tensor,
        mol_type: &Tensor,
        affinity_token_mask: &Tensor,
        token_to_rep_atom: &Tensor,
        multiplicity: i64,
        use_kernels: bool,
    ) -> AffinityOutput {
        let mut z = self.z_linear.forward(&self.z_norm.forward(z));
        z = z.repeat_interleave_self_int(multiplicity, Some(0), None);

        z = z
            + self.s_to_z_prod_in1.forward(s_inputs).unsqueeze(2)
            + self.s_to_z_prod_in2.forward(s_inputs).unsqueeze(1);

        let token_to_rep_atom =
            token_to_rep_atom.repeat_interleave_self_int(multiplicity, Some(0), None);

        let x_pred = if x_pred.dim() == 4 {
            let b = x_pred.size()[0];
            let m = x_pred.size()[1];
            let n = x_pred.size()[2];
            x_pred.view([b * m, n, -1])
        } else {
            x_pred.shallow_clone()
        };

        let x_pred_repr = token_to_rep_atom.bmm(&x_pred.to_kind(Kind::Float));
        let d = Tensor::cdist(&x_pred_repr, &x_pred_repr, 2.0, None::<i64>);
        let distogram = d
            .unsqueeze(-1)
            .gt_tensor(&self.boundaries)
            .to_kind(Kind::Float)
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .to_kind(Kind::Int64);
        let distogram = self.dist_bin_pairwise_embed.forward(&distogram);
        let z_pc = z.shallow_clone();
        z = z + self.pairwise_conditioner.forward(&z_pc, &distogram);

        let pad_token_mask = token_pad_mask.repeat_interleave_self_int(multiplicity, Some(0), None);
        let mol_b = mol_type.repeat_interleave_self_int(multiplicity, Some(0), None);
        let rec_mask = mol_b
            .eq_tensor(&Tensor::zeros_like(&mol_b))
            .to_kind(Kind::Float)
            * &pad_token_mask.to_kind(Kind::Float);
        let lig_mask = affinity_token_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .to_kind(Kind::Float)
            * &pad_token_mask.to_kind(Kind::Float);

        let cross_pair_mask = lig_mask.unsqueeze(2) * rec_mask.unsqueeze(1)
            + rec_mask.unsqueeze(2) * lig_mask.unsqueeze(1)
            + lig_mask.unsqueeze(2) * lig_mask.unsqueeze(1);
        // `TriangleMultiplication*` / attention expect `[B, N, N]` and apply `mask.unsqueeze(-1)`;
        // lig/rec outer products are `[B, N, N, 1]` — squeeze so we do not build a 5D mask.
        let pair_mask = cross_pair_mask.squeeze_dim(-1);

        let z = self.pairformer_stack.forward(&z, &pair_mask, use_kernels);

        self.affinity_heads.forward(
            &z,
            token_pad_mask,
            mol_type,
            affinity_token_mask,
            multiplicity,
        )
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }
}

/// Back-compat name used in early Boltr stubs.
pub type AffinityHead = AffinityModule;

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn mw_correction_matches_formula() {
        tch::manual_seed(0);
        let device = Device::Cpu;
        let pred = Tensor::from_slice(&[0.5, -1.0])
            .view([2, 1])
            .to_device(device);
        let mw = Tensor::from_slice(&[400.0, 100.0])
            .view([2])
            .to_device(device);
        let out = apply_affinity_mw_correction(&pred, &mw);
        let v0 = 0.5_f64;
        let mw0 = 400_f64.powf(0.3);
        let e0 = AFFINITY_MW_MODEL_COEF * v0 + AFFINITY_MW_COEF * mw0 + AFFINITY_MW_BIAS;
        let v1 = -1.0_f64;
        let mw1 = 100_f64.powf(0.3);
        let e1 = AFFINITY_MW_MODEL_COEF * v1 + AFFINITY_MW_COEF * mw1 + AFFINITY_MW_BIAS;
        let g0 = out.double_value(&[0, 0]);
        let g1 = out.double_value(&[1, 0]);
        assert!((g0 - e0).abs() < 1e-5, "got {} expected {}", g0, e0);
        assert!((g1 - e1).abs() < 1e-5, "got {} expected {}", g1, e1);
    }

    #[test]
    fn affinity_module_forward_smoke() {
        tch::manual_seed(42);
        let device = Device::Cpu;
        let token_s = 32_i64;
        let token_z = 16_i64;
        let n = 8_i64;
        let a = 8_i64;
        let cfg = AffinityModuleConfig {
            num_dist_bins: 32,
            max_dist: 22.0,
            pairformer_num_blocks: 1,
            pairformer_dropout: 0.0,
            pairformer_pairwise_head_width: 8,
            pairformer_pairwise_num_heads: 2,
            pairformer_post_layer_norm: false,
            pairformer_activation_checkpointing: false,
            head_token_s: token_s,
        };

        let vs = VarStore::new(device);
        let root = vs.root();
        let m = AffinityModule::new(root.sub("affinity_module"), device, token_s, token_z, &cfg);

        let b = 1_i64;
        let s_inputs = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let x_pred = Tensor::randn(&[b, a, 3], (Kind::Float, device));
        let token_pad_mask = Tensor::ones(&[b, n], (Kind::Float, device));
        // First 6 tokens protein (mol_type 0); last 2 ligand (non-zero mol_type).
        let mut mol_tail = Tensor::zeros(&[b, 2], (Kind::Int64, device));
        let _ = mol_tail.fill_(2);
        let mol_type = Tensor::cat(
            &[Tensor::zeros(&[b, 6], (Kind::Int64, device)), mol_tail],
            1,
        );
        let affinity_token_mask = Tensor::cat(
            &[
                Tensor::zeros(&[b, 6], (Kind::Float, device)),
                Tensor::ones(&[b, 2], (Kind::Float, device)),
            ],
            1,
        );
        let token_to_rep_atom = Tensor::eye(a, (Kind::Float, device)).unsqueeze(0);

        let out = m.forward(
            &s_inputs,
            &z,
            &x_pred,
            &token_pad_mask,
            &mol_type,
            &affinity_token_mask,
            &token_to_rep_atom,
            1,
            false,
        );
        assert_eq!(out.affinity_pred_value.size(), [b, 1]);
        assert_eq!(out.affinity_logits_binary.size(), [b, 1]);
        assert!(out.affinity_pred_value.double_value(&[0, 0]).is_finite());
        assert!(out.affinity_logits_binary.double_value(&[0, 0]).is_finite());
    }

    #[test]
    fn affinity_config_from_json() {
        let j = serde_json::json!({
            "pairformer_args": { "num_blocks": 2, "dropout": 0.1 },
            "transformer_args": { "token_s": 128 },
            "num_dist_bins": 48,
        });
        let c = AffinityModuleConfig::from_affinity_model_args(Some(&j), 384);
        assert_eq!(c.pairformer_num_blocks, 2);
        assert!((c.pairformer_dropout - 0.1).abs() < 1e-9);
        assert_eq!(c.head_token_s, 128);
        assert_eq!(c.num_dist_bins, 48);
    }
}
