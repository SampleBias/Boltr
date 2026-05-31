//! Confidence head — `boltz.model.modules.confidencev2` (`ConfidenceModule` + `ConfidenceHeads`).
//!
//! Python reference: `boltz-reference/src/boltz/model/modules/confidencev2.py`.
//! Default flags match Boltz2: `add_z_input_to_z=false`, `add_s_to_z_prod=false`, `add_s_input_to_s=false`.

use tch::nn::{embedding, EmbeddingConfig, Module, Path};
use tch::{Device, Kind, Tensor};

use super::confidence_utils::{compute_aggregated_metric, compute_ptms, CHAIN_TYPE_NONPOLYMER};
use crate::layers::PairformerModule;
use crate::tch_compat::{layer_norm_1d, linear_no_bias};

/// Configuration for [`ConfidenceModule`] (subset of Python kwargs + pairformer sizing).
#[derive(Debug, Clone)]
pub struct ConfidenceModuleConfig {
    pub num_dist_bins: i64,
    pub max_dist: f64,
    pub pairformer_num_blocks: i64,
    pub pairformer_num_heads: Option<i64>,
    /// `pairformer_args["no_update_s"]` — skip `LayerNorm` on `s` before the stack.
    pub no_update_s: bool,
    pub token_level_confidence: bool,
    pub num_plddt_bins: i64,
    pub num_pde_bins: i64,
    pub num_pae_bins: i64,
    pub use_separate_heads: bool,
}

impl Default for ConfidenceModuleConfig {
    fn default() -> Self {
        Self {
            num_dist_bins: 64,
            max_dist: 22.0,
            pairformer_num_blocks: 4,
            pairformer_num_heads: Some(16),
            no_update_s: false,
            token_level_confidence: true,
            num_plddt_bins: 50,
            num_pde_bins: 64,
            num_pae_bins: 64,
            use_separate_heads: false,
        }
    }
}

impl ConfidenceModuleConfig {
    /// Build from Lightning `confidence_model_args` JSON (see [`crate::boltz_hparams::Boltz2Hparams::confidence_model_args`]).
    #[must_use]
    pub fn from_confidence_model_args(
        v: Option<&serde_json::Value>,
        token_level_confidence: bool,
    ) -> Self {
        let mut cfg = Self::default();
        cfg.token_level_confidence = token_level_confidence;
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
            if let Some(n) = p.get("num_heads").and_then(serde_json::Value::as_i64) {
                cfg.pairformer_num_heads = Some(n);
            }
            if let Some(b) = p.get("no_update_s").and_then(serde_json::Value::as_bool) {
                cfg.no_update_s = b;
            }
        }
        if let Some(c) = v.get("confidence_args").and_then(|x| x.as_object()) {
            if let Some(n) = c.get("num_plddt_bins").and_then(serde_json::Value::as_i64) {
                cfg.num_plddt_bins = n;
            }
            if let Some(n) = c.get("num_pde_bins").and_then(serde_json::Value::as_i64) {
                cfg.num_pde_bins = n;
            }
            if let Some(n) = c.get("num_pae_bins").and_then(serde_json::Value::as_i64) {
                cfg.num_pae_bins = n;
            }
            if let Some(b) = c.get("use_separate_heads").and_then(serde_json::Value::as_bool) {
                cfg.use_separate_heads = b;
            }
        }
        cfg
    }
}

/// Full confidence stack + heads (`confidence_module` in Lightning).
pub struct ConfidenceModule {
    device: Device,
    boundaries: Tensor,
    dist_bin_pairwise_embed: tch::nn::Embedding,
    s_to_z: tch::nn::Linear,
    s_to_z_transpose: tch::nn::Linear,
    s_inputs_norm: tch::nn::LayerNorm,
    s_norm: Option<tch::nn::LayerNorm>,
    z_norm: tch::nn::LayerNorm,
    pairformer_stack: PairformerModule,
    heads: ConfidenceHeads,
    no_update_s: bool,
    token_level_confidence: bool,
}

impl ConfidenceModule {
    /// Build under `path` (e.g. `vs.root().sub("confidence_module")`) so checkpoints load as in Python.
    pub fn new<'a>(
        path: Path<'a>,
        device: Device,
        token_s: i64,
        token_z: i64,
        cfg: &ConfidenceModuleConfig,
    ) -> Self {
        let num_dist = cfg.num_dist_bins;
        let boundaries = Tensor::linspace(2.0, cfg.max_dist, num_dist - 1, (Kind::Float, device));

        let dist_bin_pairwise_embed = embedding(
            path.sub("dist_bin_pairwise_embed"),
            num_dist,
            token_z,
            EmbeddingConfig::default(),
        );

        let s_to_z = linear_no_bias(path.sub("s_to_z"), token_s, token_z);
        let s_to_z_transpose = linear_no_bias(path.sub("s_to_z_transpose"), token_s, token_z);

        let s_inputs_norm = layer_norm_1d(path.sub("s_inputs_norm"), token_s);
        let s_norm = if cfg.no_update_s {
            None
        } else {
            Some(layer_norm_1d(path.sub("s_norm"), token_s))
        };
        let z_norm = layer_norm_1d(path.sub("z_norm"), token_z);

        let pairformer_stack = PairformerModule::new(
            path.sub("pairformer_stack"),
            token_s,
            token_z,
            cfg.pairformer_num_blocks,
            cfg.pairformer_num_heads,
            Some(0.25),
            None,
            None,
            Some(false),
            Some(false),
            Some(true),
            device,
        );

        let heads = ConfidenceHeads::new(path.sub("confidence_heads"), token_s, token_z, cfg);

        Self {
            device,
            boundaries,
            dist_bin_pairwise_embed,
            s_to_z,
            s_to_z_transpose,
            s_inputs_norm,
            s_norm,
            z_norm,
            pairformer_stack,
            heads,
            no_update_s: cfg.no_update_s,
            token_level_confidence: cfg.token_level_confidence,
        }
    }

    /// Algorithm 31 — `ConfidenceModule.forward` (single multiplicity or already expanded).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: &Tensor,
        s: &Tensor,
        z: &Tensor,
        x_pred: &Tensor,
        token_pad_mask: &Tensor,
        asym_id: &Tensor,
        mol_type: &Tensor,
        token_to_rep_atom: &Tensor,
        frames_idx: &Tensor,
        pred_distogram_logits: &Tensor,
        multiplicity: i64,
    ) -> ConfidenceOutput {
        let s_inputs = self.s_inputs_norm.forward(s_inputs);
        let mut s = s.shallow_clone();
        if let Some(sn) = &self.s_norm {
            s = sn.forward(&s);
        }
        let z = self.z_norm.forward(z);

        let z = z
            + self.s_to_z.forward(&s_inputs).unsqueeze(2)
            + self.s_to_z_transpose.forward(&s_inputs).unsqueeze(1);

        let s = s.repeat_interleave_self_int(multiplicity, Some(0), None);
        let mut z = z.repeat_interleave_self_int(multiplicity, Some(0), None);
        let s_inputs = s_inputs.repeat_interleave_self_int(multiplicity, Some(0), None);

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
        let z = z + distogram;

        let mask = token_pad_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .to_kind(Kind::Float);
        let pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1);

        let (s_t, z_t) = self
            .pairformer_stack
            .forward(&s, &z, &mask, &pair_mask, false);

        let s = s_t;
        let z = z_t;

        self.heads.forward(
            &s,
            &z,
            &x_pred,
            &d,
            token_pad_mask,
            asym_id,
            mol_type,
            pred_distogram_logits,
            multiplicity,
            &frames_idx,
        )
    }
}

/// Head outputs (logits + derived scalars). Mirrors Python dict keys used in `boltz2.py`.
pub struct ConfidenceOutput {
    pub pae_logits: Tensor,
    pub pde_logits: Tensor,
    pub plddt_logits: Tensor,
    pub resolved_logits: Tensor,
    pub pae: Tensor,
    pub pde: Tensor,
    pub plddt: Tensor,
    pub complex_plddt: Tensor,
    pub complex_iplddt: Tensor,
    pub complex_pde: Tensor,
    pub complex_ipde: Tensor,
    pub ptm: Tensor,
    pub iptm: Tensor,
    pub ligand_iptm: Tensor,
    pub protein_iptm: Tensor,
    pub pair_chains_iptm: std::collections::BTreeMap<i64, std::collections::BTreeMap<i64, Tensor>>,
}

struct ConfidenceHeads {
    to_pae_logits: Option<tch::nn::Linear>,
    to_pae_intra_logits: Option<tch::nn::Linear>,
    to_pae_inter_logits: Option<tch::nn::Linear>,
    to_pde_logits: Option<tch::nn::Linear>,
    to_pde_intra_logits: Option<tch::nn::Linear>,
    to_pde_inter_logits: Option<tch::nn::Linear>,
    to_plddt_logits: tch::nn::Linear,
    to_resolved_logits: tch::nn::Linear,
    use_separate_heads: bool,
    token_level_confidence: bool,
}

impl ConfidenceHeads {
    fn new<'a>(path: Path<'a>, token_s: i64, token_z: i64, cfg: &ConfidenceModuleConfig) -> Self {
        let (
            to_pae_logits,
            to_pae_intra_logits,
            to_pae_inter_logits,
            to_pde_logits,
            to_pde_intra_logits,
            to_pde_inter_logits,
        ) = if cfg.use_separate_heads {
            (
                None,
                Some(linear_no_bias(
                    path.sub("to_pae_intra_logits"),
                    token_z,
                    cfg.num_pae_bins,
                )),
                Some(linear_no_bias(
                    path.sub("to_pae_inter_logits"),
                    token_z,
                    cfg.num_pae_bins,
                )),
                None,
                Some(linear_no_bias(
                    path.sub("to_pde_intra_logits"),
                    token_z,
                    cfg.num_pde_bins,
                )),
                Some(linear_no_bias(
                    path.sub("to_pde_inter_logits"),
                    token_z,
                    cfg.num_pde_bins,
                )),
            )
        } else {
            (
                Some(linear_no_bias(
                    path.sub("to_pae_logits"),
                    token_z,
                    cfg.num_pae_bins,
                )),
                None,
                None,
                Some(linear_no_bias(
                    path.sub("to_pde_logits"),
                    token_z,
                    cfg.num_pde_bins,
                )),
                None,
                None,
            )
        };
        let to_plddt_logits =
            linear_no_bias(path.sub("to_plddt_logits"), token_s, cfg.num_plddt_bins);
        let to_resolved_logits = linear_no_bias(path.sub("to_resolved_logits"), token_s, 2);

        Self {
            to_pae_logits,
            to_pae_intra_logits,
            to_pae_inter_logits,
            to_pde_logits,
            to_pde_intra_logits,
            to_pde_inter_logits,
            to_plddt_logits,
            to_resolved_logits,
            use_separate_heads: cfg.use_separate_heads,
            token_level_confidence: cfg.token_level_confidence,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        s: &Tensor,
        z: &Tensor,
        x_pred: &Tensor,
        d: &Tensor,
        token_pad_mask: &Tensor,
        asym_id: &Tensor,
        mol_type: &Tensor,
        pred_distogram_logits: &Tensor,
        multiplicity: i64,
        frames_idx: &Tensor,
    ) -> ConfidenceOutput {
        assert!(
            self.token_level_confidence,
            "atom-level confidence path not ported"
        );

        let pae_logits = if self.use_separate_heads {
            let is_same_chain = asym_id
                .unsqueeze(-1)
                .eq_tensor(&asym_id.unsqueeze(-2))
                .to_kind(Kind::Float);
            let is_different_chain = asym_id
                .unsqueeze(-1)
                .ne_tensor(&asym_id.unsqueeze(-2))
                .to_kind(Kind::Float);
            let intra = self
                .to_pae_intra_logits
                .as_ref()
                .expect("separate PAE intra head")
                .forward(z)
                * is_same_chain.unsqueeze(-1);
            let inter = self
                .to_pae_inter_logits
                .as_ref()
                .expect("separate PAE inter head")
                .forward(z)
                * is_different_chain.unsqueeze(-1);
            intra + inter
        } else {
            self.to_pae_logits
                .as_ref()
                .expect("unified PAE head")
                .forward(z)
        };
        let pde_logits = if self.use_separate_heads {
            let z_sym = z + z.transpose(1, 2);
            let is_same_chain = asym_id
                .unsqueeze(-1)
                .eq_tensor(&asym_id.unsqueeze(-2))
                .to_kind(Kind::Float);
            let is_different_chain = asym_id
                .unsqueeze(-1)
                .ne_tensor(&asym_id.unsqueeze(-2))
                .to_kind(Kind::Float);
            let intra = self
                .to_pde_intra_logits
                .as_ref()
                .expect("separate PDE intra head")
                .forward(&z_sym)
                * is_same_chain.unsqueeze(-1);
            let inter = self
                .to_pde_inter_logits
                .as_ref()
                .expect("separate PDE inter head")
                .forward(&z_sym)
                * is_different_chain.unsqueeze(-1);
            intra + inter
        } else {
            self.to_pde_logits
                .as_ref()
                .expect("unified PDE head")
                .forward(&(z + z.transpose(1, 2)))
        };
        let resolved_logits = self.to_resolved_logits.forward(s);
        let plddt_logits = self.to_plddt_logits.forward(s);

        let pde = compute_aggregated_metric(&pde_logits, 32.0);
        let pred_log = if pred_distogram_logits.dim() == 5 {
            pred_distogram_logits.select(3, 0)
        } else {
            pred_distogram_logits.shallow_clone()
        };
        let pred_distogram_prob = pred_log
            .softmax(-1, Kind::Float)
            .repeat_interleave_self_int(multiplicity, Some(0), None);

        let nb = *pred_distogram_prob.size().last().unwrap();
        let mut contacts =
            Tensor::zeros(&[1, 1, 1, nb], (Kind::Float, pred_distogram_prob.device()));
        let n_contact_bins = 20_i64.min(nb);
        let _ = contacts.narrow(3, 0, n_contact_bins).fill_(1.0);
        let prob_contact =
            (pred_distogram_prob * &contacts).sum_dim_intlist(&[-1i64][..], false, Kind::Float);

        let token_pad_mask_m = token_pad_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .to_kind(Kind::Float);
        let n = token_pad_mask_m.size()[1];
        let device = token_pad_mask_m.device();
        let eye = Tensor::eye(n, (Kind::Float, device));
        let token_pad_pair_mask = token_pad_mask_m.unsqueeze(2)
            * token_pad_mask_m.unsqueeze(1)
            * (Tensor::ones([1, n, n], (Kind::Float, device)) - eye.unsqueeze(0));
        let token_pair_mask = &token_pad_pair_mask * &prob_contact;

        let plddt = compute_aggregated_metric(&plddt_logits, 1.0);
        let complex_plddt =
            plddt
                .multiply(&token_pad_mask_m)
                .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
                / token_pad_mask_m.sum_dim_intlist(&[-1i64][..], false, Kind::Float);

        let is_contact = d.less_tensor(&Tensor::from(8.0)).to_kind(Kind::Float);
        let is_different_chain = asym_id
            .unsqueeze(2)
            .ne_tensor(&asym_id.unsqueeze(1))
            .to_kind(Kind::Float)
            .repeat_interleave_self_int(multiplicity, Some(0), None);
        let is_ligand_token = mol_type
            .to_kind(Kind::Float)
            .eq_tensor(&Tensor::from(CHAIN_TYPE_NONPOLYMER as f64))
            .to_kind(Kind::Float)
            .repeat_interleave_self_int(multiplicity, Some(0), None);

        let token_interface_mask = (is_contact
            * &is_different_chain
            * (Tensor::ones_like(&is_ligand_token) - &is_ligand_token).unsqueeze(-1))
        .max_dim(2, false)
        .0;
        let token_non_interface_mask = (Tensor::ones_like(&token_interface_mask)
            - &token_interface_mask)
            * (Tensor::ones_like(&is_ligand_token) - &is_ligand_token);

        let ligand_weight = 20.0;
        let non_interface_weight = 1.0;
        let interface_weight = 10.0;
        let iplddt_weight = is_ligand_token * ligand_weight
            + &token_interface_mask * interface_weight
            + token_non_interface_mask * non_interface_weight;
        let complex_iplddt = plddt
            .multiply(&token_pad_mask_m)
            .multiply(&iplddt_weight)
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            / (token_pad_mask_m * &iplddt_weight).sum_dim_intlist(&[-1i64][..], false, Kind::Float);

        let pae = compute_aggregated_metric(&pae_logits, 32.0);

        let asym_m = asym_id.repeat_interleave_self_int(multiplicity, Some(0), None);
        let token_interface_pair_mask = &token_pair_mask
            * asym_m
                .unsqueeze(2)
                .ne_tensor(&asym_m.unsqueeze(1))
                .to_kind(Kind::Float);
        let complex_pde =
            pde.multiply(&token_pair_mask)
                .sum_dim_intlist(&[1i64, 2][..], false, Kind::Float)
                / token_pair_mask.sum_dim_intlist(&[1i64, 2][..], false, Kind::Float);
        let complex_ipde =
            pde.multiply(&token_interface_pair_mask).sum_dim_intlist(
                &[1i64, 2][..],
                false,
                Kind::Float,
            ) / (token_interface_pair_mask.sum_dim_intlist(&[1i64, 2][..], false, Kind::Float)
                + 1e-5);

        let (ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm) = compute_ptms(
            &pae_logits,
            x_pred,
            frames_idx,
            asym_id,
            mol_type,
            token_pad_mask,
            multiplicity,
        );

        ConfidenceOutput {
            pae_logits,
            pde_logits,
            plddt_logits,
            resolved_logits,
            pae,
            pde,
            plddt,
            complex_plddt,
            complex_iplddt,
            complex_pde,
            complex_ipde,
            ptm,
            iptm,
            ligand_iptm,
            protein_iptm,
            pair_chains_iptm,
        }
    }
}

/// Back-compat alias (older placeholder name).
pub type ConfidenceV2 = ConfidenceModule;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn from_confidence_model_args_reads_boltz2_defaults() {
        let j = serde_json::json!({
            "num_dist_bins": 64,
            "max_dist": 22.0,
            "pairformer_args": { "num_blocks": 8, "num_heads": 16 },
            "confidence_args": {
                "num_plddt_bins": 50,
                "num_pde_bins": 64,
                "num_pae_bins": 64,
                "use_separate_heads": true
            }
        });
        let cfg = ConfidenceModuleConfig::from_confidence_model_args(Some(&j), true);
        assert_eq!(cfg.pairformer_num_blocks, 8);
        assert_eq!(cfg.pairformer_num_heads, Some(16));
        assert!(cfg.use_separate_heads);
        assert_eq!(cfg.num_pae_bins, 64);
    }

    #[test]
    fn boltz2_confidence_weights_load_without_missing_module_keys() {
        tch::maybe_init_cuda();
        let cache = Path::new("/root/.cache/boltr");
        let hparams_path = cache.join("boltz2_hparams.json");
        let safetensors_path = cache.join("boltz2_conf.safetensors");
        if !hparams_path.is_file() || !safetensors_path.is_file() {
            eprintln!("skipping: cache artifacts not present");
            return;
        }
        let hparams_bytes = std::fs::read(&hparams_path).expect("read boltz2_hparams.json");
        let h = crate::boltz_hparams::Boltz2Hparams::from_json_slice(&hparams_bytes).unwrap();
        let token_level = h
            .other
            .get("token_level_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let confidence_cfg = ConfidenceModuleConfig::from_confidence_model_args(
            h.confidence_model_args.as_ref(),
            token_level,
        );
        assert_eq!(confidence_cfg.pairformer_num_blocks, 8);
        assert!(confidence_cfg.use_separate_heads);

        let mut model = crate::boltz2::model::Boltz2Model::with_all_options(
            Device::Cpu,
            h.resolved_token_s(),
            h.resolved_token_z(),
            h.resolved_num_pairformer_blocks(),
            h.resolved_bond_type_feature(),
            crate::boltz2::model::Boltz2DiffusionArgs::from_boltz2_hparams(&h),
            crate::boltz2::diffusion::AtomDiffusionConfig::from_boltz2_hparams(&h),
            Some(confidence_cfg),
            None,
            false,
        )
        .expect("with_all_options");
        let missing = model
            .load_partial_from_safetensors(&safetensors_path)
            .expect("load_partial_from_safetensors");
        let missing_confidence: Vec<_> = missing
            .iter()
            .filter(|k| k.starts_with("confidence_module."))
            .collect();
        assert!(
            missing_confidence.is_empty(),
            "confidence_module weights should load fully; missing: {:?}",
            missing_confidence.iter().take(10).collect::<Vec<_>>()
        );
        assert!(model.confidence_module().is_some());
    }

    #[test]
    fn confidence_forward_multiplicity2_separate_heads_smoke() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let b = 1_i64;
        let n = 16_i64;
        let n_atom = 64_i64;
        let mult = 2_i64;
        let token_s = 64_i64;
        let token_z = 32_i64;

        let cfg = ConfidenceModuleConfig {
            pairformer_num_blocks: 1,
            pairformer_num_heads: Some(4),
            use_separate_heads: true,
            ..Default::default()
        };

        let vs = tch::nn::VarStore::new(device);
        let cm = ConfidenceModule::new(vs.root(), device, token_s, token_z, &cfg);

        let s_inputs = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let s = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let x_pred = Tensor::randn(&[mult, n_atom, 3], (Kind::Float, device));
        let token_pad = Tensor::ones(&[b, n], (Kind::Float, device));
        let asym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let mol_type = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let token_to_rep = Tensor::eye(n_atom, (Kind::Float, device))
            .narrow(0, 0, n)
            .unsqueeze(0);
        let frames_idx = Tensor::zeros(&[b, n, 3], (Kind::Int64, device));
        let pdistogram = Tensor::randn(&[b, n, n, 64], (Kind::Float, device));

        let _out = cm.forward(
            &s_inputs,
            &s,
            &z,
            &x_pred,
            &token_pad,
            &asym_id,
            &mol_type,
            &token_to_rep,
            &frames_idx,
            &pdistogram,
            mult,
        );
    }
}
