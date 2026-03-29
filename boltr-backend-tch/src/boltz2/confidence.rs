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
    to_pae_logits: tch::nn::Linear,
    to_pde_logits: tch::nn::Linear,
    to_plddt_logits: tch::nn::Linear,
    to_resolved_logits: tch::nn::Linear,
    token_level_confidence: bool,
}

impl ConfidenceHeads {
    fn new<'a>(path: Path<'a>, token_s: i64, token_z: i64, cfg: &ConfidenceModuleConfig) -> Self {
        let (pae, pde, plddt, resolved) = if cfg.use_separate_heads {
            panic!("use_separate_heads=true not implemented in Rust port yet");
        } else {
            (
                linear_no_bias(path.sub("to_pae_logits"), token_z, cfg.num_pae_bins),
                linear_no_bias(path.sub("to_pde_logits"), token_z, cfg.num_pde_bins),
                linear_no_bias(path.sub("to_plddt_logits"), token_s, cfg.num_plddt_bins),
                linear_no_bias(path.sub("to_resolved_logits"), token_s, 2),
            )
        };

        Self {
            to_pae_logits: pae,
            to_pde_logits: pde,
            to_plddt_logits: plddt,
            to_resolved_logits: resolved,
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

        let pae_logits = self.to_pae_logits.forward(z);
        let pde_logits = self.to_pde_logits.forward(&(z + z.transpose(1, 2)));
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
        let complex_plddt = plddt
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
        let complex_pde = pde
            .multiply(&token_pair_mask)
            .sum_dim_intlist(&[1i64, 2][..], false, Kind::Float)
            / token_pair_mask.sum_dim_intlist(&[1i64, 2][..], false, Kind::Float);
        let complex_ipde = pde
            .multiply(&token_interface_pair_mask)
            .sum_dim_intlist(&[1i64, 2][..], false, Kind::Float)
            / (token_interface_pair_mask.sum_dim_intlist(&[1i64, 2][..], false, Kind::Float)
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
