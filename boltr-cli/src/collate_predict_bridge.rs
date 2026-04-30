//! Collated [`boltr_io::FeatureBatch`] → `tch` tensors for [`boltr_backend_tch::Boltz2Model::predict_step`].
//!
//! Bridges preprocess output (`manifest.json` + `.npz` next to the input YAML) to structure/mmCIF writers.

use anyhow::{bail, Context, Result};
use boltr_backend_tch::{
    AtomEncoderBatchFeats, Boltz2Model, ContactFeatures, InputEmbedderFeats, MsaFeatures,
    PredictStepFeats, PredictStepOutput, RelPosFeatures, SteeringParams,
};
use boltr_io::{FeatureBatch, FeatureTensor, InferenceCollateResult};
use ndarray::{stack, ArrayD, Axis};
use tch::{Device, Kind, Tensor};

fn f32_to_tensor(a: &ArrayD<f32>, device: Device) -> Tensor {
    let shape: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
    let data: Vec<f32> = a.iter().cloned().collect();
    Tensor::from_slice(&data).view(&*shape).to_device(device)
}

fn i64_to_tensor(a: &ArrayD<i64>, device: Device) -> Tensor {
    let shape: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
    let data: Vec<i64> = a.iter().cloned().collect();
    Tensor::from_slice(&data).view(&*shape).to_device(device)
}

fn take_f32(batch: &FeatureBatch, key: &str, device: Device) -> Result<Tensor> {
    let ft = batch
        .tensors
        .get(key)
        .with_context(|| format!("missing tensor key {key}"))?;
    match ft {
        FeatureTensor::F32(a) => Ok(f32_to_tensor(a, device)),
        _ => bail!("key {key}: expected f32"),
    }
}

fn take_i64(batch: &FeatureBatch, key: &str, device: Device) -> Result<Tensor> {
    let ft = batch
        .tensors
        .get(key)
        .with_context(|| format!("missing tensor key {key}"))?;
    match ft {
        FeatureTensor::I64(a) => Ok(i64_to_tensor(a, device)),
        _ => bail!("key {key}: expected i64"),
    }
}

/// MSA mask may be stored as int in some collates; backend expects float mask.
fn take_msa_mask(batch: &FeatureBatch, device: Device) -> Result<Tensor> {
    if let Some(a) = batch.get_f32("msa_mask") {
        return Ok(f32_to_tensor(a, device));
    }
    if let Some(a) = batch.get_i64("msa_mask") {
        let f: ArrayD<f32> = a.mapv(|v| v as f32);
        return Ok(f32_to_tensor(&f, device));
    }
    bail!("missing msa_mask")
}

fn excluded_f32_to_tensor(
    coll: &InferenceCollateResult,
    key: &str,
    device: Device,
) -> Result<Option<Tensor>> {
    let Some(values) = coll.excluded.get(key) else {
        return Ok(None);
    };
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        match value {
            FeatureTensor::F32(a) => arrays.push(a.view()),
            _ => bail!("excluded key {key}: expected f32"),
        }
    }
    let stacked = stack(Axis(0), &arrays)
        .with_context(|| format!("stack excluded key {key} along batch axis"))?;
    Ok(Some(f32_to_tensor(&stacked, device)))
}

/// Owned tensors for [`InputEmbedderFeats`] / [`PredictStepFeats`] (single batch, `B>=1`).
pub struct OwnedPredictTensors {
    ref_pos: Tensor,
    ref_charge: Tensor,
    ref_element: Tensor,
    atom_pad_mask: Tensor,
    ref_space_uid: Tensor,
    atom_to_token: Tensor,
    res_type: Tensor,
    profile: Tensor,
    deletion_mean: Tensor,
    profile_affinity: Option<Tensor>,
    deletion_mean_affinity: Option<Tensor>,
    asym_id: Tensor,
    residue_index: Tensor,
    entity_id: Tensor,
    token_index: Tensor,
    sym_id: Tensor,
    cyclic_period: Tensor,
    token_pad_mask: Tensor,
    mol_type: Tensor,
    msa: Tensor,
    msa_mask: Tensor,
    has_deletion: Tensor,
    deletion_value: Tensor,
    msa_paired: Tensor,
    token_bonds: Tensor,
    type_bonds: Option<Tensor>,
    contact_conditioning: Tensor,
    contact_threshold: Tensor,
    /// Placeholder; confidence path expects `[B, N_atoms]`-shaped indices when enabled.
    token_to_rep_atom: Tensor,
    frames_idx: Tensor,
    /// Atom-name one-hot (Boltz `ref_atom_name_chars`); required when checkpoint has `use_no_atom_char: false`.
    ref_atom_name_chars: Tensor,
    /// Optional; required when checkpoint enables `use_atom_backbone_feat`.
    atom_backbone_feat: Option<Tensor>,
    /// Token-level `modified` (int); used when checkpoint enables `use_residue_feats_atoms`.
    modified: Tensor,
    affinity_token_mask: Option<Tensor>,
    affinity_mw: Option<Tensor>,
}

impl OwnedPredictTensors {
    pub fn from_collate(coll: &InferenceCollateResult, device: Device) -> Result<Self> {
        let batch = &coll.batch;
        let ref_pos = take_f32(batch, "ref_pos", device)?;
        let ref_charge = take_f32(batch, "ref_charge", device)?;
        let ref_element = take_f32(batch, "ref_element", device)?;
        let atom_pad_mask = take_f32(batch, "atom_pad_mask", device)?;
        let ref_space_uid = take_i64(batch, "ref_space_uid", device)?;
        let res_type = take_f32(batch, "res_type", device)?;
        let profile = take_f32(batch, "profile", device)?;
        let deletion_mean = take_f32(batch, "deletion_mean", device)?;
        let profile_affinity = batch
            .get_f32("profile_affinity")
            .map(|a| f32_to_tensor(a, device));
        let deletion_mean_affinity = batch
            .get_f32("deletion_mean_affinity")
            .map(|a| f32_to_tensor(a, device));
        let asym_id = take_i64(batch, "asym_id", device)?;
        let residue_index = take_i64(batch, "residue_index", device)?;
        let entity_id = take_i64(batch, "entity_id", device)?;
        let token_index = take_i64(batch, "token_index", device)?;
        let sym_id = take_i64(batch, "sym_id", device)?;
        let cyclic_f = take_f32(batch, "cyclic_period", device)?;
        let cyclic_period = cyclic_f.to_kind(Kind::Int64);
        let token_pad_mask = take_f32(batch, "token_pad_mask", device)?;
        let n_tok = token_pad_mask.size()[1];
        let mut atom_to_token = take_f32(batch, "atom_to_token", device)?;
        let n_att = atom_to_token.size()[2];
        if n_att != n_tok {
            if n_att > n_tok {
                atom_to_token = atom_to_token.narrow(2, 0, n_tok);
            } else {
                bail!(
                    "atom_to_token last dim {n_att} < token_pad_mask token count {n_tok} (inconsistent preprocess collate)"
                );
            }
        }
        let mol_type = take_i64(batch, "mol_type", device)?;
        let msa = take_i64(batch, "msa", device)?;
        let msa_mask = take_msa_mask(batch, device)?;
        let has_deletion = take_i64(batch, "has_deletion", device)?;
        let deletion_value = take_f32(batch, "deletion_value", device)?;
        let msa_paired = take_i64(batch, "msa_paired", device)?;
        let token_bonds = take_f32(batch, "token_bonds", device)?;
        let type_bonds = batch
            .get_i64("type_bonds")
            .map(|a| i64_to_tensor(a, device));
        let contact_conditioning = take_f32(batch, "contact_conditioning", device)?;
        let contact_threshold = take_f32(batch, "contact_threshold", device)?;

        let b = token_pad_mask.size()[0];
        let n_atom = ref_pos.size()[1];
        let token_to_rep_atom = Tensor::zeros(&[b, n_atom], (Kind::Int64, device));
        let frames_idx = Tensor::zeros(&[b, n_tok, 3], (Kind::Int64, device));

        let ref_atom_name_chars = take_f32(batch, "ref_atom_name_chars", device)?;
        let atom_backbone_feat = batch
            .get_f32("atom_backbone_feat")
            .map(|a| f32_to_tensor(a, device));
        let modified = take_i64(batch, "modified", device)?;
        let affinity_token_mask = batch
            .get_f32("affinity_token_mask")
            .map(|a| f32_to_tensor(a, device));
        let affinity_mw = excluded_f32_to_tensor(coll, "affinity_mw", device)?.or_else(|| {
            batch
                .get_f32("affinity_mw")
                .map(|a| f32_to_tensor(a, device))
        });

        Ok(Self {
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
            res_type,
            profile,
            deletion_mean,
            profile_affinity,
            deletion_mean_affinity,
            asym_id,
            residue_index,
            entity_id,
            token_index,
            sym_id,
            cyclic_period,
            token_pad_mask,
            mol_type,
            msa,
            msa_mask,
            has_deletion,
            deletion_value,
            msa_paired,
            token_bonds,
            type_bonds,
            contact_conditioning,
            contact_threshold,
            token_to_rep_atom,
            frames_idx,
            ref_atom_name_chars,
            atom_backbone_feat,
            modified,
            affinity_token_mask,
            affinity_mw,
        })
    }

    pub fn input_embedder_feats<'a>(
        &'a self,
        atom_encoder_batch: Option<&'a AtomEncoderBatchFeats<'a>>,
    ) -> InputEmbedderFeats<'a> {
        InputEmbedderFeats {
            ref_pos: &self.ref_pos,
            ref_charge: &self.ref_charge,
            ref_element: &self.ref_element,
            atom_pad_mask: &self.atom_pad_mask,
            ref_space_uid: &self.ref_space_uid,
            atom_to_token: &self.atom_to_token,
            res_type: &self.res_type,
            profile: &self.profile,
            deletion_mean: &self.deletion_mean,
            profile_affinity: self.profile_affinity.as_ref(),
            deletion_mean_affinity: self.deletion_mean_affinity.as_ref(),
            atom_encoder_batch,
        }
    }

    pub fn rel_pos_feats(&self) -> RelPosFeatures<'_> {
        RelPosFeatures {
            asym_id: &self.asym_id,
            residue_index: &self.residue_index,
            entity_id: &self.entity_id,
            token_index: &self.token_index,
            sym_id: &self.sym_id,
            cyclic_period: &self.cyclic_period,
        }
    }

    pub fn msa_feats(&self) -> MsaFeatures<'_> {
        MsaFeatures {
            msa: &self.msa,
            msa_mask: &self.msa_mask,
            has_deletion: &self.has_deletion,
            deletion_value: &self.deletion_value,
            msa_paired: &self.msa_paired,
            token_pad_mask: &self.token_pad_mask,
        }
    }

    pub fn contact_feats(&self) -> ContactFeatures<'_> {
        ContactFeatures {
            contact_conditioning: &self.contact_conditioning,
            contact_threshold: &self.contact_threshold,
        }
    }

    pub fn predict_step_feats<'a>(
        &'a self,
        atom_encoder_batch: Option<&'a AtomEncoderBatchFeats<'a>>,
    ) -> PredictStepFeats<'a> {
        PredictStepFeats {
            token_pad_mask: &self.token_pad_mask,
            asym_id: &self.asym_id,
            mol_type: &self.mol_type,
            token_to_rep_atom: &self.token_to_rep_atom,
            frames_idx: &self.frames_idx,
            ref_pos: &self.ref_pos,
            ref_charge: &self.ref_charge,
            ref_element: &self.ref_element,
            atom_pad_mask: &self.atom_pad_mask,
            ref_space_uid: &self.ref_space_uid,
            atom_to_token: &self.atom_to_token,
            atom_encoder_batch,
            affinity_token_mask: self.affinity_token_mask.as_ref(),
            affinity_mw: self.affinity_mw.as_ref(),
        }
    }
}

/// Run full `predict_step` from a collated batch (single-example or multi-example stack).
///
/// Template bias is omitted here unless the model is built with a template module and callers extend
/// this path; dummy template tensors in the featurizer batch are unused when [`Boltz2Model`] has no
/// `TemplateV2Module` (default inference graph).
pub fn predict_step_from_collate(
    model: &Boltz2Model,
    coll: &InferenceCollateResult,
    recycling_steps: Option<i64>,
    sampling_steps: Option<i64>,
    diffusion_samples: i64,
    max_parallel_samples: Option<i64>,
    use_potentials: bool,
) -> Result<PredictStepOutput> {
    let device = model.device();
    let owned = OwnedPredictTensors::from_collate(coll, device)?;
    let atom_encoder_batch = AtomEncoderBatchFeats {
        ref_atom_name_chars: Some(&owned.ref_atom_name_chars),
        atom_backbone_feat: owned.atom_backbone_feat.as_ref(),
        res_type: Some(&owned.res_type),
        modified: Some(&owned.modified),
        mol_type: Some(&owned.mol_type),
    };
    let emb = owned.input_embedder_feats(Some(&atom_encoder_batch));
    let s_inputs = model.forward_s_inputs_from_embedder(&emb, false);
    let rel = owned.rel_pos_feats();
    let msa = owned.msa_feats();
    let contact = owned.contact_feats();
    let feats = owned.predict_step_feats(Some(&atom_encoder_batch));
    if model.affinity_mw_correction() && feats.affinity_mw.is_none() {
        bail!(
            "affinity_mw_correction is enabled, but affinity_mw is missing from collated features"
        );
    }

    let type_bonds = if model.bond_type_feature() {
        owned.type_bonds.as_ref()
    } else {
        None
    };
    if use_potentials {
        bail!(
            "preprocess predict bridge: --use-potentials is not wired to PotentialBatchFeats; refusing to run an unsteered diffusion sample"
        );
    }
    let steering: Option<SteeringParams> = None;

    model.predict_step(
        &s_inputs,
        &rel,
        Some(&owned.token_bonds),
        type_bonds,
        Some(&contact),
        recycling_steps,
        Some(&msa),
        None,
        &feats,
        sampling_steps,
        diffusion_samples,
        steering,
        None,
        max_parallel_samples,
        Some(&emb),
        false,
    )
}
