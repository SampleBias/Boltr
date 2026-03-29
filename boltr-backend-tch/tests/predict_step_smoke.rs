//! Full [`Boltz2Model::predict_step`] smoke (random tensors, no checkpoint).

use boltr_backend_tch::{
    Boltz2Model, InputEmbedderFeats, PredictStepFeats, RelPosFeatures,
};
use tch::{Device, Kind, Tensor};

#[test]
fn predict_step_random_smoke() {
    tch::maybe_init_cuda();
    let device = Device::Cpu;
    let token_s = 64_i64;
    let token_z = 32_i64;
    let b = 1_i64;
    let n_tok = 4_i64;
    let n_atoms = 32_i64;

    let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));

    let ref_pos = Tensor::randn(&[b, n_atoms, 3], (Kind::Float, device));
    let ref_charge = Tensor::randn(&[b, n_atoms], (Kind::Float, device));
    let ref_element = Tensor::randn(&[b, n_atoms, 128], (Kind::Float, device));
    let atom_pad_mask = Tensor::ones(&[b, n_atoms], (Kind::Float, device));
    let ref_space_uid = Tensor::zeros(&[b, n_atoms], (Kind::Int64, device));
    let atom_tok_idx = Tensor::zeros(&[b, n_atoms], (Kind::Int64, device));
    let atom_to_token = atom_tok_idx.one_hot(n_tok).to_kind(Kind::Float);
    let res_type = Tensor::randn(
        &[b, n_tok, boltr_backend_tch::BOLTZ_NUM_TOKENS],
        (Kind::Float, device),
    );
    let profile = Tensor::randn(
        &[b, n_tok, boltr_backend_tch::BOLTZ_NUM_TOKENS],
        (Kind::Float, device),
    );
    let deletion_mean = Tensor::randn(&[b, n_tok], (Kind::Float, device));

    let asym_id = Tensor::zeros(&[b, n_tok], (Kind::Int64, device));
    let residue_index = Tensor::arange(n_tok, (Kind::Int64, device))
        .view_(&[1, n_tok])
        .expand(&[b, n_tok], false);
    let entity_id = Tensor::zeros(&[b, n_tok], (Kind::Int64, device));
    let token_index = residue_index.shallow_clone();
    let sym_id = Tensor::zeros(&[b, n_tok], (Kind::Int64, device));
    let cyclic_period = Tensor::zeros(&[b, n_tok], (Kind::Int64, device));
    let rel = RelPosFeatures {
        asym_id: &asym_id,
        residue_index: &residue_index,
        entity_id: &entity_id,
        token_index: &token_index,
        sym_id: &sym_id,
        cyclic_period: &cyclic_period,
    };

    let emb = InputEmbedderFeats {
        ref_pos: &ref_pos,
        ref_charge: &ref_charge,
        ref_element: &ref_element,
        atom_pad_mask: &atom_pad_mask,
        ref_space_uid: &ref_space_uid,
        atom_to_token: &atom_to_token,
        res_type: &res_type,
        profile: &profile,
        deletion_mean: &deletion_mean,
        profile_affinity: None,
        deletion_mean_affinity: None,
        atom_encoder_batch: None,
    };

    let s_inputs = m.forward_s_inputs_from_embedder(&emb, false);

    let token_pad_mask = Tensor::ones(&[b, n_tok], (Kind::Float, device));
    let mol_type = Tensor::zeros(&[b, n_tok], (Kind::Int64, device));
    let frames_idx = Tensor::zeros(&[b, n_tok, 3], (Kind::Int64, device));
    let feats = PredictStepFeats {
        token_pad_mask: &token_pad_mask,
        asym_id: &asym_id,
        mol_type: &mol_type,
        token_to_rep_atom: &atom_tok_idx,
        frames_idx: &frames_idx,
        ref_pos: &ref_pos,
        ref_charge: &ref_charge,
        ref_element: &ref_element,
        atom_pad_mask: &atom_pad_mask,
        ref_space_uid: &ref_space_uid,
        atom_to_token: &atom_to_token,
        atom_encoder_batch: None,
        affinity_token_mask: None,
        affinity_mw: None,
    };

    let out = m
        .predict_step(
            &s_inputs,
            &rel,
            None,
            None,
            None,
            Some(0),
            None,
            None,
            &feats,
            Some(3),
            1,
            None,
            None,
            None,
            None,
            false,
        )
        .expect("predict_step");

    let sz = out.diffusion.sample_atom_coords.size();
    assert_eq!(sz.len(), 3, "coords rank");
    assert_eq!(sz[2], 3, "xyz");
}
