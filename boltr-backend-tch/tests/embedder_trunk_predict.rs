//! Integration: full [`InputEmbedder`] → trunk (`predict_step_trunk_from_embedder`).
//!
//! Uses random tensors with shapes matching Boltz collate (no checkpoint).

use boltr_backend_tch::boltz2::{
    AtomDiffusionConfig, Boltz2DiffusionArgs, Boltz2Model, InputEmbedderFeats, MsaFeatures,
    RelPosFeatures,
};
use tch::{Device, Kind, Tensor};

#[test]
fn predict_step_trunk_from_embedder_runs() {
    tch::maybe_init_cuda();
    let device = Device::Cpu;
    let token_s = 64_i64;
    let token_z = 32_i64;
    let b = 1_i64;
    let n_tok = 4_i64;
    let n_atoms = 32_i64;

    let mut diff_args = Boltz2DiffusionArgs::default();
    diff_args.atoms_per_window_keys = diff_args.atoms_per_window_queries;
    let m = Boltz2Model::with_all_options(
        device,
        token_s,
        token_z,
        Some(1),
        false,
        diff_args,
        AtomDiffusionConfig::default(),
        None,
        None,
        false,
    )
    .expect("with_all_options");

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

    let msa = Tensor::zeros(&[b, 2, n_tok], (Kind::Int64, device));
    let msa_mask = Tensor::ones(&[b, 2, n_tok], (Kind::Float, device));
    let has_deletion = Tensor::zeros(&[b, 2, n_tok], (Kind::Int64, device));
    let deletion_value = Tensor::zeros(&[b, 2, n_tok], (Kind::Float, device));
    let msa_paired = Tensor::zeros(&[b, 2, n_tok], (Kind::Int64, device));
    let token_pad_mask = Tensor::ones(&[b, n_tok], (Kind::Float, device));
    let msa_feats = MsaFeatures {
        msa: &msa,
        msa_mask: &msa_mask,
        has_deletion: &has_deletion,
        deletion_value: &deletion_value,
        msa_paired: &msa_paired,
        token_pad_mask: &token_pad_mask,
    };

    let (s, z) = m
        .predict_step_trunk_from_embedder(
            &emb,
            false,
            &rel,
            None,
            None,
            None,
            &token_pad_mask,
            Some(0),
            Some(&msa_feats),
            None,
        )
        .expect("predict_step_trunk_from_embedder");
    assert_eq!(s.size(), vec![b, n_tok, token_s]);
    assert_eq!(z.size(), vec![b, n_tok, n_tok, token_z]);
}
