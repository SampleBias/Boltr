//! Collate golden → trunk milestone: load [`trunk_smoke_collate.safetensors`](../../boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors)
//! (from `boltr-io`, regen `cargo run -p boltr-io --bin write_trunk_collate_from_fixture`) and run
//! [`Boltz2Model::predict_step_trunk`](boltr_backend_tch::Boltz2Model::predict_step_trunk) with
//! real [`MsaFeatures`](boltr_backend_tch::MsaFeatures).
//!
//! This does **not** strict-load a full checkpoint; it proves IO tensor names/shapes line up with the
//! backend entry point (§4.5 → §5.10 wiring).

use std::path::Path;

use boltr_backend_tch::checkpoint::load_tensor_from_safetensors;
use boltr_backend_tch::{Boltz2Model, MsaFeatures, RelPosFeatures};
use tch::Device;

fn collate_fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors")
}

#[test]
fn collate_smoke_predict_step_trunk_runs_with_msa() {
    tch::maybe_init_cuda();
    let path = collate_fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run: cargo run -p boltr-io --bin write_collate_golden",
        path.display()
    );

    let device = Device::Cpu;
    let token_s = 384_i64;
    let token_z = 128_i64;
    let num_pairformer = 4_i64;

    let model = Boltz2Model::with_options(device, token_s, token_z, Some(num_pairformer));

    let s_inputs = load_tensor_from_safetensors(&path, "s_inputs", device).unwrap();
    let msa = load_tensor_from_safetensors(&path, "msa", device).unwrap();
    let msa_mask = load_tensor_from_safetensors(&path, "msa_mask", device).unwrap();
    let has_deletion = load_tensor_from_safetensors(&path, "has_deletion", device).unwrap();
    let deletion_value = load_tensor_from_safetensors(&path, "deletion_value", device).unwrap();
    let msa_paired = load_tensor_from_safetensors(&path, "msa_paired", device).unwrap();
    let token_pad_mask = load_tensor_from_safetensors(&path, "token_pad_mask", device).unwrap();

    let b = s_inputs.size()[0];
    let n = s_inputs.size()[1];
    assert_eq!(s_inputs.size(), vec![b, n, token_s]);

    let asym_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
    let residue_index = tch::Tensor::arange(n, (tch::Kind::Int64, device))
        .view_(&[1, n])
        .expand(&[b, n], false);
    let entity_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
    let token_index = residue_index.shallow_clone();
    let sym_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
    let cyclic_period = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
    let rel = RelPosFeatures {
        asym_id: &asym_id,
        residue_index: &residue_index,
        entity_id: &entity_id,
        token_index: &token_index,
        sym_id: &sym_id,
        cyclic_period: &cyclic_period,
    };

    let feats = MsaFeatures {
        msa: &msa,
        msa_mask: &msa_mask,
        has_deletion: &has_deletion,
        deletion_value: &deletion_value,
        msa_paired: &msa_paired,
        token_pad_mask: &token_pad_mask,
    };

    let (s_out, z_out) = model
        .predict_step_trunk(&s_inputs, &rel, None, None, None, Some(0), Some(&feats))
        .expect("predict_step_trunk");

    assert_eq!(s_out.size(), vec![b, n, token_s]);
    assert_eq!(z_out.size(), vec![b, n, n, token_z]);
}
