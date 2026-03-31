//! Full trunk [`InputEmbedder`] golden vs Python (`scripts/export_input_embedder_golden.py`).
//!
//! 1. Generate: `PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py`
//! 2. Opt-in: `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend input_embedder_allclose_python_golden`

use std::path::Path;

use boltr_backend_tch::checkpoint::load_tensor_from_safetensors;
use boltr_backend_tch::{AtomEncoderFlags, InputEmbedder, InputEmbedderFeats};
use tch::nn::VarStore;
use tch::Device;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/input_embedder_golden/input_embedder_golden.safetensors")
}

fn golden_requested() -> bool {
    std::env::var("BOLTR_RUN_INPUT_EMBEDDER_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn input_embedder_allclose_python_golden() {
    if !golden_requested() {
        return;
    }
    tch::maybe_init_cuda();
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_input_embedder_golden.py",
        path.display()
    );

    let device = Device::Cpu;
    let token_s = 32_i64;
    let token_z = 16_i64;
    let atom_s = 16_i64;
    let atom_z = 8_i64;
    let wq = 32_i64;
    let wk = 128_i64;
    let atom_feature_dim = 8_i64;
    let depth = 2_i64;
    let heads = 2_i64;
    let flags = AtomEncoderFlags {
        num_elements: 4,
        use_no_atom_char: true,
        use_atom_backbone_feat: false,
        use_residue_feats_atoms: false,
        backbone_feat_dim: 17,
        num_tokens: 33,
    };

    let mut vs = VarStore::new(device);
    let emb = InputEmbedder::new(
        vs.root().sub("input_embedder"),
        token_s,
        token_z,
        atom_s,
        atom_z,
        wq,
        wk,
        atom_feature_dim,
        depth,
        heads,
        flags,
        device,
    );
    vs.load_partial(&path)
        .unwrap_or_else(|e| panic!("VarStore::load_partial {}: {e}", path.display()));

    let ref_pos = load_tensor_from_safetensors(&path, "golden.in_ref_pos", device).unwrap();
    let ref_charge = load_tensor_from_safetensors(&path, "golden.in_ref_charge", device).unwrap();
    let ref_element = load_tensor_from_safetensors(&path, "golden.in_ref_element", device).unwrap();
    let atom_pad_mask =
        load_tensor_from_safetensors(&path, "golden.in_atom_pad_mask", device).unwrap();
    let ref_space_uid =
        load_tensor_from_safetensors(&path, "golden.in_ref_space_uid", device).unwrap();
    let atom_to_token =
        load_tensor_from_safetensors(&path, "golden.in_atom_to_token", device).unwrap();
    let res_type = load_tensor_from_safetensors(&path, "golden.in_res_type", device).unwrap();
    let profile = load_tensor_from_safetensors(&path, "golden.in_profile", device).unwrap();
    let deletion_mean =
        load_tensor_from_safetensors(&path, "golden.in_deletion_mean", device).unwrap();

    let feats = InputEmbedderFeats {
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

    let out = emb.forward(&feats, false);
    let ref_out = load_tensor_from_safetensors(&path, "golden.s_inputs", device).unwrap();
    let diff = (out - &ref_out).abs().max();
    let rtol = 1e-4_f64;
    let atol = 1e-5_f64;
    let scale = ref_out.abs().max().double_value(&[]).max(1.0);
    assert!(
        diff.double_value(&[]) < atol + rtol * scale,
        "InputEmbedder golden mismatch max_abs={}",
        diff.double_value(&[])
    );
}
