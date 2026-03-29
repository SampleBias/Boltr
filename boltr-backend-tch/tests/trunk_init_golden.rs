//! Golden parity for `rel_pos` and `s_init` vs Python export (`scripts/export_trunk_init_golden.py`).
//!
//! 1. Generate: `PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py`
//! 2. Opt-in: `BOLTR_RUN_TRUNK_INIT_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend trunk_init_allclose_python_golden`

use std::path::Path;

use boltr_backend_tch::checkpoint::load_tensor_from_safetensors;
use boltr_backend_tch::{RelPosFeatures, RelativePositionEncoder};
use tch::nn::{Module, VarStore};
use tch::{Device, Kind, Tensor};

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/trunk_init_golden/trunk_init_golden.safetensors")
}

fn trunk_init_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_TRUNK_INIT_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn trunk_init_allclose_python_golden() {
    if !trunk_init_golden_requested() {
        return;
    }
    tch::maybe_init_cuda();
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_trunk_init_golden.py",
        path.display()
    );

    let device = Device::Cpu;
    let token_s = 32_i64;
    let token_z = 24_i64;

    let mut vs = VarStore::new(device);
    let rel = RelativePositionEncoder::new(
        vs.root().sub("rel_pos"),
        token_z,
        None,
        None,
        false,
        false,
        device,
    );
    vs.load_partial(&path)
        .unwrap_or_else(|e| panic!("VarStore::load_partial {}: {e}", path.display()));

    let b = 2_i64;
    let n = 7_i64;
    let asym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let residue_index = Tensor::arange(n, (Kind::Int64, device))
        .view_(&[1, n])
        .expand(&[b, n], false);
    let entity_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let token_index = residue_index.shallow_clone();
    let sym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let cyclic_period = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let rel_f = RelPosFeatures {
        asym_id: &asym_id,
        residue_index: &residue_index,
        entity_id: &entity_id,
        token_index: &token_index,
        sym_id: &sym_id,
        cyclic_period: &cyclic_period,
    };
    let rel_rust = rel.forward(&rel_f);
    let rel_ref = load_tensor_from_safetensors(&path, "golden.rel_pos_out", device).unwrap();
    let diff = (rel_rust - &rel_ref).abs().max();
    let rtol = 1e-4_f64;
    let atol = 1e-5_f64;
    let scale = rel_ref.abs().max().double_value(&[]).max(1.0);
    assert!(
        diff.double_value(&[]) < atol + rtol * scale,
        "rel_pos golden mismatch max_abs={}",
        diff.double_value(&[])
    );

    let s_w = load_tensor_from_safetensors(&path, "s_init.weight", device).unwrap();
    let s_in = load_tensor_from_safetensors(&path, "golden.s_in", device).unwrap();
    let s_exp = s_in.matmul(&s_w.tr());
    let s_ref = load_tensor_from_safetensors(&path, "golden.s_init_out", device).unwrap();
    let d2 = (s_exp - &s_ref).abs().max().double_value(&[]);
    assert!(d2 < 1e-4, "s_init mismatch max_abs={d2}");
}
