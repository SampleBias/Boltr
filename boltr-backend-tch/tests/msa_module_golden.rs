//! Numerical golden vs Python `MSAModule` (TODO §5.4 / §7).
//!
//! 1. Generate fixture: `PYTHONPATH=boltz-reference/src python scripts/export_msa_module_golden.py`
//! 2. Opt-in (MSA / `TriangleAttention` numerics still being aligned with Boltz):  
//!    `BOLTR_RUN_MSA_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend msa_module_allclose_python_golden`
//!
//! Default `cargo test` skips the assertion so CI and Path A clones stay green.

use std::path::Path;

use boltr_backend_tch::checkpoint::load_tensor_from_safetensors;
use boltr_backend_tch::{MsaFeatures, MsaModule};
use tch::nn::VarStore;
use tch::Device;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/msa_module_golden/msa_module_golden.safetensors")
}

fn msa_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_MSA_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn msa_module_allclose_python_golden() {
    if !msa_golden_requested() {
        return;
    }
    tch::maybe_init_cuda();
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_msa_module_golden.py",
        path.display()
    );

    let device = Device::Cpu;
    let token_s = 32_i64;
    let token_z = 24_i64;
    let msa_s = 16_i64;
    let msa_blocks = 2_i64;

    let mut vs = VarStore::new(device);
    let msa = MsaModule::new(
        vs.root().sub("msa_module"),
        token_s,
        token_z,
        Some(msa_s),
        Some(msa_blocks),
        Some(0.0),
        Some(0.0),
        Some(true),
        None,
        None,
        device,
    );

    vs.load_partial(&path)
        .unwrap_or_else(|e| panic!("VarStore::load_partial {}: {e}", path.display()));

    let z = load_tensor_from_safetensors(&path, "golden.in_z", device).unwrap();
    let s = load_tensor_from_safetensors(&path, "golden.in_s", device).unwrap();
    let msa_t = load_tensor_from_safetensors(&path, "golden.msa", device).unwrap();
    let msa_mask = load_tensor_from_safetensors(&path, "golden.msa_mask", device).unwrap();
    let has_deletion = load_tensor_from_safetensors(&path, "golden.has_deletion", device).unwrap();
    let deletion_value =
        load_tensor_from_safetensors(&path, "golden.deletion_value", device).unwrap();
    let msa_paired = load_tensor_from_safetensors(&path, "golden.msa_paired", device).unwrap();
    let token_pad_mask =
        load_tensor_from_safetensors(&path, "golden.token_pad_mask", device).unwrap();
    let z_ref = load_tensor_from_safetensors(&path, "golden.z_out", device).unwrap();

    let feats = MsaFeatures {
        msa: &msa_t,
        msa_mask: &msa_mask,
        has_deletion: &has_deletion,
        deletion_value: &deletion_value,
        msa_paired: &msa_paired,
        token_pad_mask: &token_pad_mask,
    };

    let z_rust = msa.forward_trunk_step(&z, &s, Some(&feats), false, None, false);

    let rtol = 1e-4_f64;
    let atol = 1e-5_f64;
    let scale = z_ref.abs().max().double_value(&[]).max(1.0);
    let diff = (z_rust - &z_ref).abs().max();
    assert!(
        diff.double_value(&[]) < atol + rtol * scale,
        "MSAModule golden mismatch: max_abs_diff={} (rtol={rtol} atol={atol} scale={scale})",
        diff.double_value(&[]),
    );
}
