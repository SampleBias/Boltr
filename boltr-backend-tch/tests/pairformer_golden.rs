//! Numerical golden vs Python `PairformerLayer` (Boltz2 / v2 attention, §5.5 / §7).
//!
//! 1. Generate fixture: `PYTHONPATH=boltz-reference/src python scripts/export_pairformer_golden.py`
//! 2. Opt-in: `BOLTR_RUN_PAIRFORMER_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend pairformer_layer_allclose_python_golden`
//!
//! Default `cargo test` skips the assertion (LibTorch + fixture + numeric tolerance).

use std::path::Path;

use boltr_backend_tch::checkpoint::load_tensor_from_safetensors;
use boltr_backend_tch::PairformerLayer;
use tch::nn::VarStore;
use tch::Device;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors")
}

fn pairformer_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_PAIRFORMER_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn pairformer_layer_allclose_python_golden() {
    if !pairformer_golden_requested() {
        return;
    }
    tch::maybe_init_cuda();
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_pairformer_golden.py",
        path.display()
    );

    let device = Device::Cpu;
    let token_s = 32_i64;
    let token_z = 24_i64;
    let num_heads = 4_i64;

    let mut vs = VarStore::new(device);
    let layer = PairformerLayer::new(
        vs.root().sub("layers").sub("0"),
        token_s,
        token_z,
        Some(num_heads),
        Some(0.0),
        Some(32),
        Some(4),
        Some(false),
        Some(true),
        device,
    );

    vs.load_partial(&path)
        .unwrap_or_else(|e| panic!("VarStore::load_partial {}: {e}", path.display()));

    let s = load_tensor_from_safetensors(&path, "golden.in_s", device).unwrap();
    let z = load_tensor_from_safetensors(&path, "golden.in_z", device).unwrap();
    let mask = load_tensor_from_safetensors(&path, "golden.mask", device).unwrap();
    let pair_mask = load_tensor_from_safetensors(&path, "golden.pair_mask", device).unwrap();
    let s_ref = load_tensor_from_safetensors(&path, "golden.s_out", device).unwrap();
    let z_ref = load_tensor_from_safetensors(&path, "golden.z_out", device).unwrap();

    let (s_rust, z_rust) = layer.forward(&s, &z, &mask, &pair_mask, None, false);

    let rtol = 1e-4_f64;
    let atol = 1e-5_f64;
    for (name, a, b) in [("s_out", &s_rust, &s_ref), ("z_out", &z_rust, &z_ref)] {
        let scale = b.abs().max().double_value(&[]).max(1.0);
        let diff = (a - b).abs().max();
        assert!(
            diff.double_value(&[]) < atol + rtol * scale,
            "PairformerLayer golden mismatch ({name}): max_abs_diff={} (rtol={rtol} atol={atol} scale={scale})",
            diff.double_value(&[]),
        );
    }
}
