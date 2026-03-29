//! Opt-in LibTorch goldens for template + diffusion (`BOLTR_RUN_TEMPLATE_GOLDEN`, `BOLTR_RUN_DIFFUSION_GOLDEN`).
//!
//! Generate fixtures with `scripts/export_*_golden.py` when the Python Boltz tree is available.
//! Default `cargo test` skips heavy assertions.

use std::path::Path;

fn template_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_TEMPLATE_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn diffusion_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_DIFFUSION_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn template_fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/template_golden/template_module_golden.safetensors")
}

fn diffusion_fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diffusion_golden/diffusion_step_golden.safetensors")
}

#[test]
fn template_module_golden_opt_in() {
    if !template_golden_requested() {
        return;
    }
    let p = template_fixture();
    assert!(
        p.is_file(),
        "missing {}; generate with scripts/export_template_golden.py when present upstream",
        p.display()
    );
}

#[test]
fn diffusion_step_golden_opt_in() {
    if !diffusion_golden_requested() {
        return;
    }
    let p = diffusion_fixture();
    assert!(
        p.is_file(),
        "missing {}; generate with scripts/export_diffusion_golden.py when present upstream",
        p.display()
    );
}
