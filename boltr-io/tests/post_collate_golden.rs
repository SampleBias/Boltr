//! Post-collate integration: [`collate_inference_batches`](boltr_io::collate_inference_batches) on
//! `load_input_smoke` + [`trunk_smoke_feature_batch_from_inference_input`](boltr_io::trunk_smoke_feature_batch_from_inference_input).
//!
//! Golden [`trunk_smoke_collate.safetensors`](fixtures/collate_golden/trunk_smoke_collate.safetensors)
//! is **Rust-native**: regenerate with `cargo run -p boltr-io --bin write_trunk_collate_from_fixture`.
//! Per-key numeric parity (including synthetic zero `s_inputs` for embedder smoke).

use std::path::Path;

use boltr_io::collate_inference_batches;
use boltr_io::compare_inference_collate_to_safetensors;
use boltr_io::{load_input, parse_manifest_path, trunk_smoke_feature_batch_from_inference_input};

const TRUNK_SMOKE_GOLDEN: &[u8] =
    include_bytes!("fixtures/collate_golden/trunk_smoke_collate.safetensors");

#[test]
fn post_collate_trunk_smoke_matches_golden_allclose() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke");
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let input = load_input(
        &manifest.records[0],
        &dir,
        &dir,
        None,
        None,
        None,
        false,
    )
    .expect("load_input");
    let template_dim = 4;
    let fb = trunk_smoke_feature_batch_from_inference_input(&input, template_dim);
    let coll = collate_inference_batches(&[fb], 0.0, 0, 0).expect("collate");

    compare_inference_collate_to_safetensors(&coll, TRUNK_SMOKE_GOLDEN, 1e-6, 1e-5)
        .expect("golden parity");
}
