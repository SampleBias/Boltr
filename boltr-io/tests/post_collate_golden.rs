//! Post-collate integration: [`collate_inference_batches`](boltr_io::collate_inference_batches) on
//! `load_input_smoke` + [`trunk_smoke_feature_batch_from_inference_input`](boltr_io::trunk_smoke_feature_batch_from_inference_input).
//!
//! Compares **key coverage** against `trunk_smoke_collate.safetensors`: every tensor name in that
//! file (except embedder-only `s_inputs`) must appear in the Rust collated result (stacked `batch`
//! or `excluded`). Numeric shapes are **not** compared here — the golden was built from a
//! different nominal target than `load_input_smoke`; full post-collate `allclose` belongs with a
//! fixture generated from the same manifest + preprocess as the golden.

use std::path::Path;

use boltr_io::collate_inference_batches;
use boltr_io::feature_batch::FeatureTensor;
use boltr_io::{load_input, parse_manifest_path, trunk_smoke_feature_batch_from_inference_input};
use safetensors::SafeTensors;

const TRUNK_SMOKE_GOLDEN: &[u8] =
    include_bytes!("fixtures/collate_golden/trunk_smoke_collate.safetensors");

fn tensor_present_in_collate<'a>(
    coll: &'a boltr_io::InferenceCollateResult,
    name: &str,
) -> Option<&'a FeatureTensor> {
    coll.batch
        .tensors
        .get(name)
        .or_else(|| coll.excluded.get(name).and_then(|v| v.first()))
}

#[test]
fn post_collate_trunk_smoke_golden_key_coverage() {
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

    let st = SafeTensors::deserialize(TRUNK_SMOKE_GOLDEN).expect("safetensors");
    for (name, _) in st.tensors() {
        if name == "s_inputs" {
            continue;
        }
        assert!(
            tensor_present_in_collate(&coll, &name).is_some(),
            "golden key {name} missing from Rust collated batch/excluded"
        );
    }
}
