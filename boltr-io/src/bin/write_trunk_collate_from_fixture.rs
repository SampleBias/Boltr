//! Regenerate `tests/fixtures/collate_golden/trunk_smoke_collate.safetensors` from Rust
//! `load_input_smoke` + `trunk_smoke_feature_batch_from_inference_input` + `collate_inference_batches`.
//!
//! ```text
//! cargo run -p boltr-io --bin write_trunk_collate_from_fixture
//! ```

use std::env;
use std::path::{Path, PathBuf};

use boltr_io::inference_collate_serialize::write_inference_collate_golden;
use boltr_io::{
    collate_inference_batches, load_input, parse_manifest_path,
    trunk_smoke_feature_batch_from_inference_input,
};

fn main() -> anyhow::Result<()> {
    let manifest_dir: PathBuf = match env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke"),
    };
    let out: PathBuf = match env::args().nth(2) {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/collate_golden/trunk_smoke_collate.safetensors"),
    };

    let template_dim: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let manifest = parse_manifest_path(&manifest_dir.join("manifest.json"))?;
    let record = &manifest.records[0];
    let input = load_input(
        record,
        &manifest_dir,
        &manifest_dir,
        None,
        None,
        None,
        false,
    )?;
    let fb = trunk_smoke_feature_batch_from_inference_input(&input, template_dim);
    let coll = collate_inference_batches(&[fb], 0.0, 0, 0)?;

    write_inference_collate_golden(&coll, &out).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Wrote {} (Rust-native trunk collate golden)", out.display());
    Ok(())
}
