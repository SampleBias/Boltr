//! Helpers for the checked-in collate golden (`tests/fixtures/collate_golden/`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use safetensors::SafeTensors;

/// Absolute path to [`trunk_smoke_collate.safetensors`](tests/fixtures/collate_golden/trunk_smoke_collate.safetensors) inside this crate.
pub fn trunk_smoke_collate_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/collate_golden/trunk_smoke_collate.safetensors")
}

/// Names and shapes of tensors in `trunk_smoke_collate.safetensors`.
pub fn trunk_smoke_collate_shapes() -> Result<Vec<(String, Vec<usize>)>> {
    trunk_smoke_collate_shapes_from_bytes(TRUNK_SMOKE_BYTES)
}

fn trunk_smoke_collate_shapes_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Vec<usize>)>> {
    let st = SafeTensors::deserialize(bytes).context("parse trunk_smoke_collate.safetensors")?;
    let mut out: Vec<(String, Vec<usize>)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name.to_string(), view.shape().to_vec()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Load shapes from a path (for tooling/tests that materialize the file elsewhere).
pub fn trunk_smoke_collate_shapes_from_path(path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    trunk_smoke_collate_shapes_from_bytes(&bytes)
}

const TRUNK_SMOKE_BYTES: &[u8] =
    include_bytes!("../tests/fixtures/collate_golden/trunk_smoke_collate.safetensors");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trunk_smoke_includes_s_inputs() {
        let shapes = trunk_smoke_collate_shapes().unwrap();
        let s = shapes
            .iter()
            .find(|(n, _)| n == "s_inputs")
            .expect("s_inputs present");
        assert_eq!(s.1, vec![1, 4, 384]);
    }

    #[test]
    fn trunk_smoke_collate_path_points_at_file() {
        let p = trunk_smoke_collate_path();
        assert!(p.is_file(), "expected {}", p.display());
    }
}
