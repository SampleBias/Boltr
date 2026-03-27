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
    use ndarray::arr2;
    use safetensors::tensor::Dtype;

    use crate::collate_inference_batches;
    use crate::feature_batch::FeatureBatch;

    /// [`collate_two_msa_golden.safetensors`](../../tests/fixtures/collate_golden/collate_two_msa_golden.safetensors) (`scripts/dump_collate_two_example_golden.py`).
    const COLLATE_TWO_MSA_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/collate_golden/collate_two_msa_golden.safetensors");

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

    fn read_i64_le(buf: &[u8]) -> Vec<i64> {
        buf.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Python `pad_to_max` / NumPy mirror (`dump_collate_two_example_golden.py`) vs Rust `collate_inference_batches`.
    #[test]
    fn collate_two_msa_matches_golden_pad_to_max() {
        let st = safetensors::SafeTensors::deserialize(COLLATE_TWO_MSA_BYTES).unwrap();
        let tv = st.tensor("msa").expect("msa");
        assert_eq!(tv.dtype(), Dtype::I64);
        assert_eq!(tv.shape(), &[2, 2, 3]);
        let exp_flat = read_i64_le(tv.data());

        let mut a = FeatureBatch::new();
        a.insert_i64("msa", arr2(&[[1_i64, 2], [3, 4]]).into_dyn());
        let mut b = FeatureBatch::new();
        b.insert_i64("msa", arr2(&[[10_i64, 20, 30], [40, 50, 60]]).into_dyn());

        let out = collate_inference_batches(&[a, b], 0.0, 0, 0).unwrap();
        let got = out.batch.get_i64("msa").unwrap();
        assert_eq!(got.shape(), &[2, 2, 3]);
        let got_flat: Vec<i64> = got.iter().copied().collect();
        assert_eq!(got_flat, exp_flat);
    }
}
