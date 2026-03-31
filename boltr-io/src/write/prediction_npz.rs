//! Token-level confidence tensors as `numpy.savez_compressed`-compatible `.npz` (ZIP of one `.npy`).
//!
//! Each file holds a **single** array under the keyword name Boltz uses with `np.savez_compressed(..., pae=…)`:
//! the ZIP entry is `{key}.npy` (e.g. `pae.npy`). Keys are **`pae`**, **`pde`**, **`plddt`**.
//!
//! Verify against a full Boltz run with `numpy.load("pae_*.npz").files` if strict upstream parity is required.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::ArrayView2;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::structure_v2_npz::write_npy_f32_c_order;

fn write_npz_one_f32_array(
    path: &Path,
    array_key: &str,
    shape: &[usize],
    data: &[f32],
) -> Result<()> {
    let npy = write_npy_f32_c_order(shape, data)?;
    let inner = format!("{array_key}.npy");
    let mut file = File::create(path).with_context(|| path.display().to_string())?;
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut file);
    zw.start_file(inner, opts)?;
    zw.write_all(&npy)?;
    zw.finish()?;
    Ok(())
}

/// Write `pae_{record_id}_model_{rank}.npz` under `output_dir / record_id /` with array key `pae`.
pub fn write_pae_npz_path(
    output_dir: &Path,
    record_id: &str,
    model_rank: usize,
    pae: ArrayView2<f32>,
) -> Result<std::path::PathBuf> {
    let dir = output_dir.join(record_id);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(super::writer::pae_npz_filename(record_id, model_rank));
    let shape = vec![pae.nrows(), pae.ncols()];
    let data: Vec<f32> = pae.iter().copied().collect();
    write_npz_one_f32_array(&path, "pae", &shape, &data)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Write `pde_{record_id}_model_{rank}.npz` under `output_dir / record_id /` with array key `pde`.
pub fn write_pde_npz_path(
    output_dir: &Path,
    record_id: &str,
    model_rank: usize,
    pde: ArrayView2<f32>,
) -> Result<std::path::PathBuf> {
    let dir = output_dir.join(record_id);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(super::writer::pde_npz_filename(record_id, model_rank));
    let shape = vec![pde.nrows(), pde.ncols()];
    let data: Vec<f32> = pde.iter().copied().collect();
    write_npz_one_f32_array(&path, "pde", &shape, &data)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Write `plddt_{record_id}_model_{rank}.npz` under `output_dir / record_id /` with array key `plddt` (1-D `(N,)`).
pub fn write_plddt_npz_path(
    output_dir: &Path,
    record_id: &str,
    model_rank: usize,
    plddt: &[f32],
) -> Result<std::path::PathBuf> {
    let dir = output_dir.join(record_id);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(super::writer::plddt_npz_filename(record_id, model_rank));
    let shape = vec![plddt.len()];
    write_npz_one_f32_array(&path, "plddt", &shape, plddt)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use zip::ZipArchive;

    #[test]
    fn pae_npz_has_numpy_magic_and_key() {
        let dir =
            std::env::temp_dir().join(format!("boltr_pae_npz_{:016x}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = ndarray::arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
        let path = write_pae_npz_path(&dir, "case1", 0, a.view()).unwrap();
        let buf = std::fs::read(&path).unwrap();
        let mut za = ZipArchive::new(std::io::Cursor::new(&buf)).unwrap();
        let mut names: Vec<String> = (0..za.len())
            .map(|i| za.by_index(i).unwrap().name().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["pae.npy"]);
        let mut f = za.by_name("pae.npy").unwrap();
        let mut npy = Vec::new();
        f.read_to_end(&mut npy).unwrap();
        assert!(npy.starts_with(b"\x93NUMPY"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
