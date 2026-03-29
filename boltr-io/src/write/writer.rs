//! Confidence JSON and output naming — layout from [prediction.md](../../../../boltz-reference/docs/prediction.md)
//! (`confidence_{id}_model_{rank}.json`; sibling `plddt_*` / `pae_*` / `pde_*` `.npz` under each record dir).
//! Tensor payloads are written by [`crate::write::prediction_npz`](prediction_npz).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// Scalar confidence metrics written next to structures (Boltz `confidence_summary_dict`).
#[derive(Debug, Clone, Serialize, Default)]
pub struct ConfidenceSummary {
    pub confidence_score: f64,
    pub ptm: f64,
    pub iptm: f64,
    pub ligand_iptm: f64,
    pub protein_iptm: f64,
    pub complex_plddt: f64,
    pub complex_iplddt: f64,
    pub complex_pde: f64,
    pub complex_ipde: f64,
    /// Per-chain pTM (`chains_ptm` in Python).
    #[serde(default)]
    pub chains_ptm: HashMap<i32, f64>,
    /// Nested pair IPTM (`pair_chains_iptm`).
    #[serde(default)]
    pub pair_chains_iptm: HashMap<i32, HashMap<i32, f64>>,
}

/// Filename helpers aligned with Boltz `struct_dir` layout.
#[derive(Debug, Clone)]
pub struct PredictionFileNames {
    pub record_id: String,
}

impl PredictionFileNames {
    #[must_use]
    pub fn new(record_id: impl Into<String>) -> Self {
        Self {
            record_id: record_id.into(),
        }
    }

    #[must_use]
    pub fn struct_dir(&self, output_dir: &Path) -> PathBuf {
        output_dir.join(&self.record_id)
    }

    #[must_use]
    pub fn pdb_path(&self, output_dir: &Path, model_rank: usize) -> PathBuf {
        self.struct_dir(output_dir).join(format!(
            "{}_model_{model_rank}.pdb",
            self.record_id
        ))
    }

    #[must_use]
    pub fn mmcif_path(&self, output_dir: &Path, model_rank: usize) -> PathBuf {
        self.struct_dir(output_dir).join(format!(
            "{}_model_{model_rank}.cif",
            self.record_id
        ))
    }

    /// `pae_{record_id}_model_{rank}.npz` under `output_dir / record_id /`.
    #[must_use]
    pub fn pae_npz_path(&self, output_dir: &Path, model_rank: usize) -> PathBuf {
        self.struct_dir(output_dir).join(pae_npz_filename(&self.record_id, model_rank))
    }

    /// `pde_{record_id}_model_{rank}.npz` under `output_dir / record_id /`.
    #[must_use]
    pub fn pde_npz_path(&self, output_dir: &Path, model_rank: usize) -> PathBuf {
        self.struct_dir(output_dir).join(pde_npz_filename(&self.record_id, model_rank))
    }

    /// `plddt_{record_id}_model_{rank}.npz` under `output_dir / record_id /`.
    #[must_use]
    pub fn plddt_npz_path(&self, output_dir: &Path, model_rank: usize) -> PathBuf {
        self.struct_dir(output_dir).join(plddt_npz_filename(&self.record_id, model_rank))
    }
}

#[must_use]
pub fn confidence_json_filename(record_id: &str, model_rank: usize) -> String {
    format!("confidence_{record_id}_model_{model_rank}.json")
}

#[must_use]
pub fn pae_npz_filename(record_id: &str, model_rank: usize) -> String {
    format!("pae_{record_id}_model_{model_rank}.npz")
}

#[must_use]
pub fn pde_npz_filename(record_id: &str, model_rank: usize) -> String {
    format!("pde_{record_id}_model_{model_rank}.npz")
}

#[must_use]
pub fn plddt_npz_filename(record_id: &str, model_rank: usize) -> String {
    format!("plddt_{record_id}_model_{model_rank}.npz")
}

/// Write `confidence_{record_id}_model_{rank}.json` under `output_dir / record_id /`.
pub fn write_confidence_json(
    output_dir: &Path,
    record_id: &str,
    model_rank: usize,
    summary: &ConfidenceSummary,
) -> Result<PathBuf> {
    let dir = output_dir.join(record_id);
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(confidence_json_filename(record_id, model_rank));
    let j = serde_json::to_string_pretty(summary).context("serialize confidence")?;
    fs::write(&path, j).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn prediction_npz_paths_under_record_dir() {
        let p = PredictionFileNames::new("input_file1");
        let out = Path::new("/predictions");
        assert_eq!(
            p.pae_npz_path(out, 0),
            Path::new("/predictions/input_file1/pae_input_file1_model_0.npz")
        );
        assert_eq!(
            p.pde_npz_path(out, 1),
            Path::new("/predictions/input_file1/pde_input_file1_model_1.npz")
        );
        assert_eq!(
            p.plddt_npz_path(out, 2),
            Path::new("/predictions/input_file1/plddt_input_file1_model_2.npz")
        );
        assert_eq!(pae_npz_filename("x", 0), "pae_x_model_0.npz");
    }

    #[test]
    fn confidence_serializes_nested_maps() {
        let mut s = ConfidenceSummary {
            confidence_score: 0.9,
            ptm: 0.8,
            iptm: 0.7,
            ligand_iptm: 0.0,
            protein_iptm: 0.0,
            complex_plddt: 0.0,
            complex_iplddt: 0.0,
            complex_pde: 0.0,
            complex_ipde: 0.0,
            chains_ptm: HashMap::from([(0, 0.5)]),
            pair_chains_iptm: HashMap::from([(0, HashMap::from([(1, 0.4)]))]),
        };
        let j = serde_json::to_string_pretty(&s).unwrap();
        assert!(j.contains("\"chains_ptm\""));
        assert!(j.contains("\"pair_chains_iptm\""));
        // ensure Default works for empty maps
        s = ConfidenceSummary::default();
        let _ = serde_json::to_string(&s).unwrap();
    }
}
