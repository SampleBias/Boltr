//! Prediction run summary JSON. Structure export and confidence JSON live in [`crate::write`].

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::config::BoltzInput;

#[derive(Debug, Serialize)]
pub struct PredictionRunSummary {
    pub input_path: String,
    pub chain_ids: Vec<String>,
    pub use_msa_server: bool,
    pub device: String,
    /// Recorded for the run; also feeds `diffusion_samples` in `boltr_predict_args.json` when
    /// `--diffusion-samples` is omitted (LibTorch build).
    pub num_samples: usize,
    pub backend_note: String,
    pub affinity: bool,
    pub use_potentials: bool,
    pub spike_only: bool,
    /// Relative to the output directory when present (LibTorch build writes this file).
    pub boltr_predict_args_path: Option<String>,
}

impl PredictionRunSummary {
    pub fn from_input(
        input_path: impl Into<String>,
        input: &BoltzInput,
        use_msa_server: bool,
        device: impl Into<String>,
        num_samples: usize,
        backend_note: impl Into<String>,
        affinity: bool,
        use_potentials: bool,
        spike_only: bool,
        boltr_predict_args_path: Option<String>,
    ) -> Self {
        Self {
            input_path: input_path.into(),
            chain_ids: input.summary_chain_ids(),
            use_msa_server,
            device: device.into(),
            num_samples,
            backend_note: backend_note.into(),
            affinity,
            use_potentials,
            spike_only,
            boltr_predict_args_path,
        }
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).with_context(|| format!("mkdir {}", p.display()))?;
        }
        let j = serde_json::to_string_pretty(self).context("serialize summary")?;
        fs::write(path, j).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}
