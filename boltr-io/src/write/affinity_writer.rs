//! Affinity prediction JSON — mirrors [`BoltzAffinityWriter`](../../../boltz-reference/src/boltz/data/write/writer.py) (`affinity_{id}.json`).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// Fields written by Boltz `BoltzAffinityWriter` (`affinity_pred_value`, optional paired values).
#[derive(Debug, Clone, Serialize)]
pub struct AffinitySummary {
    pub affinity_pred_value: f64,
    pub affinity_probability_binary: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_pred_value1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_probability_binary1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_pred_value2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_probability_binary2: Option<f64>,
}

impl AffinitySummary {
    #[must_use]
    pub fn single(pred: f64, prob: f64) -> Self {
        Self {
            affinity_pred_value: pred,
            affinity_probability_binary: prob,
            affinity_pred_value1: None,
            affinity_probability_binary1: None,
            affinity_pred_value2: None,
            affinity_probability_binary2: None,
        }
    }

    /// Matches Python when `affinity_pred_value1` etc. are present (paired heads).
    #[must_use]
    pub fn paired(
        affinity_pred_value: f64,
        affinity_probability_binary: f64,
        v1: f64,
        p1: f64,
        v2: f64,
        p2: f64,
    ) -> Self {
        Self {
            affinity_pred_value,
            affinity_probability_binary,
            affinity_pred_value1: Some(v1),
            affinity_probability_binary1: Some(p1),
            affinity_pred_value2: Some(v2),
            affinity_probability_binary2: Some(p2),
        }
    }
}

/// Write `affinity_{record_id}.json` under `output_dir / record_id /`.
pub fn write_affinity_json(
    output_dir: &Path,
    record_id: &str,
    summary: &AffinitySummary,
) -> Result<std::path::PathBuf> {
    let dir = output_dir.join(record_id);
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(format!("affinity_{record_id}.json"));
    let j = serde_json::to_string_pretty(summary).context("serialize affinity summary")?;
    fs::write(&path, j).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_roundtrips_json_keys() {
        let s = AffinitySummary::single(-1.23, 0.88);
        let j = serde_json::to_value(&s).unwrap();
        assert_eq!(j["affinity_pred_value"], -1.23);
        assert_eq!(j["affinity_probability_binary"], 0.88);
        assert!(j.get("affinity_pred_value1").is_none());
    }
}
