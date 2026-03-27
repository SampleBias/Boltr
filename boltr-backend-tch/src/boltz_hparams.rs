//! Partial Boltz2 hyperparameters from Lightning checkpoints (`hyper_parameters` dict).
//!
//! Export JSON with [`scripts/export_hparams_from_ckpt.py`](../../scripts/export_hparams_from_ckpt.py).

use anyhow::Result;
use serde::Deserialize;

/// Subset of keys used to size the Rust inference graph. Extend as needed.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Boltz2Hparams {
    #[serde(default)]
    pub token_s: Option<i64>,
    #[serde(default)]
    pub token_z: Option<i64>,
    #[serde(default)]
    pub num_blocks: Option<i64>,
    /// Matches Python `Boltz2(bond_type_feature=…)` when present in checkpoint hparams.
    #[serde(default)]
    pub bond_type_feature: Option<bool>,
    #[serde(default)]
    pub pairformer_args: Option<PairformerArgs>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PairformerArgs {
    #[serde(default)]
    pub token_s: Option<i64>,
    #[serde(default)]
    pub token_z: Option<i64>,
    #[serde(default)]
    pub num_blocks: Option<i64>,
}

impl Boltz2Hparams {
    /// Parse from JSON bytes (exported by the Python script).
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes)?;
        Ok(serde_json::from_value(v)?)
    }

    #[must_use]
    pub fn resolved_token_s(&self) -> i64 {
        self.token_s
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.token_s))
            .unwrap_or(384)
    }

    #[must_use]
    pub fn resolved_token_z(&self) -> i64 {
        self.token_z
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.token_z))
            .unwrap_or(128)
    }

    #[must_use]
    pub fn resolved_num_pairformer_blocks(&self) -> Option<i64> {
        self.num_blocks
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.num_blocks))
    }

    #[must_use]
    pub fn resolved_bond_type_feature(&self) -> bool {
        self.bond_type_feature.unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn parses_minimal_json() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert_eq!(h.resolved_token_s(), 384);
        assert_eq!(h.resolved_token_z(), 128);
        assert_eq!(h.resolved_num_pairformer_blocks(), Some(4));
        assert!(!h.resolved_bond_type_feature());
    }

    #[test]
    fn parses_bond_type_feature() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4, "bond_type_feature": true}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert!(h.resolved_bond_type_feature());
    }

    #[test]
    fn parses_committed_minimal_fixture() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hparams/minimal.json");
        let raw = std::fs::read(&p).expect("minimal.json");
        let h = Boltz2Hparams::from_json_slice(&raw).unwrap();
        assert_eq!(h.resolved_token_s(), 384);
        assert_eq!(h.resolved_token_z(), 128);
    }
}
