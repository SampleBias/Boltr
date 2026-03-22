//! Parse Boltz YAML/JSON input files.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::BoltzInput;

/// Parse a `.yaml` / `.yml` input into structured config.
pub fn parse_input_path(path: impl AsRef<Path>) -> Result<BoltzInput> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    parse_input_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn parse_input_str(text: &str) -> Result<BoltzInput> {
    let v: BoltzInput = serde_yaml::from_str(text).context("serde_yaml")?;
    Ok(v)
}
