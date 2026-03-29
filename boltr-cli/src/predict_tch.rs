//! Full `predict_step` wiring and `predict_args` resolution for `boltr predict` (`--features tch`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use boltr_backend_tch::{Boltz2Hparams, Boltz2PredictArgs, PredictArgsCliOverrides, resolve_predict_args};

/// Load optional `hyper_parameters` JSON next to safetensors or from `BOLTR_HPARAMS_JSON`.
pub fn load_hparams_for_predict(cache_dir: &Path) -> Result<Boltz2Hparams> {
    if let Ok(p) = std::env::var("BOLTR_HPARAMS_JSON") {
        let path = PathBuf::from(p);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read BOLTR_HPARAMS_JSON {}", path.display()))?;
        return Boltz2Hparams::from_json_slice(&bytes)
            .with_context(|| format!("parse hparams {}", path.display()));
    }
    let candidate = cache_dir.join("boltz2_hparams.json");
    if candidate.is_file() {
        let bytes = std::fs::read(&candidate)
            .with_context(|| format!("read {}", candidate.display()))?;
        return Boltz2Hparams::from_json_slice(&bytes)
            .with_context(|| format!("parse {}", candidate.display()));
    }
    Ok(Boltz2Hparams::default())
}

/// Parse optional `predict_args:` from the input YAML file (Boltz-style).
pub fn yaml_predict_args_from_input_path(input: &Path) -> Result<Option<serde_json::Value>> {
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("read {}", input.display()))?;
    let v: serde_yaml::Value =
        serde_yaml::from_str(&text).with_context(|| format!("YAML {}", input.display()))?;
    match v.get("predict_args") {
        Some(node) => Ok(Some(serde_json::to_value(node).unwrap_or(serde_json::Value::Null))),
        None => Ok(None),
    }
}

/// Merge checkpoint hparams + YAML + CLI; write `boltr_predict_args.json` into `out_dir`.
pub fn write_resolved_predict_args(
    out_dir: &Path,
    cache_dir: &Path,
    input: &Path,
    cli: PredictArgsCliOverrides,
) -> Result<Boltz2PredictArgs> {
    let h = load_hparams_for_predict(cache_dir)?;
    let yaml = yaml_predict_args_from_input_path(input)?;
    let resolved = resolve_predict_args(&h, yaml.as_ref(), cli);
    let path = out_dir.join("boltr_predict_args.json");
    let j = serde_json::to_string_pretty(&resolved).context("serialize Boltz2PredictArgs")?;
    std::fs::write(&path, j).with_context(|| format!("write {}", path.display()))?;
    Ok(resolved)
}
