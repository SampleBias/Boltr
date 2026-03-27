//! Boltz2 inference dataset: manifest JSON + `load_input` aligned with
//! [`boltz-reference/src/boltz/data/module/inferencev2.py`](../../boltz-reference/src/boltz/data/module/inferencev2.py).
//!
//! Loads preprocess artifacts (`StructureV2` `.npz`, per-chain `MSA` `.npz`, optional template
//! structures). Residue constraints and extra molecules are not implemented yet.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::a3m::A3mMsa;
use crate::boltz_const::MAX_MSA_SEQS;
use crate::featurizer::{process_msa_features, process_token_features, MsaFeatureTensors, TokenFeatureTensors};
use crate::msa_npz::read_msa_npz_path;
use crate::structure_v2::StructureV2Tables;
use crate::structure_v2_npz::read_structure_v2_npz_path;
use crate::tokenize::boltz2::tokenize_structure;

fn default_true() -> bool {
    true
}

/// Boltz `StructureInfo` (manifest); all fields optional in JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructureInfo {
    #[serde(default)]
    pub resolution: Option<f64>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub deposited: Option<String>,
    #[serde(default)]
    pub released: Option<String>,
    #[serde(default)]
    pub revised: Option<String>,
    #[serde(default)]
    pub num_chains: Option<i32>,
    #[serde(default)]
    pub num_interfaces: Option<i32>,
    /// Boltz JSON key is `pH`.
    #[serde(default, rename = "pH")]
    pub ph: Option<f64>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

/// Boltz `ChainInfo` — `cluster_id` / `msa_id` may be int or string in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boltz2ChainInfo {
    pub chain_id: i32,
    pub chain_name: String,
    pub mol_type: i32,
    pub cluster_id: Value,
    pub msa_id: Value,
    pub num_residues: i32,
    #[serde(default = "default_true")]
    pub valid: bool,
    #[serde(default)]
    pub entity_id: Option<Value>,
}

/// Boltz `InterfaceInfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boltz2InterfaceInfo {
    pub chain_1: i32,
    pub chain_2: i32,
    #[serde(default = "default_true")]
    pub valid: bool,
}

/// Template entry in a [`Boltz2Record`]; only `name` is required for filesystem paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateInfo {
    pub name: String,
}

/// Boltz `Record` (manifest row).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boltz2Record {
    pub id: String,
    #[serde(default)]
    pub structure: StructureInfo,
    pub chains: Vec<Boltz2ChainInfo>,
    #[serde(default)]
    pub interfaces: Vec<Boltz2InterfaceInfo>,
    #[serde(default)]
    pub inference_options: Option<Value>,
    #[serde(default)]
    pub templates: Option<Vec<TemplateInfo>>,
    #[serde(default)]
    pub md: Option<Value>,
    #[serde(default)]
    pub affinity: Option<Value>,
}

/// Boltz `Manifest` — either `{"records":[...]}` or a bare JSON array of records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boltz2Manifest {
    pub records: Vec<Boltz2Record>,
}

/// Parse manifest JSON (object with `records` or top-level array of records).
pub fn parse_manifest_json(data: &[u8]) -> Result<Boltz2Manifest> {
    let v: Value =
        serde_json::from_slice(data).context("parse manifest JSON: invalid JSON")?;
    match v {
        Value::Array(_) => {
            let records: Vec<Boltz2Record> =
                serde_json::from_value(v).context("parse manifest: records array")?;
            Ok(Boltz2Manifest { records })
        }
        Value::Object(_) => serde_json::from_value(v).context("parse manifest: object"),
        _ => bail!("manifest must be a JSON object or array"),
    }
}

/// Parse manifest from a file path.
pub fn parse_manifest_path(path: &Path) -> Result<Boltz2Manifest> {
    let data = std::fs::read(path).with_context(|| path.display().to_string())?;
    parse_manifest_json(&data).with_context(|| format!("manifest {}", path.display()))
}

/// Loaded preprocess bundle matching Python `Input` (subset: no RDKit / pickle fields yet).
#[derive(Debug, Clone)]
pub struct Boltz2InferenceInput {
    pub structure: StructureV2Tables,
    pub msas: HashMap<i32, A3mMsa>,
    pub record: Boltz2Record,
    /// Template id → structure tables (`{record.id}_{template_id}.npz`).
    pub templates: Option<HashMap<String, StructureV2Tables>>,
}

fn msa_id_for_path(msa_id: &Value) -> Result<String> {
    match msa_id {
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => Ok(s.clone()),
        _ => bail!("msa_id must be a number or string, got {msa_id}"),
    }
}

fn msa_id_is_active(msa_id: &Value) -> bool {
    match msa_id {
        Value::Number(n) => n.as_i64() != Some(-1),
        Value::String(s) => s != "-1",
        _ => true,
    }
}

/// Load preprocess data for one record (Python `load_input`).
///
/// **Supported:** structure + MSAs; optional template `.npz` when `record.templates` and
/// `template_dir` are set. When `affinity == true`, loads
/// `{target_dir}/{id}/pre_affinity_{id}.npz` instead of `{target_dir}/{id}.npz` (Boltz preprocess layout).
///
/// **Not implemented:** `constraints_dir`, `extra_mols_dir` (returns an error if set).
pub fn load_input(
    record: &Boltz2Record,
    target_dir: &Path,
    msa_dir: &Path,
    constraints_dir: Option<&Path>,
    template_dir: Option<&Path>,
    extra_mols_dir: Option<&Path>,
    affinity: bool,
) -> Result<Boltz2InferenceInput> {
    if constraints_dir.is_some() {
        bail!("load_input: residue constraints loading is not implemented; pass constraints_dir=None");
    }
    if extra_mols_dir.is_some() {
        bail!("load_input: extra_mols pickle loading is not implemented; pass extra_mols_dir=None");
    }

    let structure_path = if affinity {
        target_dir
            .join(&record.id)
            .join(format!("pre_affinity_{}.npz", record.id))
    } else {
        target_dir.join(format!("{}.npz", record.id))
    };
    let structure = read_structure_v2_npz_path(&structure_path).with_context(|| {
        format!(
            "StructureV2.load: {}",
            structure_path.display()
        )
    })?;

    let mut msas = HashMap::new();
    for chain in &record.chains {
        if !msa_id_is_active(&chain.msa_id) {
            continue;
        }
        let fname = msa_id_for_path(&chain.msa_id)?;
        let msa_path = msa_dir.join(format!("{fname}.npz"));
        let msa = read_msa_npz_path(&msa_path).with_context(|| {
            format!("MSA.load: {}", msa_path.display())
        })?;
        msas.insert(chain.chain_id, msa);
    }

    let templates = match (template_dir, &record.templates) {
        (Some(td), Some(infos)) if !infos.is_empty() => {
            let mut map = HashMap::new();
            for t in infos {
                let path = td.join(format!("{}_{}.npz", record.id, t.name));
                let tables = read_structure_v2_npz_path(&path).with_context(|| {
                    format!("template StructureV2.load: {}", path.display())
                })?;
                map.insert(t.name.clone(), tables);
            }
            Some(map)
        }
        _ => None,
    };

    Ok(Boltz2InferenceInput {
        structure,
        msas,
        record: record.clone(),
        templates,
    })
}

/// Token-level features after `load_input`: `tokenize_structure` + `process_token_features`.
///
/// Aligns with Python `Boltz2Tokenizer.tokenize` → `Boltz2Featurizer.process` **token** slice only
/// (non-affinity). `msas`, templates, and molecules are ignored here; add `process_msa_features` /
/// atom paths separately.
#[must_use]
pub fn token_features_from_inference_input(input: &Boltz2InferenceInput) -> TokenFeatureTensors {
    let (tokens, bonds) = tokenize_structure(&input.structure, None);
    process_token_features(&tokens, &bonds, None)
}

/// MSA tensors after `load_input`: `tokenize_structure` + [`process_msa_features`](crate::featurizer::process_msa_features).
///
/// Uses deterministic RNG seed **42** (matches typical Boltz inference seeding in `inferencev2`).
/// Templates, constraints, and affinity paths are out of scope here.
#[must_use]
pub fn msa_features_from_inference_input(input: &Boltz2InferenceInput) -> MsaFeatureTensors {
    let (tokens, _bonds) = tokenize_structure(&input.structure, None);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    process_msa_features(
        &tokens,
        &input.structure,
        &input.msas,
        &mut rng,
        MAX_MSA_SEQS,
        MAX_MSA_SEQS,
        None,
        false,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_object_and_array() {
        let obj = br#"{"records":[{"id":"x","structure":{},"chains":[]}]}"#;
        let m = parse_manifest_json(obj).unwrap();
        assert_eq!(m.records.len(), 1);
        assert_eq!(m.records[0].id, "x");

        let arr = br#"[{"id":"y","structure":{},"chains":[]}]"#;
        let m = parse_manifest_json(arr).unwrap();
        assert_eq!(m.records[0].id, "y");
    }
}
