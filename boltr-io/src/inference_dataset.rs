//! Boltz2 inference dataset: manifest JSON + `load_input` aligned with
//! [`boltz-reference/src/boltz/data/module/inferencev2.py`](../../boltz-reference/src/boltz/data/module/inferencev2.py).
//!
//! Loads preprocess artifacts (`StructureV2` `.npz`, per-chain `MSA` `.npz`, optional template
//! structures, optional residue constraints). Extra molecules: load `*.json` from
//! [`CcdMolProvider::load_all_json_in_dir`](crate::ccd::CcdMolProvider::load_all_json_in_dir) when
//! `extra_mols_dir` is set (Boltz pickle cache must be pre-extracted to JSON).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::a3m::A3mMsa;
use crate::boltz_const::MAX_MSA_SEQS;
use crate::ccd::CcdMolProvider;
use crate::feature_batch::FeatureBatch;
use crate::featurizer::{
    inference_ensemble_features, load_dummy_templates_features, pad_template_tdim,
    process_atom_features, process_msa_features, process_symmetry_features,
    process_template_features, process_token_features,
    AtomFeatureConfig, AtomFeatureTensors, InferenceAtomRefProvider, MsaFeatureTensors,
    StandardAminoAcidRefData, TemplateAlignment, TokenFeatureTensors,
};
use crate::msa_npz::read_msa_npz_path;
use crate::residue_constraints::ResidueConstraints;
use crate::structure_v2::StructureV2Tables;
use crate::structure_v2_npz::read_structure_v2_npz_path;
use crate::tokenize::boltz2::{tokenize_structure, TokenBondV2, TokenData};

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

/// Template entry in a [`Boltz2Record`] (`boltz.data.types.TemplateInfo`).
/// For real [`process_template_features`](crate::featurizer::process_template_features), set
/// `query_chain` / `template_chain` and residue bounds; `name` selects `{record.id}_{name}.npz`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateInfo {
    pub name: String,
    #[serde(default)]
    pub query_chain: String,
    #[serde(default)]
    pub query_st: i32,
    #[serde(default)]
    pub query_en: i32,
    #[serde(default)]
    pub template_chain: String,
    #[serde(default)]
    pub template_st: i32,
    #[serde(default)]
    pub template_en: i32,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub threshold: Option<f32>,
}

impl From<&TemplateInfo> for TemplateAlignment {
    fn from(t: &TemplateInfo) -> Self {
        Self {
            name: t.name.clone(),
            query_chain: t.query_chain.clone(),
            query_st: t.query_st,
            query_en: t.query_en,
            template_chain: t.template_chain.clone(),
            template_st: t.template_st,
            template_en: t.template_en,
            force: t.force,
            threshold: t.threshold,
        }
    }
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
    let v: Value = serde_json::from_slice(data).context("parse manifest JSON: invalid JSON")?;
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

/// Loaded preprocess bundle matching Python `Input` (subset: no RDKit pickle in Rust).
#[derive(Debug, Clone)]
pub struct Boltz2InferenceInput {
    pub structure: StructureV2Tables,
    pub msas: HashMap<i32, A3mMsa>,
    pub record: Boltz2Record,
    /// Template id → structure tables (`{record.id}_{template_id}.npz`).
    pub templates: Option<HashMap<String, StructureV2Tables>>,
    pub residue_constraints: Option<ResidueConstraints>,
    /// Optional extra CCD molecules (ligands, modified residues) from `extra_mols_dir` (`*.json`).
    pub extra_mols: Option<CcdMolProvider>,
}

/// Token/bond tensors from Boltz [`Boltz2Tokenizer`](TokenizeBoltz2Input) — mirrors Python
/// `Boltz2Tokenizer.tokenize` output for the **token** slice only (structure / MSA / record are not duplicated).
#[derive(Debug, Clone, PartialEq)]
pub struct Boltz2Tokenized {
    pub tokens: Vec<TokenData>,
    pub bonds: Vec<TokenBondV2>,
    pub template_tokens: Option<HashMap<String, Vec<TokenData>>>,
    pub template_bonds: Option<HashMap<String, Vec<TokenBondV2>>>,
}

impl From<&Boltz2Tokenized> for crate::featurizer::AffinityTokenized {
    fn from(t: &Boltz2Tokenized) -> Self {
        Self {
            tokens: t.tokens.clone(),
            bonds: t.bonds.clone(),
        }
    }
}

impl From<crate::featurizer::AffinityTokenized> for Boltz2Tokenized {
    fn from(a: crate::featurizer::AffinityTokenized) -> Self {
        Self {
            tokens: a.tokens,
            bonds: a.bonds,
            template_tokens: None,
            template_bonds: None,
        }
    }
}

/// Ligand `asym_id` to mark with `affinity_mask`, parsed from manifest `record.affinity` (Python `AffinityInfo.chain_id`).
#[must_use]
pub fn affinity_asym_id_from_record(record: &Boltz2Record) -> Option<i32> {
    let v = record.affinity.as_ref()?;
    match v {
        Value::Null => None,
        Value::Object(map) => {
            if let Some(n) = map.get("chain_id").and_then(Value::as_i64) {
                return Some(n as i32);
            }
            if let Some(n) = map.get("asym_id").and_then(Value::as_i64) {
                return Some(n as i32);
            }
            map.get("chain_id")
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<i32>().ok())
        }
        Value::Number(n) => n.as_i64().map(|i| i as i32),
        _ => None,
    }
}

/// Full Boltz2 tokenization: main structure (with optional affinity mask) + each template via `tokenize_structure` (Python `Boltz2Tokenizer.tokenize`).
#[must_use]
pub fn tokenize_boltz2_inference(input: &Boltz2InferenceInput) -> Boltz2Tokenized {
    let aff = affinity_asym_id_from_record(&input.record);
    let (tokens, bonds) = tokenize_structure(&input.structure, aff);

    let (template_tokens, template_bonds) = match &input.templates {
        Some(map) => {
            let mut tt = HashMap::with_capacity(map.len());
            let mut tb = HashMap::with_capacity(map.len());
            for (id, tmpl) in map {
                let (t, b) = tokenize_structure(tmpl, None);
                tt.insert(id.clone(), t);
                tb.insert(id.clone(), b);
            }
            (Some(tt), Some(tb))
        }
        None => (None, None),
    };

    Boltz2Tokenized {
        tokens,
        bonds,
        template_tokens,
        template_bonds,
    }
}

/// Mirrors `boltz.data.tokenize.boltz2.Boltz2Tokenizer` (stateless).
#[derive(Debug, Default, Clone, Copy)]
pub struct Boltz2Tokenizer;

/// Mirrors `boltz.data.tokenize.tokenizer.Tokenizer` + `Boltz2Tokenizer.tokenize`.
pub trait TokenizeBoltz2Input {
    fn tokenize(&self, input: &Boltz2InferenceInput) -> Boltz2Tokenized;
}

impl TokenizeBoltz2Input for Boltz2Tokenizer {
    fn tokenize(&self, input: &Boltz2InferenceInput) -> Boltz2Tokenized {
        tokenize_boltz2_inference(input)
    }
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
/// `template_dir` are set; optional residue constraints from `constraints_dir`; optional
/// `extra_mols_dir` with one `*.json` file per CCD code (see [`CcdMolProvider::load_all_json_in_dir`]).
/// When `affinity == true`, loads `{target_dir}/{id}/pre_affinity_{id}.npz` instead of
/// `{target_dir}/{id}.npz` (Boltz preprocess layout).
pub fn load_input(
    record: &Boltz2Record,
    target_dir: &Path,
    msa_dir: &Path,
    constraints_dir: Option<&Path>,
    template_dir: Option<&Path>,
    extra_mols_dir: Option<&Path>,
    affinity: bool,
) -> Result<Boltz2InferenceInput> {
    let extra_mols = match extra_mols_dir {
        Some(dir) => Some(
            CcdMolProvider::load_all_json_in_dir(dir)
                .with_context(|| format!("load_input: extra_mols_dir {}", dir.display()))?,
        ),
        None => None,
    };
    let structure_path = if affinity {
        target_dir
            .join(&record.id)
            .join(format!("pre_affinity_{}.npz", record.id))
    } else {
        target_dir.join(format!("{}.npz", record.id))
    };
    let structure = read_structure_v2_npz_path(&structure_path)
        .with_context(|| format!("StructureV2.load: {}", structure_path.display()))?;

    let mut msas = HashMap::new();
    for chain in &record.chains {
        if !msa_id_is_active(&chain.msa_id) {
            continue;
        }
        let fname = msa_id_for_path(&chain.msa_id)?;
        let msa_path = msa_dir.join(format!("{fname}.npz"));
        let msa = read_msa_npz_path(&msa_path)
            .with_context(|| format!("MSA.load: {}", msa_path.display()))?;
        msas.insert(chain.chain_id, msa);
    }

    let templates = match (template_dir, &record.templates) {
        (Some(td), Some(infos)) if !infos.is_empty() => {
            let mut map = HashMap::new();
            for t in infos {
                let path = td.join(format!("{}_{}.npz", record.id, t.name));
                let tables = read_structure_v2_npz_path(&path)
                    .with_context(|| format!("template StructureV2.load: {}", path.display()))?;
                map.insert(t.name.clone(), tables);
            }
            Some(map)
        }
        _ => None,
    };

    // Load residue constraints
    let residue_constraints_local = match constraints_dir {
        Some(cd) => {
            let path = cd.join(format!("{}.npz", record.id));
            match ResidueConstraints::load_from_npz(&path) {
                Ok(rc) if !rc.is_empty() => Some(rc),
                _ => None,
            }
        }
        None => None,
    };

    Ok(Boltz2InferenceInput {
        structure,
        msas,
        record: record.clone(),
        templates,
        residue_constraints: residue_constraints_local,
        extra_mols,
    })
}

/// Token-level features after `load_input`: `tokenize_structure` + `process_token_features`.
///
/// Aligns with Python `Boltz2Tokenizer.tokenize` → `Boltz2Featurizer.process` **token** slice only.
/// Uses [`affinity_asym_id_from_record`] like Python `tokenize_structure(..., record.affinity)`.
/// `msas` and template tensors are handled elsewhere (`process_msa_features`, template featurizer).
#[must_use]
pub fn token_features_from_inference_input(input: &Boltz2InferenceInput) -> TokenFeatureTensors {
    let aff = affinity_asym_id_from_record(&input.record);
    let (tokens, bonds) = tokenize_structure(&input.structure, aff);
    process_token_features(&tokens, &bonds, None)
}

/// Atom-level features after `load_input`: `tokenize_structure` + [`process_atom_features`](crate::featurizer::process_atom_features).
///
/// Uses [`StandardAminoAcidRefData`] for canonical residue chemistry (matches Boltz `load_canonicals`
/// for standard amino acids). When [`Boltz2InferenceInput::extra_mols`] is set, non-standard
/// residues are resolved via [`InferenceAtomRefProvider`] and CCD JSON graphs.
#[must_use]
pub fn atom_features_from_inference_input(input: &Boltz2InferenceInput) -> AtomFeatureTensors {
    let aff = affinity_asym_id_from_record(&input.record);
    let (tokens, _bonds) = tokenize_structure(&input.structure, aff);
    let standard = StandardAminoAcidRefData::new();
    let provider = InferenceAtomRefProvider {
        standard: &standard,
        extra_mols: input.extra_mols.as_ref(),
    };
    let config = AtomFeatureConfig::default();
    process_atom_features(
        &tokens,
        &input.structure,
        &inference_ensemble_features(),
        &provider,
        &config,
    )
}

/// MSA tensors after `load_input`: `tokenize_structure` + [`process_msa_features`](crate::featurizer::process_msa_features).
///
/// Uses deterministic RNG seed **42** (matches typical Boltz inference seeding in `inferencev2`).
/// Templates, constraints, and affinity paths are out of scope here.
#[must_use]
pub fn msa_features_from_inference_input(input: &Boltz2InferenceInput) -> MsaFeatureTensors {
    let aff = affinity_asym_id_from_record(&input.record);
    let (tokens, _bonds) = tokenize_structure(&input.structure, aff);
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

/// Template tensors: real [`process_template_features`](crate::featurizer::process_template_features)
/// when `record.templates`, loaded template `.npz`, and tokenized template loops are present and
/// `query_chain` / `template_chain` are set; otherwise padded dummy rows (same keys as Boltz).
#[must_use]
pub fn template_features_from_tokenized(
    input: &Boltz2InferenceInput,
    tokenized: &Boltz2Tokenized,
    max_tokens: usize,
    min_tdim: usize,
) -> crate::featurizer::DummyTemplateTensors {
    let min_t = min_tdim.max(1);
    let dummy = || load_dummy_templates_features(min_t, max_tokens);
    let record = match &input.record.templates {
        Some(t) if !t.is_empty() => t.as_slice(),
        _ => return pad_template_tdim(dummy(), min_t),
    };
    let (templates, tmpl_tok) = match (&input.templates, &tokenized.template_tokens) {
        (Some(ts), Some(tt)) if !ts.is_empty() && !tt.is_empty() => (ts, tt),
        _ => return pad_template_tdim(dummy(), min_t),
    };
    let alignments: Vec<TemplateAlignment> = record.iter().map(|t| t.into()).collect();
    if alignments.iter().all(|a| a.query_chain.is_empty()) {
        return pad_template_tdim(dummy(), min_t);
    }
    match process_template_features(
        &tokenized.tokens,
        &input.structure,
        templates,
        tmpl_tok,
        &alignments,
        max_tokens,
    ) {
        Ok(t) => {
            let need = t.template_restype.shape()[0].max(min_t);
            pad_template_tdim(t, need)
        }
        Err(_) => pad_template_tdim(dummy(), min_t),
    }
}

/// All featurizer tensors merged into one [`FeatureBatch`] for a single example.
///
/// Matches the Boltz featurizer output for inference:
/// - `process_token_features` (token-level)
/// - `process_msa_features` (MSA)
/// - `process_atom_features` (atom-level)
/// - `process_symmetry_features` (symmetry / all_coords)
/// - `process_residue_constraint_features` (constraints)
/// - template features (real or dummy)
///
/// Does **not** include `s_inputs` (computed inside the model from the embedder stack).
#[must_use]
pub fn trunk_smoke_feature_batch_from_inference_input(
    input: &Boltz2InferenceInput,
    template_dim: usize,
) -> FeatureBatch {
    let tokenized = tokenize_boltz2_inference(input);
    let n = tokenized.tokens.len();

    // Token features
    let tok = process_token_features(&tokenized.tokens, &tokenized.bonds, None);
    // MSA features
    let msa = msa_features_from_inference_input(input);
    // Atom features
    let atoms = atom_features_from_inference_input(input);
    // Symmetry features (all_coords, all_resolved_mask, crop_to_all_atom_map)
    let symm = process_symmetry_features(&input.structure, &tokenized.tokens);

    let mut batch = tok.to_feature_batch();
    batch.merge(msa.to_feature_batch());
    batch.merge(atoms.to_feature_batch());
    batch.merge(symm.to_feature_batch());

    // Residue constraint features (optional, empty tensors when None)
    let residue_constraint_features =
        crate::featurizer::process_residue_constraint_features(
            input.residue_constraints.as_ref(),
        );
    batch.merge(residue_constraint_features.into_feature_batch());

    // Template features (real or dummy)
    let tmpl = template_features_from_tokenized(input, &tokenized, n, template_dim);
    batch.merge(tmpl.into_feature_batch());

    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;

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

    #[test]
    fn affinity_asym_id_parses_chain_id() {
        let mut r: Boltz2Record = serde_json::from_str(
            r#"{"id":"x","structure":{},"chains":[],"affinity":{"chain_id":3}}"#,
        )
        .unwrap();
        assert_eq!(affinity_asym_id_from_record(&r), Some(3));
        r.affinity = Some(serde_json::json!({"asym_id": 7}));
        assert_eq!(affinity_asym_id_from_record(&r), Some(7));
        r.affinity = None;
        assert_eq!(affinity_asym_id_from_record(&r), None);
    }

    #[test]
    fn tokenize_boltz2_matches_structure_only_on_preprocess_fixture() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke");
        let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
        let record = &manifest.records[0];
        let input = load_input(record, &dir, &dir, None, None, None, false).expect("load_input");
        let out = tokenize_boltz2_inference(&input);
        let aff = affinity_asym_id_from_record(&input.record);
        let (t, b) = tokenize_structure(&input.structure, aff);
        assert_eq!(out.tokens, t);
        assert_eq!(out.bonds, b);
        assert!(out.template_tokens.is_none());
        assert!(out.template_bonds.is_none());
    }

    #[test]
    fn tokenize_boltz2_template_loop_matches_isolated_tokenize() {
        let s = structure_v2_single_ala();
        let mut templates = HashMap::new();
        templates.insert("tmpl1".to_string(), s.clone());
        let record: Boltz2Record = serde_json::from_str(
            r#"{"id":"demo","structure":{},"chains":[{"chain_id":0,"chain_name":"A","mol_type":0,"cluster_id":0,"msa_id":0,"num_residues":1,"valid":true}],"interfaces":[]}"#,
        )
        .unwrap();
        let input = Boltz2InferenceInput {
            structure: s.clone(),
            msas: HashMap::new(),
            record,
            templates: Some(templates),
            residue_constraints: None,
            extra_mols: None,
        };
        let out = TokenizeBoltz2Input::tokenize(&Boltz2Tokenizer, &input);
        let (main_t, main_b) = tokenize_structure(&s, None);
        let (tmpl_t, tmpl_b) = tokenize_structure(&s, None);
        assert_eq!(out.tokens, main_t);
        assert_eq!(out.bonds, main_b);
        assert_eq!(
            out.template_tokens.as_ref().unwrap().get("tmpl1").unwrap(),
            &tmpl_t
        );
        assert_eq!(
            out.template_bonds.as_ref().unwrap().get("tmpl1").unwrap(),
            &tmpl_b
        );
    }

    #[test]
    fn trunk_smoke_batch_includes_symmetry_keys() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke");
        let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
        let input =
            load_input(&manifest.records[0], &dir, &dir, None, None, None, false).expect("load_input");
        let batch = trunk_smoke_feature_batch_from_inference_input(&input, 1);
        // Symmetry keys should be present
        assert!(batch.tensors.contains_key("all_coords"), "missing all_coords");
        assert!(batch.tensors.contains_key("all_resolved_mask"), "missing all_resolved_mask");
        assert!(batch.tensors.contains_key("crop_to_all_atom_map"), "missing crop_to_all_atom_map");
    }

    #[test]
    fn load_input_extra_mols_dir_empty_ok() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke");
        let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
        let empty = std::env::temp_dir().join(format!("boltr_extra_mols_{}", std::process::id()));
        std::fs::create_dir_all(&empty).expect("mkdir");
        let input = load_input(
            &manifest.records[0],
            &dir,
            &dir,
            None,
            None,
            Some(&empty),
            false,
        )
        .expect("load_input");
        let _ = std::fs::remove_dir_all(&empty);
        assert!(input.extra_mols.as_ref().is_some_and(|m| m.is_empty()));
    }
}
