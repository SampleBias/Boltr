//! Minimal Rust-only preprocess bundle for **canonical protein-only** inputs (no ligands/DNA/RNA,
//! no templates, no YAML `constraints:`). Produces `manifest.json` + `{id}.npz` + `{msa_id}.npz`
//! beside the output directory so [`crate::inference_dataset::load_input`] can run.
//!
//! Coordinates are **placeholder** (extended along +X); use Boltz Python preprocess for
//! biologically realistic starting coordinates.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::boltz_const::{chain_type_id, prot_letter_to_token_id, token_name, unk_token_id};
use crate::config::{BoltzInput, SequenceEntry};
use crate::inference_dataset::{Boltz2ChainInfo, Boltz2Manifest, Boltz2Record, StructureInfo};
use crate::msa_npz::write_msa_npz_compressed;
use crate::parser::parse_input_path;
use crate::ref_atoms::ref_atom_names;
use crate::structure_v2::{AtomV2Row, ChainRow, EnsembleRow, ResidueRow, StructureV2Tables};
use crate::structure_v2_npz::write_structure_v2_npz_compressed;

/// Errors when the input is outside the native-supported profile.
#[derive(Debug)]
pub enum NativePreprocessError {
    LigandOrNucleicAcid,
    Modifications,
    TemplatesOrConstraintsInYaml,
    NoProteinChains,
}

impl std::fmt::Display for NativePreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LigandOrNucleicAcid => write!(f, "ligand/dna/rna entries are not supported"),
            Self::Modifications => write!(f, "protein `modifications:` are not supported"),
            Self::TemplatesOrConstraintsInYaml => {
                write!(f, "`templates:` or `constraints:` blocks are not supported in native mode")
            }
            Self::NoProteinChains => write!(f, "no protein sequences found"),
        }
    }
}

impl std::error::Error for NativePreprocessError {}

/// Validate YAML for native preprocess.
pub fn validate_native_eligible(input: &BoltzInput) -> std::result::Result<(), NativePreprocessError> {
    if input.templates.is_some() {
        return Err(NativePreprocessError::TemplatesOrConstraintsInYaml);
    }
    if input.constraints.is_some() {
        return Err(NativePreprocessError::TemplatesOrConstraintsInYaml);
    }
    let mut any_protein = false;
    for entry in &input.sequences {
        match entry {
            SequenceEntry::Protein { protein } => {
                any_protein = true;
                if protein.modifications.is_some() {
                    return Err(NativePreprocessError::Modifications);
                }
            }
            SequenceEntry::Dna { .. } | SequenceEntry::Rna { .. } | SequenceEntry::Ligand { .. } => {
                return Err(NativePreprocessError::LigandOrNucleicAcid);
            }
        }
    }
    if !any_protein {
        return Err(NativePreprocessError::NoProteinChains);
    }
    Ok(())
}

/// Write `manifest.json`, `{record_id}.npz`, and per-entity `msa_id` MSA `.npz` into `out_dir`.
///
/// When `fetched_msa_dir` is set (e.g. `.../msa` after `boltr predict` wrote `{chain}.a3m` there),
/// protein chains with no `msa:` in YAML load `fetched_msa_dir/{chain_id}.a3m`.
pub fn write_native_preprocess_bundle(
    yaml_path: &Path,
    out_dir: &Path,
    record_id: Option<&str>,
    max_msa_seqs: Option<usize>,
    fetched_msa_dir: Option<&Path>,
) -> Result<()> {
    let input = parse_input_path(yaml_path)?;
    validate_native_eligible(&input).map_err(|e| anyhow::anyhow!(e))?;

    let rid = record_id
        .map(std::string::ToString::to_string)
        .or_else(|| {
            yaml_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(std::string::ToString::to_string)
        })
        .context("record id")?;

    fs::create_dir_all(out_dir)?;

    let yaml_parent = yaml_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let structure = build_placeholder_protein_structure_v2(&input)?;
    write_structure_v2_npz_compressed(&out_dir.join(format!("{rid}.npz")), &structure)?;

    let mut chains_meta: Vec<Boltz2ChainInfo> = Vec::new();
    let mut chain_id = 0i32;
    let mut msa_group = 0i32;

    for entry in &input.sequences {
        let SequenceEntry::Protein { protein } = entry else {
            continue;
        };
        let chain_names = protein.id.to_vec();
        let n_ch = chain_names.len().max(1);
        let seq: String = protein.sequence.chars().filter(|c| !c.is_whitespace()).collect();
        let nres = seq.len() as i32;

        let msa = match protein.msa.as_deref() {
            Some(s) if s.eq_ignore_ascii_case("empty") => empty_single_sequence_msa(&seq)?,
            None => {
                if let Some(dir) = fetched_msa_dir {
                    let first = chain_names
                        .first()
                        .map(String::as_str)
                        .unwrap_or("A");
                    let p = dir.join(format!("{first}.a3m"));
                    if p.is_file() {
                        crate::parse_a3m_path(&p, max_msa_seqs)
                            .with_context(|| format!("parse MSA {}", p.display()))?
                    } else {
                        empty_single_sequence_msa(&seq)?
                    }
                } else {
                    empty_single_sequence_msa(&seq)?
                }
            }
            Some(path) => {
                let p = yaml_parent.join(path);
                crate::parse_a3m_path(&p, max_msa_seqs)
                    .with_context(|| format!("parse MSA {}", p.display()))?
            }
        };
        write_msa_npz_compressed(&out_dir.join(format!("{msa_group}.npz")), &msa)?;

        for sub in 0..n_ch {
            let cname = chain_names
                .get(sub)
                .cloned()
                .unwrap_or_else(|| "A".to_string());
            chains_meta.push(Boltz2ChainInfo {
                chain_id,
                chain_name: cname,
                mol_type: 0,
                cluster_id: serde_json::json!(0),
                msa_id: serde_json::json!(msa_group),
                num_residues: nres,
                valid: true,
                entity_id: None,
            });
            chain_id += 1;
        }
        msa_group += 1;
    }

    let record = Boltz2Record {
        id: rid.clone(),
        structure: StructureInfo::default(),
        chains: chains_meta,
        interfaces: vec![],
        inference_options: None,
        templates: None,
        md: None,
        affinity: None,
    };

    let manifest = Boltz2Manifest {
        records: vec![record],
    };
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(out_dir.join("manifest.json"), json)?;

    Ok(())
}

fn empty_single_sequence_msa(sequence: &str) -> Result<crate::A3mMsa> {
    let seq: String = sequence.chars().filter(|c| !c.is_whitespace()).collect();
    crate::parse_a3m_str(&format!(">query\n{seq}\n"), Some(1))
}

/// Build extended placeholder coordinates for all protein chains (canonical AA only).
fn build_placeholder_protein_structure_v2(input: &BoltzInput) -> Result<StructureV2Tables> {
    let p = chain_type_id("PROTEIN").context("PROTEIN chain type")? as i8;
    let unk = unk_token_id("PROTEIN").context("unk protein")?;

    let mut atoms: Vec<AtomV2Row> = Vec::new();
    let mut residues: Vec<ResidueRow> = Vec::new();
    let mut chains_out: Vec<ChainRow> = Vec::new();
    let mut chain_mask: Vec<bool> = Vec::new();

    let mut global_res = 0i32;
    let mut asym = 0i32;

    for entry in &input.sequences {
        let SequenceEntry::Protein { protein } = entry else {
            continue;
        };
        let seq: String = protein.sequence.chars().filter(|c| !c.is_whitespace()).collect();
        let chain_names = protein.id.to_vec();
        let n_sub = chain_names.len().max(1);
        for sub in 0..n_sub {
            let cname = chain_names
                .get(sub)
                .cloned()
                .unwrap_or_else(|| "A".to_string());
            let chain_atom_start = atoms.len() as i32;
            let res_start = global_res;
            let mut res_local = 0i32;
            for ch in seq.chars() {
                let tid = prot_letter_to_token_id(ch).unwrap_or(unk);
                let res_name = token_name(tid).unwrap_or("UNK");
                let ref_key = match res_name {
                    "UNK" => "UNK",
                    other => other,
                };
                let atom_names = ref_atom_names(ref_key).unwrap_or(ref_atom_names("UNK").unwrap());
                let n_atom = atom_names.len() as i32;
                let center = 1_i32;
                let disto = (n_atom - 1).max(1);
                let base_x = asym as f32 * 500.0 + res_local as f32 * 3.8;
                let atom_base = atoms.len() as i32;
                for (i, aname) in atom_names.iter().enumerate() {
                    atoms.push(AtomV2Row {
                        name: (*aname).to_string(),
                        coords: [base_x + i as f32 * 0.5, res_local as f32, 0.0],
                        is_present: true,
                        bfactor: 0.0,
                        plddt: 0.0,
                    });
                }
                residues.push(ResidueRow {
                    name: res_name.to_string(),
                    res_type: tid as i8,
                    res_idx: global_res,
                    atom_idx: atom_base,
                    atom_num: n_atom,
                    atom_center: center,
                    atom_disto: disto,
                    is_standard: true,
                    is_present: true,
                });
                global_res += 1;
                res_local += 1;
            }
            let chain_atom_num = atoms.len() as i32 - chain_atom_start;
            chains_out.push(ChainRow {
                name: cname,
                mol_type: p,
                sym_id: asym,
                asym_id: asym,
                entity_id: asym,
                atom_idx: chain_atom_start,
                atom_num: chain_atom_num,
                res_idx: res_start,
                res_num: res_local,
                cyclic_period: 0,
            });
            chain_mask.push(true);
            asym += 1;
        }
    }

    if atoms.is_empty() {
        bail!("no protein atoms built");
    }

    let n_coords = atoms.len();
    let coords: Vec<[f32; 3]> = atoms.iter().map(|a| a.coords).collect();
    let ensemble = vec![
        EnsembleRow {
            atom_coord_idx: 0,
            atom_num: n_coords as i32,
        };
        3
    ];

    Ok(StructureV2Tables {
        atoms,
        residues,
        chains: chains_out,
        chain_mask,
        coords,
        ensemble,
        ensemble_atom_coord_idx: 0,
        bonds: vec![],
    })
}
