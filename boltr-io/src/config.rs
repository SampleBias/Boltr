//! Full Boltz YAML input schema matching Python `parse/yaml.py` and `parse/schema.py`.
//!
//! Supports all entity types (protein, dna, rna, ligand), constraints (bond, pocket, contact),
//! templates (cif/pdb), modifications, cyclic peptides, properties (affinity), and version.
//!
//! ## Python references
//!
//! - `boltz.data.parse.yaml.parse_yaml` — top-level YAML → `Target` conversion
//! - `boltz.data.parse.schema` — schema validation and entity resolution
//!
//! ## Usage
//!
//! ```rust
//! use boltr_io::config::BoltzInput;
//!
//! let yaml = r#"
//! sequences:
//!   - protein:
//!       id: A
//!       sequence: "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSQ"
//!       msa: ./msa.a3m
//!   - ligand:
//!       id: B
//!       ccd: HEM
//! constraints:
//!   - bond:
//!       atom1: [A, 1, N]
//!       atom2: [B, 1, C1]
//! templates:
//!   - cif: template.cif
//!     chain_id: A
//! "#;
//!
//! let input: BoltzInput = serde_yaml::from_str(yaml).unwrap();
//! assert_eq!(input.summary_chain_ids(), vec!["A", "B"]);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// ───────────────────────────────────────────────────────────────────────────────
// Root document
// ───────��───────────────────────────────────────────────────────────────────────

/// Root document for a Boltz job input file.
///
/// Full schema:
/// ```yaml
/// version: 1
/// sequences:
///   - ENTITY_TYPE:
///       id: CHAIN_ID or [CHAIN_ID, ...]
///       sequence: SEQUENCE       # protein, dna, rna
///       smiles: SMILES           # ligand (exclusive with ccd)
///       ccd: CCD_CODE            # ligand (exclusive with smiles)
///       msa: MSA_PATH            # protein only
///       modifications:
///         - position: RES_IDX
///           ccd: CCD_CODE
///       cyclic: false
/// constraints:
///   - bond:
///       atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
///       atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
///   - pocket:
///       binder: CHAIN_ID
///       contacts: [[CHAIN_ID, RES_IDX], ...]
///       max_distance: DIST
///       force: false
///   - contact:
///       token1: [CHAIN_ID, RES_IDX]
///       token2: [CHAIN_ID, RES_IDX]
///       max_distance: DIST
///       force: false
/// templates:
///   - cif: PATH
///     chain_id: CHAIN_ID or [CHAIN_ID, ...]
///     template_id: TEMPLATE_ID or [TEMPLATE_ID, ...]
///     force: false
///     threshold: DIST
///   - pdb: PATH
///     chain_id: ...
/// properties:
///   - affinity:
///       binder: CHAIN_ID
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoltzInput {
    /// Schema version (default: 1).
    #[serde(default)]
    pub version: Option<u32>,
    /// Entity entries (proteins, dna, rna, ligands).
    pub sequences: Vec<SequenceEntry>,
    /// Optional constraints (bonds, pockets, contacts).
    #[serde(default)]
    pub constraints: Option<Vec<ConstraintEntry>>,
    /// Optional templates (cif/pdb files).
    #[serde(default)]
    pub templates: Option<Vec<TemplateEntry>>,
    /// Optional properties (affinity prediction).
    #[serde(default)]
    pub properties: Option<Vec<PropertyEntry>>,
}

// ───────────────────────────────────────────────────────────────────────────────
// Sequence entries
// ───────────────────────────────────────────────────────────────────────────────

/// One chain / molecule entry under `sequences:`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SequenceEntry {
    Protein {
        protein: PolymerEntity,
    },
    Dna {
        dna: PolymerEntity,
    },
    Rna {
        rna: PolymerEntity,
    },
    Ligand {
        ligand: LigandEntity,
    },
}

// ───────────────────────────────────────────────────────────────────────────────
// Chain ID spec (single string or list of strings)
// ───────────────────────────────────────────────────────────────────────────────

/// Chain ID specification: either a single string `"A"` or a list `["A", "B"]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChainIdSpec {
    Single(String),
    Many(Vec<String>),
}

impl ChainIdSpec {
    /// Collect all chain IDs as a vec.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            ChainIdSpec::Single(s) => vec![s.clone()],
            ChainIdSpec::Many(v) => v.clone(),
        }
    }

    /// Number of chains.
    pub fn len(&self) -> usize {
        match self {
            ChainIdSpec::Single(_) => 1,
            ChainIdSpec::Many(v) => v.len(),
        }
    }

    /// Whether there are zero chains.
    pub fn is_empty(&self) -> bool {
        match self {
            ChainIdSpec::Single(_) => false,
            ChainIdSpec::Many(v) => v.is_empty(),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Polymer entity (protein/dna/rna)
// ───────────────────────────────────────────────────────────────────────────────

/// A polymer entity (protein, dna, or rna) under `sequences:`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolymerEntity {
    /// Chain ID(s) for this entity.
    pub id: ChainIdSpec,
    /// Residue sequence string.
    pub sequence: String,
    /// Optional path to precomputed MSA (.a3m or .csv).
    #[serde(default)]
    pub msa: Option<String>,
    /// Optional cyclic peptide flag.
    #[serde(default)]
    pub cyclic: Option<bool>,
    /// Optional residue modifications.
    #[serde(default)]
    pub modifications: Option<Vec<ModificationEntry>>,
}

// ───────────────────────────────────────────────────────────────────────────────
// Ligand entity
// ───────────────────────────────────────────────────────────────────────────────

/// A ligand (non-polymer) entity under `sequences:`.
///
/// Either `smiles` or `ccd` must be provided (mutually exclusive).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LigandEntity {
    /// Chain ID(s) for this ligand.
    pub id: ChainIdSpec,
    /// SMILES string for the ligand (exclusive with `ccd`).
    #[serde(default)]
    pub smiles: Option<String>,
    /// CCD code for the ligand (exclusive with `smiles`).
    #[serde(default)]
    pub ccd: Option<LigandCcdCode>,
}

impl LigandEntity {
    /// Whether this ligand uses a CCD code.
    pub fn is_ccd(&self) -> bool {
        self.ccd.is_some()
    }

    /// Whether this ligand uses a SMILES string.
    pub fn is_smiles(&self) -> bool {
        self.smiles.is_some()
    }

    /// Get the ligand type for dispatching.
    pub fn ligand_type(&self) -> LigandType {
        match (&self.smiles, &self.ccd) {
            (Some(_), None) => LigandType::Smiles,
            (None, Some(_)) => LigandType::Ccd,
            _ => LigandType::Unspecified,
        }
    }
}

/// How the ligand is specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LigandType {
    /// SMILES string.
    Smiles,
    /// CCD code from the Chemical Component Dictionary.
    Ccd,
    /// Neither specified (invalid YAML).
    Unspecified,
}

impl fmt::Display for LigandType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LigandType::Smiles => write!(f, "SMILES"),
            LigandType::Ccd => write!(f, "CCD"),
            LigandType::Unspecified => write!(f, "unspecified"),
        }
    }
}

/// CCD code for a ligand — can be a string or a list of strings (for multi-code ligands).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LigandCcdCode {
    /// Single CCD code (e.g., `"HEM"`).
    Single(String),
    /// Multiple CCD codes (rare, e.g., `["HEM", "HEC"]`).
    Many(Vec<String>),
}

impl LigandCcdCode {
    /// Get the primary CCD code.
    pub fn primary(&self) -> &str {
        match self {
            LigandCcdCode::Single(s) => s,
            LigandCcdCode::Many(v) => v.first().map(|s| s.as_str()).unwrap_or(""),
        }
    }

    /// Collect all CCD codes.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            LigandCcdCode::Single(s) => vec![s.clone()],
            LigandCcdCode::Many(v) => v.clone(),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Modifications
// ───────────────────────────────────────────────────────────────────────────────

/// A residue modification in a polymer entity.
///
/// Python reference: `schema.py` Modification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModificationEntry {
    /// Residue index (1-based).
    pub position: i32,
    /// CCD code of the modified residue.
    pub ccd: String,
}

// ───────────────────────────────────────────────────────────────────────────────
// Constraints
// ───────��───────────────────────────────────────────────────────────────────────

/// A constraint entry under `constraints:`.
///
/// Python reference: `schema.py` BondConstraint, PocketConstraint, ContactConstraint
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[serde(untagged)]
pub enum ConstraintEntry {
    /// Covalent bond between two atoms.
    #[serde(rename = "bond")]
    Bond(BondConstraint),
    /// Pocket binding site specification.
    #[serde(rename = "pocket")]
    Pocket(PocketConstraint),
    /// Contact between two residues/atoms.
    #[serde(rename = "contact")]
    Contact(ContactConstraint),
}

/// Covalent bond constraint between two atoms.
///
/// ```yaml
/// - bond:
///     atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
///     atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BondConstraint {
    /// First atom: [chain_id, residue_index, atom_name]
    pub bond: BondConstraintInner,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BondConstraintInner {
    /// First atom: [chain_id, residue_index, atom_name]
    pub atom1: ConstraintAtomRef,
    /// Second atom: [chain_id, residue_index, atom_name]
    pub atom2: ConstraintAtomRef,
}

/// Atom reference in a constraint: `[CHAIN_ID, RES_IDX, ATOM_NAME]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ConstraintAtomRef {
    /// As a list: [chain_id, residue_index, atom_name]
    List(Vec<serde_yaml::Value>),
}

/// Pocket constraint specifying binding site.
///
/// ```yaml
/// - pocket:
///     binder: CHAIN_ID
///     contacts: [[CHAIN_ID, RES_IDX], ...]
///     max_distance: DIST
///     force: false
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PocketConstraint {
    /// Chain ID of the binder molecule.
    pub pocket: PocketConstraintInner,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PocketConstraintInner {
    /// Chain ID of the binder.
    pub binder: String,
    /// Contact residues/atoms: [[chain_id, res_idx/atom_name], ...]
    pub contacts: Vec<PocketContactRef>,
    /// Maximum distance in Ångströms (4–20, default 6).
    #[serde(default = "default_pocket_max_distance")]
    pub max_distance: f32,
    /// Whether to enforce with a potential.
    #[serde(default)]
    pub force: bool,
}

fn default_pocket_max_distance() -> f32 {
    6.0
}

/// A pocket contact reference: `[CHAIN_ID, RES_IDX]` or `[CHAIN_ID, ATOM_NAME]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PocketContactRef {
    /// As a list: [chain_id, residue_index] or [chain_id, atom_name]
    List(Vec<serde_yaml::Value>),
}

/// Contact constraint between two residues/atoms.
///
/// ```yaml
/// - contact:
///     token1: [CHAIN_ID, RES_IDX]
///     token2: [CHAIN_ID, RES_IDX]
///     max_distance: DIST
///     force: false
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContactConstraint {
    /// Contact specification.
    pub contact: ContactConstraintInner,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContactConstraintInner {
    /// First token: [chain_id, residue_index]
    pub token1: ConstraintTokenRef,
    /// Second token: [chain_id, residue_index]
    pub token2: ConstraintTokenRef,
    /// Maximum distance in Ångströms (4–20, default 6).
    #[serde(default = "default_contact_max_distance")]
    pub max_distance: f32,
    /// Whether to enforce with a potential.
    #[serde(default)]
    pub force: bool,
}

fn default_contact_max_distance() -> f32 {
    6.0
}

/// Token reference in a contact constraint: `[CHAIN_ID, RES_IDX/ATOM_NAME]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ConstraintTokenRef {
    /// As a list: [chain_id, residue_index_or_atom_name]
    List(Vec<serde_yaml::Value>),
}

// ───────────────────────────────────────────────────────────────────────────────
// Templates
// ───────────────────────────────────────────────────────────────────────────────

/// A template entry under `templates:`.
///
/// Either `cif` or `pdb` must be provided.
/// Python reference: `schema.py` Template
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TemplateEntry {
    /// Path to mmCIF template file.
    #[serde(default)]
    pub cif: Option<String>,
    /// Path to PDB template file.
    #[serde(default)]
    pub pdb: Option<String>,
    /// Chain ID(s) to template (optional, auto-detected if absent).
    #[serde(default)]
    pub chain_id: Option<TemplateChainId>,
    /// Template chain ID(s) for explicit mapping (optional).
    #[serde(default)]
    pub template_id: Option<TemplateChainId>,
    /// Whether to enforce with a potential.
    #[serde(default)]
    pub force: bool,
    /// Distance threshold in Ångströms (required when force=true).
    #[serde(default)]
    pub threshold: Option<f32>,
}

impl TemplateEntry {
    /// Get the template path (either cif or pdb).
    pub fn path(&self) -> Option<&str> {
        self.cif.as_deref().or(self.pdb.as_deref())
    }

    /// Whether this is a CIF template.
    pub fn is_cif(&self) -> bool {
        self.cdf.is_some()
    }

    /// Whether this is a PDB template.
    pub fn is_pdb(&self) -> bool {
        self.pdb.is_some()
    }
}

/// Template chain ID spec (single or list).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TemplateChainId {
    /// Single chain ID.
    Single(String),
    /// Multiple chain IDs.
    Many(Vec<String>),
}

impl TemplateChainId {
    /// Collect as vec.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            TemplateChainId::Single(s) => vec![s.clone()],
            TemplateChainId::Many(v) => v.clone(),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Properties
// ───────────────────────────────────────────────────────────────────────────────

/// A property entry under `properties:`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum PropertyEntry {
    /// Affinity prediction.
    Affinity {
        affinity: AffinityProperty,
    },
}

/// Affinity prediction specification.
///
/// ```yaml
/// properties:
///   - affinity:
///       binder: CHAIN_ID
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AffinityProperty {
    /// Chain ID of the binder molecule (must be a ligand, max ~128 atoms).
    pub binder: String,
}

// ───────────────────────────────────────────────────────────────────────────────
// Helper methods
// ───────────────────────────────────────────────────────────────────────────────

impl BoltzInput {
    /// All chain IDs across all entities, preserving order.
    pub fn summary_chain_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        for entry in &self.sequences {
            match entry {
                SequenceEntry::Protein { protein } => ids.extend(protein.id.to_vec()),
                SequenceEntry::Dna { dna } => ids.extend(dna.id.to_vec()),
                SequenceEntry::Rna { rna } => ids.extend(rna.id.to_vec()),
                SequenceEntry::Ligand { ligand } => ids.extend(ligand.id.to_vec()),
            }
        }
        ids
    }

    /// Protein chains that need an MSA path or server fetch when `msa` is absent.
    pub fn protein_sequences_for_msa(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for entry in &self.sequences {
            if let SequenceEntry::Protein { protein } = entry {
                if protein.msa.is_some() {
                    continue;
                }
                for id in protein.id.to_vec() {
                    out.push((id, protein.sequence.clone()));
                }
            }
        }
        out
    }

    /// Protein chains with explicit MSA paths.
    pub fn protein_msa_paths(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for entry in &self.sequences {
            if let SequenceEntry::Protein { protein } = entry {
                if let Some(ref msa) = protein.msa {
                    for id in protein.id.to_vec() {
                        out.push((id, msa.clone()));
                    }
                }
            }
        }
        out
    }

    /// All protein entities.
    pub fn proteins(&self) -> Vec<&PolymerEntity> {
        self.sequences
            .iter()
            .filter_map(|e| match e {
                SequenceEntry::Protein { protein } => Some(protein),
                _ => None,
            })
            .collect()
    }

    /// All DNA entities.
    pub fn dnas(&self) -> Vec<&PolymerEntity> {
        self.sequences
            .iter()
            .filter_map(|e| match e {
                SequenceEntry::Dna { dna } => Some(dna),
                _ => None,
            })
            .collect()
    }

    /// All RNA entities.
    pub fn rnas(&self) -> Vec<&PolymerEntity> {
        self.sequences
            .iter()
            .filter_map(|e| match e {
                SequenceEntry::Rna { rna } => Some(rna),
                _ => None,
            })
            .collect()
    }

    /// All ligand entities.
    pub fn ligands(&self) -> Vec<&LigandEntity> {
        self.sequences
            .iter()
            .filter_map(|e => match e {
                SequenceEntry::Ligand { ligand } => Some(ligand),
                _ => None,
            })
            .collect()
    }

    /// Affinity binder chain ID, if properties include affinity.
    pub fn affinity_binder(&self) -> Option<String> {
        self.properties.as_ref().and_then(|props| {
            props.iter().find_map(|p| match p {
                PropertyEntry::Affinity { affinity } => Some(affinity.binder.clone()),
            })
        })
    }

    /// Bond constraints, if any.
    pub fn bond_constraints(&self) -> Vec<&BondConstraint> {
        self.constraints
            .as_ref()
            .map(|c| {
                c.iter()
                    .filter_map(|e| match e {
                        ConstraintEntry::Bond(b) => Some(b),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Pocket constraints, if any.
    pub fn pocket_constraints(&self) -> Vec<&PocketConstraint> {
        self.constraints
            .as_ref()
            .map(|c| {
                c.iter()
                    .filter_map(|e| match e {
                        ConstraintEntry::Pocket(p) => Some(p),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Contact constraints, if any.
    pub fn contact_constraints(&self) -> Vec<&ContactConstraint> {
        self.constraints
            .as_ref()
            .map(|c| {
                c.iter()
                    .filter_map(|e| match e {
                        ConstraintEntry::Contact(c) => Some(c),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Template entries, if any.
    pub fn template_entries(&self) -> &[TemplateEntry] {
        self.templates.as_ref().map(|t| t.as_slice()).unwrap_or(&[])
    }

    /// Whether any entity has modifications.
    pub fn has_modifications(&self) -> bool {
        self.sequences.iter().any(|e| match e {
            SequenceEntry::Protein { protein } => protein.modifications.is_some(),
            SequenceEntry::Dna { dna } => dna.modifications.is_some(),
            SequenceEntry::Rna { rna } => rna.modifications.is_some(),
            _ => false,
        })
    }

    /// Whether any protein entity is cyclic.
    pub fn has_cyclic(&self) -> bool {
        self.sequences.iter().any(|e| match e {
            SequenceEntry::Protein { protein } => protein.cyclic.unwrap_or(false),
            _ => false,
        })
    }
}

/// Backward-compatible alias.
pub type BoltzConfig = BoltzInput;
