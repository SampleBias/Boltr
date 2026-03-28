//! Boltz `mol.py` — CCD / molecule loading for Boltz2 inference.
//!
//! In Boltz2, molecules are stored as individual `.pkl` files in the `mols/` cache directory.
//! Each file contains an RDKit `Mol` object pickled with atom properties (name, element,
//! charge, leaving_atom, conformer coordinates, bonds, chirality, etc.).
//!
//! ## Design
//!
//! Since Rust does not have RDKit, we define [`CcdMolData`] as a lightweight representation
//! of the molecule data needed for schema parsing and featurization. The data can be loaded
//! from either:
//!
//! 1. **Pickle files** (`mols/{ccd_code}.pkl`) — requires Python-preprocessed data.
//! 2. **Pre-extracted JSON/binpk** files — a future offline extraction path.
//!
//! For Boltz2 inference, the preprocess step already extracts structures to `.npz`, so
//! the molecule data is primarily needed during the **schema parse** phase
//! (YAML → `StructureV2Tables`) to resolve ligand atoms/bonds/conformers.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::boltz_const::CANONICAL_TOKENS;

/// A single atom in a CCD molecule (heavy atoms only, hydrogens stripped).
#[derive(Clone, Debug, PartialEq)]
pub struct CcdAtom {
    /// PDB atom name (e.g., `"C1"`, `"O2"`, `"N1"`).
    pub name: String,
    /// Atomic number (e.g., 6 for Carbon, 7 for Nitrogen).
    pub atomic_num: i32,
    /// Formal charge.
    pub formal_charge: i32,
    /// Whether this atom is a leaving atom (for modified residue handling).
    pub leaving_atom: bool,
    /// Conformer coordinates (ideal or computed) in Ångströms.
    pub conformer_coords: [f32; 3],
    /// Chirality tag string (e.g., `"CHI_UNSPECIFIED"`, `"CHI_TETRAHEDRAL_CW"`).
    pub chirality_tag: String,
}

/// A bond between two atoms in a CCD molecule.
#[derive(Clone, Debug, PartialEq)]
pub struct CcdBond {
    /// Index of atom 1 (within the atom list).
    pub atom_idx_1: usize,
    /// Index of atom 2 (within the atom list).
    pub atom_idx_2: usize,
    /// Bond type string (e.g., `"SINGLE"`, `"DOUBLE"`, `"AROMATIC"`).
    pub bond_type: String,
}

/// A fully resolved CCD molecule, equivalent to an RDKit `Mol` with one conformer,
/// hydrogens removed (heavy atoms only), and atom properties extracted.
///
/// This is the Rust representation of the data that Python stores in `mols/{code}.pkl`.
#[derive(Clone, Debug, PartialEq)]
pub struct CcdMolData {
    /// CCD code (e.g., `"SAH"`, `"ATP"`, `"LIG1"`).
    pub code: String,
    /// Heavy atoms in canonical order (hydrogen-stripped).
    pub atoms: Vec<CcdAtom>,
    /// Bonds between heavy atoms.
    pub bonds: Vec<CcdBond>,
    /// Number of heavy atoms (cached from `atoms.len()`).
    pub num_heavy_atoms: usize,
}

impl CcdMolData {
    /// Build from raw atom/bond data.
    pub fn new(code: String, atoms: Vec<CcdAtom>, bonds: Vec<CcdBond>) -> Self {
        let num_heavy_atoms = atoms.len();
        Self {
            code,
            atoms,
            bonds,
            num_heavy_atoms,
        }
    }

    /// Look up an atom by name.
    pub fn atom_by_name(&self, name: &str) -> Option<(usize, &CcdAtom)> {
        self.atoms
            .iter()
            .enumerate()
            .find(|(_, a)| a.name == name)
    }

    /// Heavy atom names in order.
    pub fn atom_names(&self) -> Vec<&str> {
        self.atoms.iter().map(|a| a.name.as_str()).collect()
    }

    /// Check if this is a single-heavy-atom molecule.
    pub fn is_single_atom(&self) -> bool {
        self.num_heavy_atoms == 1
    }
}

/// A provider of CCD molecule data, loaded from the `mols/` directory.
///
/// Wraps a `HashMap<String, CcdMolData>` with lazy loading support.
/// Mirrors Python `load_molecules` / `load_canonicals` / `get_mol`.
#[derive(Clone, Debug, Default)]
pub struct CcdMolProvider {
    /// Loaded molecules, keyed by CCD code.
    mols: HashMap<String, CcdMolData>,
    /// Optional root directory for lazy loading.
    mol_dir: Option<String>,
}

impl CcdMolProvider {
    /// Create an empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a provider backed by a `mols/` directory for lazy loading.
    pub fn with_mol_dir(mol_dir: impl Into<String>) -> Self {
        Self {
            mols: HashMap::new(),
            mol_dir: Some(mol_dir.into()),
        }
    }

    /// Insert a pre-built molecule.
    pub fn insert(&mut self, mol: CcdMolData) {
        self.mols.insert(mol.code.clone(), mol);
    }

    /// Look up a molecule already loaded in memory (no disk I/O).
    #[must_use]
    pub fn get_loaded(&self, ccd_code: &str) -> Option<&CcdMolData> {
        self.mols.get(ccd_code)
    }

    /// Get a molecule by CCD code, loading from `mol_dir` if needed.
    ///
    /// Returns `None` if the molecule is not found in memory or on disk.
    /// For pickle files, this will fail gracefully since we cannot deserialize
    /// RDKit pickles in Rust — the molecule must be pre-loaded or pre-extracted.
    pub fn get(&mut self, ccd_code: &str) -> Option<&CcdMolData> {
        if self.mols.contains_key(ccd_code) {
            return self.mols.get(ccd_code);
        }

        // Try lazy loading from mol_dir (only works for pre-extracted formats)
        if let Some(ref dir) = self.mol_dir {
            let path = Path::new(dir).join(format!("{ccd_code}.json"));
            if path.exists() {
                if let Ok(mol) = load_ccd_json(&path) {
                    self.mols.insert(ccd_code.to_string(), mol);
                    return self.mols.get(ccd_code);
                }
            }
        }

        None
    }

    /// Check if a molecule is available (in memory or on disk).
    pub fn contains(&self, ccd_code: &str) -> bool {
        if self.mols.contains_key(ccd_code) {
            return true;
        }
        if let Some(ref dir) = self.mol_dir {
            Path::new(dir).join(format!("{ccd_code}.json")).exists()
        } else {
            false
        }
    }

    /// Load all canonical molecules (20 standard amino acids + UNK).
    ///
    /// In Boltz2, canonical residues are loaded via `load_canonicals(mol_dir)`.
    /// In Rust, we rely on the built-in [`StandardAminoAcidRefData`](crate::featurizer::StandardAminoAcidRefData)
    /// for canonical residues, so this is mainly for non-standard CCD codes.
    pub fn load_canonicals_from_dir(mol_dir: &Path) -> Result<Self> {
        let mut provider = Self::with_mol_dir(mol_dir.to_string_lossy().into_owned());
        for code in &CANONICAL_TOKENS {
            let json_path = mol_dir.join(format!("{code}.json"));
            if json_path.exists() {
                if let Ok(mol) = load_ccd_json(&json_path) {
                    provider.insert(mol);
                }
            }
        }
        Ok(provider)
    }

    /// Load every `*.json` file in `dir` (Boltz `extra_mols` cache as pre-extracted JSON).
    ///
    /// Molecule codes come from each file's `"code"` field (see [`load_ccd_json`]). This mirrors
    /// Python loading a dict of extra CCD mols without RDKit pickles.
    pub fn load_all_json_in_dir(dir: &Path) -> Result<Self> {
        let mut provider = Self::new();
        for ent in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
            let ent = ent?;
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let mol = load_ccd_json(&path)
                .with_context(|| format!("load extra mol JSON {}", path.display()))?;
            provider.insert(mol);
        }
        Ok(provider)
    }

    /// Load specific molecules by CCD codes from the `mols/` directory.
    pub fn load_molecules(mol_dir: &Path, codes: &[&str]) -> Result<Self> {
        let mut provider = Self::new();
        for code in codes {
            let json_path = mol_dir.join(format!("{code}.json"));
            if json_path.exists() {
                let mol = load_ccd_json(&json_path)
                    .with_context(|| format!("load CCD molecule {code} from {}", json_path.display()))?;
                provider.insert(mol);
            } else {
                bail!(
                    "CCD component {code} not found at {} (tried .json)",
                    json_path.display()
                );
            }
        }
        Ok(provider)
    }

    /// Number of loaded molecules.
    pub fn len(&self) -> usize {
        self.mols.len()
    }

    /// Whether any molecules are loaded.
    pub fn is_empty(&self) -> bool {
        self.mols.is_empty()
    }

    /// Iterate over all loaded molecules.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &CcdMolData)> {
        self.mols.iter()
    }
}

/// Load a CCD molecule from a JSON file (pre-extracted from RDKit pickle).
///
/// The JSON format is:
/// ```json
/// {
///   "code": "SAH",
///   "atoms": [{"name": "C1", "atomic_num": 6, "formal_charge": 0, "leaving_atom": false,
///              "conformer_coords": [1.0, 2.0, 3.0], "chirality_tag": "CHI_UNSPECIFIED"}],
///   "bonds": [{"atom_idx_1": 0, "atom_idx_2": 1, "bond_type": "SINGLE"}]
/// }
/// ```
fn load_ccd_json(path: &Path) -> Result<CcdMolData> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("read CCD JSON: {}", path.display()))?;
    let mol: SerializableCcdMol = serde_json::from_str(&data)
        .with_context(|| format!("parse CCD JSON: {}", path.display()))?;
    Ok(mol.into_ccd_mol_data())
}

/// Serializable representation for JSON I/O.
#[derive(serde::Deserialize, serde::Serialize)]
struct SerializableCcdMol {
    code: String,
    atoms: Vec<SerializableCcdAtom>,
    bonds: Vec<SerializableCcdBond>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct SerializableCcdAtom {
    name: String,
    atomic_num: i32,
    formal_charge: i32,
    #[serde(default)]
    leaving_atom: bool,
    #[serde(default)]
    conformer_coords: [f32; 3],
    #[serde(default = "default_chirality")]
    chirality_tag: String,
}

fn default_chirality() -> String {
    "CHI_UNSPECIFIED".to_string()
}

#[derive(serde::Deserialize, serde::Serialize)]
struct SerializableCcdBond {
    atom_idx_1: usize,
    atom_idx_2: usize,
    bond_type: String,
}

impl SerializableCcdMol {
    fn into_ccd_mol_data(self) -> CcdMolData {
        CcdMolData::new(
            self.code,
            self.atoms
                .into_iter()
                .map(|a| CcdAtom {
                    name: a.name,
                    atomic_num: a.atomic_num,
                    formal_charge: a.formal_charge,
                    leaving_atom: a.leaving_atom,
                    conformer_coords: a.conformer_coords,
                    chirality_tag: a.chirality_tag,
                })
                .collect(),
            self.bonds
                .into_iter()
                .map(|b| CcdBond {
                    atom_idx_1: b.atom_idx_1,
                    atom_idx_2: b.atom_idx_2,
                    bond_type: b.bond_type,
                })
                .collect(),
        )
    }
}

/// Serialize a [`CcdMolData`] to JSON bytes.
pub fn serialize_ccd_mol_json(mol: &CcdMolData) -> Result<String> {
    let s = SerializableCcdMol {
        code: mol.code.clone(),
        atoms: mol
            .atoms
            .iter()
            .map(|a| SerializableCcdAtom {
                name: a.name.clone(),
                atomic_num: a.atomic_num,
                formal_charge: a.formal_charge,
                leaving_atom: a.leaving_atom,
                conformer_coords: a.conformer_coords,
                chirality_tag: a.chirality_tag.clone(),
            })
            .collect(),
        bonds: mol
            .bonds
            .iter()
            .map(|b| SerializableCcdBond {
                atom_idx_1: b.atom_idx_1,
                atom_idx_2: b.atom_idx_2,
                bond_type: b.bond_type.clone(),
            })
            .collect(),
    };
    serde_json::to_string_pretty(&s).context("serialize CCD mol to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccd_mol_data_construction() {
        let mol = CcdMolData::new(
            "TEST".to_string(),
            vec![
                CcdAtom {
                    name: "C1".to_string(),
                    atomic_num: 6,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [0.0, 0.0, 0.0],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
                CcdAtom {
                    name: "O1".to_string(),
                    atomic_num: 8,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [1.4, 0.0, 0.0],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
            ],
            vec![CcdBond {
                atom_idx_1: 0,
                atom_idx_2: 1,
                bond_type: "SINGLE".to_string(),
            }],
        );
        assert_eq!(mol.num_heavy_atoms, 2);
        assert!(!mol.is_single_atom());
        assert_eq!(mol.atom_by_name("C1").unwrap().0, 0);
        assert_eq!(mol.atom_names(), vec!["C1", "O1"]);
    }

    #[test]
    fn ccd_mol_provider_insert_and_get() {
        let mut provider = CcdMolProvider::new();
        let mol = CcdMolData::new(
            "SAH".to_string(),
            vec![CcdAtom {
                name: "C1".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [0.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            }],
            vec![],
        );
        provider.insert(mol);
        assert!(provider.contains("SAH"));
        assert!(!provider.contains("ATP"));
        let m = provider.get("SAH").unwrap();
        assert_eq!(m.code, "SAH");
        assert_eq!(m.num_heavy_atoms, 1);
        assert!(m.is_single_atom());
    }

    #[test]
    fn load_all_json_in_dir_empty_is_ok() {
        let dir = std::env::temp_dir().join(format!("boltr_empty_mols_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = CcdMolProvider::load_all_json_in_dir(&dir).expect("load_all");
        assert!(p.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mol = CcdMolData::new(
            "TEST".to_string(),
            vec![
                CcdAtom {
                    name: "C1".to_string(),
                    atomic_num: 6,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [1.0, 2.0, 3.0],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
            ],
            vec![],
        );
        let json = serialize_ccd_mol_json(&mol).unwrap();
        let deserialized: SerializableCcdMol = serde_json::from_str(&json).unwrap();
        let back = deserialized.into_ccd_mol_data();
        assert_eq!(back.code, "TEST");
        assert_eq!(back.atoms.len(), 1);
        assert_eq!(back.atoms[0].conformer_coords, [1.0, 2.0, 3.0]);
    }
}
