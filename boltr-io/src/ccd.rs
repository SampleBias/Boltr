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
        self.atoms.iter().enumerate().find(|(_, a)| a.name == name)
    }

    /// Heavy atom names in order.
    pub fn atom_names(&self) -> Vec<&str> {
        self.atoms.iter().map(|a| a.name.as_str()).collect()
    }

    /// Check if this is a single-heavy-atom molecule.
    pub fn is_single_atom(&self) -> bool {
        self.num_heavy_atoms == 1
    }
    /// Extract symmetry swap groups from this molecule.
    ///
    /// This identifies symmetric atom pairs based on bond topology.
    /// Returns groups of `(atom_idx_1, atom_idx_2)` pairs where swapping
    /// these atoms would result in an equivalent molecular graph.
    ///
    /// Note: This is a simplified symmetry detection. Full symmetry detection
    /// requires RDKit's graph isomorphism algorithms. Six-membered aromatic
    /// pairing uses the first found cycle and DFS visit order (not canonical
    /// ring numbering). This implementation identifies:
    /// - Aromatic ring symmetries (e.g., phenyl ring 180° rotations)
    /// - Bond-type symmetries (equivalent atoms in symmetric substructures)
    ///
    /// Returns: A vector of symmetry groups, where each group is a vector
    /// of atom index pairs that can be swapped.
    #[must_use]
    pub fn extract_symmetry_groups(&self) -> Vec<Vec<(usize, usize)>> {
        let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();

        // Find aromatic bonds for ring symmetry detection
        let aromatic_bonds: Vec<(usize, usize)> = self
            .bonds
            .iter()
            .filter(|b| b.bond_type == "AROMATIC")
            .map(|b| (b.atom_idx_1, b.atom_idx_2))
            .collect();

        // Build adjacency list for aromatic rings
        let mut adjacency: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for &(i, j) in &aromatic_bonds {
            adjacency.entry(i).or_default().push(j);
            adjacency.entry(j).or_default().push(i);
        }

        // Detect 6-membered aromatic rings (phenyl, pyridine, etc.)
        // These typically have 180° rotational symmetry
        for &start in adjacency.keys() {
            let visited = self.find_aromatic_ring(&adjacency, start, 6);
            if visited.len() == 6 {
                // Found a 6-membered aromatic ring
                // Identify opposite atoms in the ring for 180° rotation symmetry
                let ring: Vec<usize> = visited;
                for i in 0..3 {
                    let atom1 = ring[i];
                    let atom2 = ring[i + 3];
                    // Only add if both atoms exist and are not already paired
                    if atom1 < self.atoms.len() && atom2 < self.atoms.len() {
                        groups.push(vec![(atom1, atom2), (atom2, atom1)]);
                    }
                }
                break; // Only process the first ring found
            }
        }

        // Detect symmetric terminal groups (e.g., -CH3, -OH in symmetric positions)
        // Find atoms with identical bond environments
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                if self.are_atoms_equivalent(i, j, &aromatic_bonds) {
                    // Check if this pair is not already in a group
                    let already_grouped = groups.iter().any(|g| {
                        g.iter()
                            .any(|(a, b)| (*a == i && *b == j) || (*a == j && *b == i))
                    });
                    if !already_grouped {
                        groups.push(vec![(i, j), (j, i)]);
                    }
                }
            }
        }

        groups
    }

    /// Find an aromatic ring of given size starting from a specific atom.
    fn find_aromatic_ring(
        &self,
        adjacency: &std::collections::HashMap<usize, Vec<usize>>,
        start: usize,
        target_size: usize,
    ) -> Vec<usize> {
        let mut visited: Vec<usize> = Vec::new();
        let mut stack: Vec<(usize, Option<usize>)> = vec![(start, None)];

        while let Some((current, parent)) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.push(current);

            if let Some(neighbors) = adjacency.get(&current) {
                for &neighbor in neighbors {
                    if Some(neighbor) != parent && !visited.contains(&neighbor) {
                        stack.push((neighbor, Some(current)));
                    }
                }
            }

            if visited.len() >= target_size {
                // Check if it forms a ring (last connects to first)
                if let Some(neighbors) = adjacency.get(&current) {
                    if neighbors.contains(&start) && visited.len() == target_size {
                        return visited;
                    }
                }
                break;
            }
        }

        Vec::new()
    }

    /// Check if two atoms have equivalent bond environments.
    fn are_atoms_equivalent(&self, i: usize, j: usize, _aromatic_bonds: &[(usize, usize)]) -> bool {
        // Get bond partners for each atom
        let partners_i: Vec<usize> = self
            .bonds
            .iter()
            .filter_map(|b| {
                if b.atom_idx_1 == i {
                    Some(b.atom_idx_2)
                } else if b.atom_idx_2 == i {
                    Some(b.atom_idx_1)
                } else {
                    None
                }
            })
            .collect();

        let partners_j: Vec<usize> = self
            .bonds
            .iter()
            .filter_map(|b| {
                if b.atom_idx_1 == j {
                    Some(b.atom_idx_2)
                } else if b.atom_idx_2 == j {
                    Some(b.atom_idx_1)
                } else {
                    None
                }
            })
            .collect();

        // Different number of bonds -> not equivalent
        if partners_i.len() != partners_j.len() {
            return false;
        }

        // Check if atoms have same element
        if self.atoms[i].atomic_num != self.atoms[j].atomic_num {
            return false;
        }

        // Check if bonds have same types
        for &p_i in &partners_i {
            let bond_type_i = self.get_bond_type(i, p_i);
            let has_match = partners_j.iter().any(|&p_j| {
                let bond_type_j = self.get_bond_type(j, p_j);
                bond_type_i == bond_type_j
            });
            if !has_match {
                return false;
            }
        }

        true
    }

    /// Get bond type between two atoms.
    fn get_bond_type(&self, i: usize, j: usize) -> &str {
        self.bonds
            .iter()
            .find(|b| {
                (b.atom_idx_1 == i && b.atom_idx_2 == j) || (b.atom_idx_1 == j && b.atom_idx_2 == i)
            })
            .map(|b| b.bond_type.as_str())
            .unwrap_or("NONE")
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
                let mol = load_ccd_json(&json_path).with_context(|| {
                    format!("load CCD molecule {code} from {}", json_path.display())
                })?;
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
    /// Build a symmetry map for all loaded molecules.
    ///
    /// Returns a HashMap where keys are CCD codes and values are
    /// symmetry groups (vectors of atom index pairs that can be swapped).
    ///
    /// This is used for ligand symmetry features in the folding prediction pipeline.
    #[must_use]
    pub fn build_symmetry_map(&self) -> HashMap<String, Vec<Vec<(usize, usize)>>> {
        self.mols
            .iter()
            .map(|(code, mol)| (code.clone(), mol.extract_symmetry_groups()))
            .collect()
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
            vec![CcdAtom {
                name: "C1".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [1.0, 2.0, 3.0],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            }],
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

#[cfg(test)]
mod symmetry_tests {
    use super::*;

    #[test]
    fn test_extract_symmetry_groups_single_atom() {
        let mol = CcdMolData::new(
            "LIG".to_string(),
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
        let groups = mol.extract_symmetry_groups();
        assert!(
            groups.is_empty(),
            "Single atom should have no symmetry groups"
        );
    }

    #[test]
    fn test_extract_symmetry_groups_aromatic_ring() {
        // Create a simple 6-membered aromatic ring (simplified benzene)
        let atoms = vec![
            CcdAtom {
                name: "C1".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [0.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "C2".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [1.4; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "C3".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [2.8; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "C4".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [4.2; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "C5".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [5.6; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "C6".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [7.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
        ];

        // Create aromatic bonds in a ring
        let bonds = vec![
            CcdBond {
                atom_idx_1: 0,
                atom_idx_2: 1,
                bond_type: "AROMATIC".to_string(),
            },
            CcdBond {
                atom_idx_1: 1,
                atom_idx_2: 2,
                bond_type: "AROMATIC".to_string(),
            },
            CcdBond {
                atom_idx_1: 2,
                atom_idx_2: 3,
                bond_type: "AROMATIC".to_string(),
            },
            CcdBond {
                atom_idx_1: 3,
                atom_idx_2: 4,
                bond_type: "AROMATIC".to_string(),
            },
            CcdBond {
                atom_idx_1: 4,
                atom_idx_2: 5,
                bond_type: "AROMATIC".to_string(),
            },
            CcdBond {
                atom_idx_1: 5,
                atom_idx_2: 0,
                bond_type: "AROMATIC".to_string(),
            },
        ];

        let mol = CcdMolData::new("BENZENE".to_string(), atoms, bonds);
        let groups = mol.extract_symmetry_groups();

        // Should find 3 symmetry pairs (opposite atoms in 6-membered ring)
        assert!(
            !groups.is_empty(),
            "Aromatic ring should have symmetry groups"
        );

        // Each group should have 2 pairs (i,j) and (j,i)
        for group in &groups {
            assert_eq!(group.len(), 2, "Each symmetry group should have 2 pairs");
        }
    }

    #[test]
    fn test_extract_symmetry_groups_equivalent_atoms() {
        // Create a molecule with two equivalent terminal atoms
        let atoms = vec![
            CcdAtom {
                name: "C1".to_string(),
                atomic_num: 6,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [0.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "H1".to_string(),
                atomic_num: 1,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [1.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
            CcdAtom {
                name: "H2".to_string(),
                atomic_num: 1,
                formal_charge: 0,
                leaving_atom: false,
                conformer_coords: [2.0; 3],
                chirality_tag: "CHI_UNSPECIFIED".to_string(),
            },
        ];

        // Both H1 and H2 are bonded to C1 (symmetric)
        let bonds = vec![
            CcdBond {
                atom_idx_1: 0,
                atom_idx_2: 1,
                bond_type: "SINGLE".to_string(),
            },
            CcdBond {
                atom_idx_1: 0,
                atom_idx_2: 2,
                bond_type: "SINGLE".to_string(),
            },
        ];

        let mol = CcdMolData::new("CH3".to_string(), atoms, bonds);
        let groups = mol.extract_symmetry_groups();

        // Should find symmetry between H1 and H2
        assert!(
            !groups.is_empty(),
            "Equivalent atoms should have symmetry groups"
        );

        // Check that H1-H2 symmetry is found
        let has_h1_h2_symmetry = groups.iter().any(|g| {
            g.iter()
                .any(|(a, b)| (*a == 1 && *b == 2) || (*a == 2 && *b == 1))
        });
        assert!(
            has_h1_h2_symmetry,
            "Should find symmetry between equivalent H atoms"
        );
    }

    #[test]
    fn test_build_symmetry_map_provider() {
        let mut provider = CcdMolProvider::new();

        // Add two molecules
        let mol1 = CcdMolData::new(
            "MOL1".to_string(),
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

        let mol2 = CcdMolData::new(
            "MOL2".to_string(),
            vec![
                CcdAtom {
                    name: "C1".to_string(),
                    atomic_num: 6,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [0.0; 3],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
                CcdAtom {
                    name: "H1".to_string(),
                    atomic_num: 1,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [1.0; 3],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
                CcdAtom {
                    name: "H2".to_string(),
                    atomic_num: 1,
                    formal_charge: 0,
                    leaving_atom: false,
                    conformer_coords: [2.0; 3],
                    chirality_tag: "CHI_UNSPECIFIED".to_string(),
                },
            ],
            vec![
                CcdBond {
                    atom_idx_1: 0,
                    atom_idx_2: 1,
                    bond_type: "SINGLE".to_string(),
                },
                CcdBond {
                    atom_idx_1: 0,
                    atom_idx_2: 2,
                    bond_type: "SINGLE".to_string(),
                },
            ],
        );

        provider.insert(mol1);
        provider.insert(mol2);

        let symmetry_map = provider.build_symmetry_map();

        assert_eq!(
            symmetry_map.len(),
            2,
            "Should have symmetry for both molecules"
        );
        assert!(symmetry_map.contains_key("MOL1"));
        assert!(symmetry_map.contains_key("MOL2"));

        // MOL1 has no symmetry (single atom)
        assert!(symmetry_map.get("MOL1").unwrap().is_empty());

        // MOL2 has symmetry between H1 and H2
        assert!(!symmetry_map.get("MOL2").unwrap().is_empty());
    }
}
