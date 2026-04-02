//! In-memory `StructureV2`-shaped tables for tokenization (`boltz.data.types` AtomV2 / Residue / Chain / BondV2 subset).
//!
//! Field layout matches the numpy structured dtypes consumed by [`crate::tokenize::boltz2::tokenize_structure`].

/// One atom row (`AtomV2` in Python: `name`, `coords`, `is_present`, `bfactor`, `plddt`).
///
/// `name` is a max-4-character PDB atom name (e.g. `"N"`, `"CA"`, `"OP1"`, `"C1'"`).
/// `bfactor` and `plddt` default to `0.0` in inference.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomV2Row {
    /// PDB atom name (up to 4 chars, stripped).
    pub name: String,
    pub coords: [f32; 3],
    pub is_present: bool,
    pub bfactor: f32,
    pub plddt: f32,
}

/// One residue row (`Residue` dtype).
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueRow {
    pub name: String,
    pub res_type: i8,
    pub res_idx: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub atom_center: i32,
    pub atom_disto: i32,
    pub is_standard: bool,
    pub is_present: bool,
}

/// One chain row (`Chain` dtype).
#[derive(Clone, Debug, PartialEq)]
pub struct ChainRow {
    /// PDB chain name (e.g. `"A"`), used by template alignment (`TemplateInfo.query_chain`).
    pub name: String,
    pub mol_type: i8,
    pub sym_id: i32,
    pub asym_id: i32,
    pub entity_id: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub res_idx: i32,
    pub res_num: i32,
    pub cyclic_period: i32,
}

/// One `ensemble` row (`StructureV2.ensemble[i]`): base offset into `coords` + atom count.
#[derive(Clone, Debug, PartialEq)]
pub struct EnsembleRow {
    pub atom_coord_idx: i32,
    pub atom_num: i32,
}

/// Bond row subset used by tokenizer (`BondV2`: global atom indices + bond type).
#[derive(Clone, Debug, PartialEq)]
pub struct BondV2AtomRow {
    pub atom_1: i32,
    pub atom_2: i32,
    /// Raw `type` from structure (tokenizer stores `type + 1` in `TokenBondV2`).
    pub bond_type: i8,
}

/// Minimal structure matching Boltz `StructureV2` fields needed for `tokenize_structure`.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureV2Tables {
    pub atoms: Vec<AtomV2Row>,
    pub residues: Vec<ResidueRow>,
    pub chains: Vec<ChainRow>,
    /// Per-chain include flag (`struct.mask` in Python).
    pub chain_mask: Vec<bool>,
    /// Flattened coords table (`Coords`: one `[f32;3]` per atom slot in the ensemble slice).
    pub coords: Vec<[f32; 3]>,
    /// Conformer table (`ensemble.npy`): one row per available structure conformer.
    pub ensemble: Vec<EnsembleRow>,
    /// `ensemble[0].atom_coord_idx` — duplicated for callers that only track the first conformer.
    pub ensemble_atom_coord_idx: i32,
    pub bonds: Vec<BondV2AtomRow>,
}

impl StructureV2Tables {
    /// Copy positions from the flat `coords` table into each [`AtomV2Row::coords`] for the first
    /// ensemble conformer (`ensemble_base_offset(0) + atom_index`).
    ///
    /// Boltz preprocess may store coordinates only in `coords` while per-atom records contain
    /// zeros. [`crate::featurizer::process_atom_features`] reads the flat table; mmCIF/PDB export
    /// reads `atom.coords` — without syncing, exports can show all zeros while the
    /// model still sees correct inputs.
    pub fn sync_atom_coords_from_flat_table(&mut self) {
        let base = self.ensemble_base_offset(0) as usize;
        for i in 0..self.atoms.len() {
            if let Some(c) = self.coords.get(base + i) {
                self.atoms[i].coords = *c;
            }
        }
    }

    /// Number of conformers (`len(structure.ensemble)` in Boltz).
    #[inline]
    #[must_use]
    pub fn num_ensemble_conformers(&self) -> usize {
        if !self.ensemble.is_empty() {
            self.ensemble.len()
        } else {
            1
        }
    }

    /// Base offset into `coords` for conformer `ensemble_idx` (defaults to first row).
    #[inline]
    pub(crate) fn ensemble_base_offset(&self, ensemble_idx: usize) -> i64 {
        self.ensemble
            .get(ensemble_idx)
            .map(|e| i64::from(e.atom_coord_idx))
            .unwrap_or(i64::from(self.ensemble_atom_coord_idx))
    }

    /// Coords row for `ensemble[0].atom_coord_idx + atom_index` (Boltz coord table).
    #[inline]
    pub(crate) fn ensemble_coords(&self, atom_index: i32) -> Option<[f32; 3]> {
        let o = self.ensemble_base_offset(0) + i64::from(atom_index);
        let u = usize::try_from(o).ok()?;
        self.coords.get(u).copied()
    }

    /// Apply first `xyz.len()` predicted positions to atom rows (PDB/mmCIF use [`AtomV2Row::coords`])
    /// and the matching slice of the flat `coords` table for the first ensemble.
    pub fn apply_predicted_atom_coords(&mut self, xyz: &[[f32; 3]]) {
        let n = xyz.len().min(self.atoms.len());
        for i in 0..n {
            self.atoms[i].coords = xyz[i];
        }
        let base = self.ensemble_base_offset(0) as usize;
        for i in 0..n {
            let j = base + i;
            if j < self.coords.len() {
                self.coords[j] = xyz[i];
            }
        }
    }

    /// Returns true when at least one atom is marked present and every present atom has coordinates
    /// within `eps` of the origin (placeholder or failed coordinate write).
    #[must_use]
    pub fn present_atoms_all_coords_near_zero(&self, eps: f32) -> bool {
        let mut any_present = false;
        for atom in &self.atoms {
            if atom.is_present {
                any_present = true;
                if atom.coords[0].abs() > eps
                    || atom.coords[1].abs() > eps
                    || atom.coords[2].abs() > eps
                {
                    return false;
                }
            }
        }
        any_present
    }
}

#[cfg(test)]
mod present_coords_tests {
    use super::{AtomV2Row, StructureV2Tables};

    fn empty() -> StructureV2Tables {
        StructureV2Tables {
            atoms: vec![],
            residues: vec![],
            chains: vec![],
            chain_mask: vec![],
            coords: vec![],
            ensemble: vec![],
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        }
    }

    #[test]
    fn empty_structure_not_all_zero_flag() {
        assert!(!empty().present_atoms_all_coords_near_zero(1e-12));
    }

    #[test]
    fn all_present_zero_is_flagged() {
        let s = StructureV2Tables {
            atoms: vec![AtomV2Row {
                name: "CA".into(),
                coords: [0.0, 0.0, 0.0],
                is_present: true,
                bfactor: 0.0,
                plddt: 0.0,
            }],
            residues: vec![],
            chains: vec![],
            chain_mask: vec![],
            coords: vec![],
            ensemble: vec![],
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        };
        assert!(s.present_atoms_all_coords_near_zero(1e-12));
    }

    #[test]
    fn nonzero_present_not_flagged() {
        let s = StructureV2Tables {
            atoms: vec![AtomV2Row {
                name: "CA".into(),
                coords: [1.0, 0.0, 0.0],
                is_present: true,
                bfactor: 0.0,
                plddt: 0.0,
            }],
            residues: vec![],
            chains: vec![],
            chain_mask: vec![],
            coords: vec![],
            ensemble: vec![],
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        };
        assert!(!s.present_atoms_all_coords_near_zero(1e-12));
    }
}
