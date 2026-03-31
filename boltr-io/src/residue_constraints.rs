//! Residue constraints for Boltz2 inference.
//!
//! Matches Python `boltz.data.types.ResidueConstraints` which is stored in NPZ format.
//! Used by the featurizer to incorporate geometry constraints during inference.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

use crate::structure_v2_npz::{parse_npy_shape_and_payload, read_f32_le, read_i32_le};

/// RDKit bounds constraint: distance/angle/dihedral limits between atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct RDKitBoundsConstraint {
    /// Indices of the two atoms involved in the constraint.
    pub atom_idxs: [i32; 2],
    /// Whether this represents a bond constraint.
    pub is_bond: bool,
    /// Whether this represents an angle constraint.
    pub is_angle: bool,
    /// Upper bound for the constraint value (in Å or degrees).
    pub upper_bound: f32,
    /// Lower bound for the constraint value (in Å or degrees).
    pub lower_bound: f32,
}

/// Chiral atom constraint: enforces tetrahedral stereochemistry.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiralAtomConstraint {
    /// Indices of the four atoms (central atom + 3 substituents).
    pub atom_idxs: [i32; 4],
    /// Whether this is a reference constraint (vs. to be enforced).
    pub is_reference: bool,
    /// Whether chirality should be R (vs. S).
    pub is_r: bool,
}

/// Stereo bond constraint: enforces double-bond stereochemistry (E/Z).
#[derive(Debug, Clone, PartialEq)]
pub struct StereoBondConstraint {
    /// Indices of the four atoms (atoms of the double bond + two substituents).
    pub atom_idxs: [i32; 4],
    /// Whether this is a reference constraint.
    pub is_reference: bool,
    /// Whether stereochemistry should be E (vs. Z).
    pub is_e: bool,
}

/// Planar bond constraint: enforces sp2 hybridization geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarBondConstraint {
    /// Indices of the six atoms involved.
    pub atom_idxs: [i32; 6],
}

/// Planar 5-ring constraint: enforces aromatic ring geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarRing5Constraint {
    /// Indices of the five atoms in the ring.
    pub atom_idxs: [i32; 5],
}

/// Planar 6-ring constraint: enforces aromatic ring geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarRing6Constraint {
    /// Indices of the six atoms in the ring.
    pub atom_idxs: [i32; 6],
}

/// Residue constraints loaded from Boltz preprocessing.
///
/// Matches Python `boltz.data.types.ResidueConstraints` and is loaded from NPZ format.
/// These constraints are used by the featurizer to incorporate known geometry information
/// during inference, particularly for ligands and non-standard residues.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResidueConstraints {
    /// Distance, angle, and dihedral constraints from RDKit.
    pub rdkit_bounds_constraints: Vec<RDKitBoundsConstraint>,
    /// Tetrahedral chirality constraints.
    pub chiral_atom_constraints: Vec<ChiralAtomConstraint>,
    /// Double-bond stereochemistry constraints.
    pub stereo_bond_constraints: Vec<StereoBondConstraint>,
    /// sp2 hybridization planar constraints.
    pub planar_bond_constraints: Vec<PlanarBondConstraint>,
    /// 5-membered aromatic ring planar constraints.
    pub planar_ring_5_constraints: Vec<PlanarRing5Constraint>,
    /// 6-membered aromatic ring planar constraints.
    pub planar_ring_6_constraints: Vec<PlanarRing6Constraint>,
}

impl ResidueConstraints {
    /// Load residue constraints from an NPZ file.
    ///
    /// Matches Python `ResidueConstraints.load(path)` which loads from NPZ using
    /// `np.load(path, allow_pickle=True)`.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the constraints NPZ file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if required arrays are missing.
    pub fn load_from_npz(path: &Path) -> Result<Self> {
        let mut f = File::open(path).with_context(|| {
            format!(
                "Failed to open residue constraints file: {}",
                path.display()
            )
        })?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::load_from_npz_bytes(&buf)
    }

    /// Load residue constraints from NPZ file bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - NPZ file bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if NPZ data is invalid.
    pub fn load_from_npz_bytes(bytes: &[u8]) -> Result<Self> {
        let cur = Cursor::new(bytes);
        let mut z = ZipArchive::new(cur).with_context(|| "Invalid NPZ archive")?;

        let rdkit_bounds = Self::load_rdkit_bounds(&mut z)?;
        let chiral_atoms = Self::load_chiral_atoms(&mut z)?;
        let stereo_bonds = Self::load_stereo_bonds(&mut z)?;
        let planar_bonds = Self::load_planar_bonds(&mut z)?;
        let planar_rings_5 = Self::load_planar_rings_5(&mut z)?;
        let planar_rings_6 = Self::load_planar_rings_6(&mut z)?;

        Ok(Self {
            rdkit_bounds_constraints: rdkit_bounds,
            chiral_atom_constraints: chiral_atoms,
            stereo_bond_constraints: stereo_bonds,
            planar_bond_constraints: planar_bonds,
            planar_ring_5_constraints: planar_rings_5,
            planar_ring_6_constraints: planar_rings_6,
        })
    }

    fn read_zip_npy<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
        stem: &str,
    ) -> Result<Option<Vec<u8>>> {
        let p = format!("{stem}.npy");
        if let Ok(mut f) = z.by_name(&p) {
            let mut v = Vec::new();
            f.read_to_end(&mut v)?;
            return Ok(Some(v));
        }
        Ok(None)
    }

    fn load_rdkit_bounds<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<RDKitBoundsConstraint>> {
        let Some(data) = Self::read_zip_npy(z, "rdkit_bounds_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (2,)), ('is_bond', '?'), ('is_angle', '?'), ('upper_bound', '<f4'), ('lower_bound', '<f4')]
        // Total record size: 2*4 + 1 + 1 + 4 + 4 = 18 bytes
        let rec_size = 18;
        if payload.len() % rec_size != 0 {
            bail!("Invalid rdkit_bounds_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [read_i32_le(payload, base)?, read_i32_le(payload, base + 4)?];
            let is_bond = payload[base + 8] != 0;
            let is_angle = payload[base + 9] != 0;
            let upper_bound = read_f32_le(payload, base + 12)?;
            let lower_bound = read_f32_le(payload, base + 16)?;

            constraints.push(RDKitBoundsConstraint {
                atom_idxs,
                is_bond,
                is_angle,
                upper_bound,
                lower_bound,
            });
        }

        Ok(constraints)
    }

    fn load_chiral_atoms<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<ChiralAtomConstraint>> {
        let Some(data) = Self::read_zip_npy(z, "chiral_atom_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (4,)), ('is_reference', '?'), ('is_r', '?')]
        // Total record size: 4*4 + 1 + 1 = 18 bytes
        let rec_size = 18;
        if payload.len() % rec_size != 0 {
            bail!("Invalid chiral_atom_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [
                read_i32_le(payload, base)?,
                read_i32_le(payload, base + 4)?,
                read_i32_le(payload, base + 8)?,
                read_i32_le(payload, base + 12)?,
            ];
            let is_reference = payload[base + 16] != 0;
            let is_r = payload[base + 17] != 0;

            constraints.push(ChiralAtomConstraint {
                atom_idxs,
                is_reference,
                is_r,
            });
        }

        Ok(constraints)
    }

    fn load_stereo_bonds<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<StereoBondConstraint>> {
        let Some(data) = Self::read_zip_npy(z, "stereo_bond_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (4,)), ('is_reference', '?'), ('is_e', '?')]
        // Total record size: 4*4 + 1 + 1 = 18 bytes
        let rec_size = 18;
        if payload.len() % rec_size != 0 {
            bail!("Invalid stereo_bond_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [
                read_i32_le(payload, base)?,
                read_i32_le(payload, base + 4)?,
                read_i32_le(payload, base + 8)?,
                read_i32_le(payload, base + 12)?,
            ];
            let is_reference = payload[base + 16] != 0;
            let is_e = payload[base + 17] != 0;

            constraints.push(StereoBondConstraint {
                atom_idxs,
                is_reference,
                is_e,
            });
        }

        Ok(constraints)
    }

    fn load_planar_bonds<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<PlanarBondConstraint>> {
        let Some(data) = Self::read_zip_npy(z, "planar_bond_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (6,))]
        // Total record size: 6*4 = 24 bytes
        let rec_size = 24;
        if payload.len() % rec_size != 0 {
            bail!("Invalid planar_bond_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [
                read_i32_le(payload, base)?,
                read_i32_le(payload, base + 4)?,
                read_i32_le(payload, base + 8)?,
                read_i32_le(payload, base + 12)?,
                read_i32_le(payload, base + 16)?,
                read_i32_le(payload, base + 20)?,
            ];

            constraints.push(PlanarBondConstraint { atom_idxs });
        }

        Ok(constraints)
    }

    fn load_planar_rings_5<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<PlanarRing5Constraint>> {
        let Some(data) = Self::read_zip_npy(z, "planar_ring_5_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (5,))]
        // Total record size: 5*4 = 20 bytes
        let rec_size = 20;
        if payload.len() % rec_size != 0 {
            bail!("Invalid planar_ring_5_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [
                read_i32_le(payload, base)?,
                read_i32_le(payload, base + 4)?,
                read_i32_le(payload, base + 8)?,
                read_i32_le(payload, base + 12)?,
                read_i32_le(payload, base + 16)?,
            ];

            constraints.push(PlanarRing5Constraint { atom_idxs });
        }

        Ok(constraints)
    }

    fn load_planar_rings_6<R: Read + std::io::Seek>(
        z: &mut ZipArchive<R>,
    ) -> Result<Vec<PlanarRing6Constraint>> {
        let Some(data) = Self::read_zip_npy(z, "planar_ring_6_constraints")? else {
            return Ok(Vec::new());
        };

        let (_shape, payload) = parse_npy_shape_and_payload(&data)?;
        let mut constraints = Vec::new();

        // Python dtype: [('atom_idxs', '<i4', (6,))]
        // Total record size: 6*4 = 24 bytes
        let rec_size = 24;
        if payload.len() % rec_size != 0 {
            bail!("Invalid planar_ring_6_constraints payload length");
        }

        for i in 0..(payload.len() / rec_size) {
            let base = i * rec_size;
            let atom_idxs = [
                read_i32_le(payload, base)?,
                read_i32_le(payload, base + 4)?,
                read_i32_le(payload, base + 8)?,
                read_i32_le(payload, base + 12)?,
                read_i32_le(payload, base + 16)?,
                read_i32_le(payload, base + 20)?,
            ];

            constraints.push(PlanarRing6Constraint { atom_idxs });
        }

        Ok(constraints)
    }

    /// Check if any constraints are present.
    pub fn is_empty(&self) -> bool {
        self.rdkit_bounds_constraints.is_empty()
            && self.chiral_atom_constraints.is_empty()
            && self.stereo_bond_constraints.is_empty()
            && self.planar_bond_constraints.is_empty()
            && self.planar_ring_5_constraints.is_empty()
            && self.planar_ring_6_constraints.is_empty()
    }

    /// Get the total number of constraints.
    pub fn total_count(&self) -> usize {
        self.rdkit_bounds_constraints.len()
            + self.chiral_atom_constraints.len()
            + self.stereo_bond_constraints.len()
            + self.planar_bond_constraints.len()
            + self.planar_ring_5_constraints.len()
            + self.planar_ring_6_constraints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constraints() {
        let rc = ResidueConstraints::default();
        assert!(rc.is_empty());
        assert_eq!(rc.total_count(), 0);
    }

    #[test]
    fn test_constraint_counts() {
        let mut rc = ResidueConstraints::default();
        rc.rdkit_bounds_constraints.push(RDKitBoundsConstraint {
            atom_idxs: [0, 1],
            is_bond: true,
            is_angle: false,
            upper_bound: 2.0,
            lower_bound: 1.0,
        });
        rc.chiral_atom_constraints.push(ChiralAtomConstraint {
            atom_idxs: [0, 1, 2, 3],
            is_reference: false,
            is_r: true,
        });

        assert_eq!(rc.total_count(), 2);
        assert!(!rc.is_empty());
    }
}
