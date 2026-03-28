//! Boltz `process_residue_constraint_features` (`featurizerv2.py`) — RDKit-derived geometry constraints.
//!
//! Converts loaded `ResidueConstraints` into tensor format for the model. When no constraints
//! are present, returns empty tensors with fixed ranks matching Python behavior.

use crate::residue_constraints::{
    ChiralAtomConstraint, PlanarBondConstraint, PlanarRing5Constraint, PlanarRing6Constraint,
    RDKitBoundsConstraint, ResidueConstraints, StereoBondConstraint,
};
use ndarray::{Array1, Array2};

/// Tensors matching Python `process_residue_constraint_features` output.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueConstraintTensors {
    /// Shape: (2, N_bounds) where N_bounds is number of RDKit bounds constraints
    /// Transposed from atom_idxs which is (N_bounds, 2)
    pub rdkit_bounds_index: Array2<i64>,
    /// Shape: (N_bounds,)
    pub rdkit_bounds_bond_mask: Array1<bool>,
    /// Shape: (N_bounds,)
    pub rdkit_bounds_angle_mask: Array1<bool>,
    /// Shape: (N_bounds,)
    pub rdkit_upper_bounds: Array1<f32>,
    /// Shape: (N_bounds,)
    pub rdkit_lower_bounds: Array1<f32>,
    /// Shape: (4, N_chiral) where N_chiral is number of chiral atom constraints
    /// Transposed from atom_idxs which is (N_chiral, 4)
    pub chiral_atom_index: Array2<i64>,
    /// Shape: (N_chiral,)
    pub chiral_reference_mask: Array1<bool>,
    /// Shape: (N_chiral,)
    pub chiral_atom_orientations: Array1<bool>,
    /// Shape: (4, N_stereo) where N_stereo is number of stereo bond constraints
    /// Transposed from atom_idxs which is (N_stereo, 4)
    pub stereo_bond_index: Array2<i64>,
    /// Shape: (N_stereo,)
    pub stereo_reference_mask: Array1<bool>,
    /// Shape: (N_stereo,)
    pub stereo_bond_orientations: Array1<bool>,
    /// Shape: (6, N_planar_bond) where N_planar_bond is number of planar bond constraints
    /// Transposed from atom_idxs which is (N_planar_bond, 6)
    pub planar_bond_index: Array2<i64>,
    /// Shape: (5, N_ring5) where N_ring5 is number of 5-ring constraints
    /// Transposed from atom_idxs which is (N_ring5, 5)
    pub planar_ring_5_index: Array2<i64>,
    /// Shape: (6, N_ring6) where N_ring6 is number of 6-ring constraints
    /// Transposed from atom_idxs which is (N_ring6, 6)
    pub planar_ring_6_index: Array2<i64>,
}

/// Process residue constraints from loaded `ResidueConstraints` into tensor format.
///
/// Matches Python `process_residue_constraint_features(data: Tokenized) -> dict[str, Tensor]`
/// which converts constraints from NPZ arrays to PyTorch tensors with specific shapes and
/// transpositions.
///
/// # Arguments
///
/// * `constraints` - Optional residue constraints loaded from NPZ
///
/// # Returns
///
/// Tensor representation of constraints, or empty tensors if `constraints` is `None` or empty
#[must_use]
pub fn process_residue_constraint_features(
    constraints: Option<&ResidueConstraints>,
) -> ResidueConstraintTensors {
    match constraints {
        Some(rc) if !rc.is_empty() => {
            // Convert RDKit bounds constraints
            let rdkit_bounds = &rc.rdkit_bounds_constraints;
            let n_bounds = rdkit_bounds.len();
            let mut rdkit_bounds_index = Array2::zeros((2, n_bounds));
            let mut rdkit_bounds_bond_mask = Array1::from_elem(n_bounds, false);
            let mut rdkit_bounds_angle_mask = Array1::from_elem(n_bounds, false);
            let mut rdkit_upper_bounds = Array1::<f32>::zeros(n_bounds);
            let mut rdkit_lower_bounds = Array1::<f32>::zeros(n_bounds);

            for (i, c) in rdkit_bounds.iter().enumerate() {
                rdkit_bounds_index[[0, i]] = c.atom_idxs[0] as i64;
                rdkit_bounds_index[[1, i]] = c.atom_idxs[1] as i64;
                rdkit_bounds_bond_mask[i] = c.is_bond;
                rdkit_bounds_angle_mask[i] = c.is_angle;
                rdkit_upper_bounds[i] = c.upper_bound;
                rdkit_lower_bounds[i] = c.lower_bound;
            }

            // Convert chiral atom constraints
            let chiral_atoms = &rc.chiral_atom_constraints;
            let n_chiral = chiral_atoms.len();
            let mut chiral_atom_index = Array2::zeros((4, n_chiral));
            let mut chiral_reference_mask = Array1::from_elem(n_chiral, false);
            let mut chiral_atom_orientations = Array1::from_elem(n_chiral, false);

            for (i, c) in chiral_atoms.iter().enumerate() {
                for (j, &atom_idx) in c.atom_idxs.iter().enumerate() {
                    chiral_atom_index[[j, i]] = atom_idx as i64;
                }
                chiral_reference_mask[i] = c.is_reference;
                chiral_atom_orientations[i] = c.is_r;
            }

            // Convert stereo bond constraints
            let stereo_bonds = &rc.stereo_bond_constraints;
            let n_stereo = stereo_bonds.len();
            let mut stereo_bond_index = Array2::zeros((4, n_stereo));
            let mut stereo_reference_mask = Array1::from_elem(n_stereo, false);
            let mut stereo_bond_orientations = Array1::from_elem(n_stereo, false);

            for (i, c) in stereo_bonds.iter().enumerate() {
                for (j, &atom_idx) in c.atom_idxs.iter().enumerate() {
                    stereo_bond_index[[j, i]] = atom_idx as i64;
                }
                stereo_reference_mask[i] = c.is_reference;
                stereo_bond_orientations[i] = c.is_e;
            }

            // Convert planar bond constraints
            let planar_bonds = &rc.planar_bond_constraints;
            let n_planar_bond = planar_bonds.len();
            let mut planar_bond_index = Array2::zeros((6, n_planar_bond));

            for (i, c) in planar_bonds.iter().enumerate() {
                for (j, &atom_idx) in c.atom_idxs.iter().enumerate() {
                    planar_bond_index[[j, i]] = atom_idx as i64;
                }
            }

            // Convert planar 5-ring constraints
            let planar_rings_5 = &rc.planar_ring_5_constraints;
            let n_ring5 = planar_rings_5.len();
            let mut planar_ring_5_index = Array2::zeros((5, n_ring5));

            for (i, c) in planar_rings_5.iter().enumerate() {
                for (j, &atom_idx) in c.atom_idxs.iter().enumerate() {
                    planar_ring_5_index[[j, i]] = atom_idx as i64;
                }
            }

            // Convert planar 6-ring constraints
            let planar_rings_6 = &rc.planar_ring_6_constraints;
            let n_ring6 = planar_rings_6.len();
            let mut planar_ring_6_index = Array2::zeros((6, n_ring6));

            for (i, c) in planar_rings_6.iter().enumerate() {
                for (j, &atom_idx) in c.atom_idxs.iter().enumerate() {
                    planar_ring_6_index[[j, i]] = atom_idx as i64;
                }
            }

            ResidueConstraintTensors {
                rdkit_bounds_index,
                rdkit_bounds_bond_mask,
                rdkit_bounds_angle_mask,
                rdkit_upper_bounds,
                rdkit_lower_bounds,
                chiral_atom_index,
                chiral_reference_mask,
                chiral_atom_orientations,
                stereo_bond_index,
                stereo_reference_mask,
                stereo_bond_orientations,
                planar_bond_index,
                planar_ring_5_index,
                planar_ring_6_index,
            }
        }
        _ => {
            // Return empty tensors matching Python else branch
            ResidueConstraintTensors {
                rdkit_bounds_index: Array2::zeros((2, 0)),
                rdkit_bounds_bond_mask: Array1::from_elem(0, false),
                rdkit_bounds_angle_mask: Array1::from_elem(0, false),
                rdkit_upper_bounds: Array1::<f32>::zeros(0),
                rdkit_lower_bounds: Array1::<f32>::zeros(0),
                chiral_atom_index: Array2::zeros((4, 0)),
                chiral_reference_mask: Array1::from_elem(0, false),
                chiral_atom_orientations: Array1::from_elem(0, false),
                stereo_bond_index: Array2::zeros((4, 0)),
                stereo_reference_mask: Array1::from_elem(0, false),
                stereo_bond_orientations: Array1::from_elem(0, false),
                planar_bond_index: Array2::zeros((6, 0)),
                planar_ring_5_index: Array2::zeros((5, 0)),
                planar_ring_6_index: Array2::zeros((6, 0)),
            }
        }
    }
}

/// Same empty tensors as Python `residue_constraints is None` (`torch.empty` shapes).
#[must_use]
pub fn inference_residue_constraint_features() -> ResidueConstraintTensors {
    ResidueConstraintTensors {
        rdkit_bounds_index: Array2::zeros((2, 0)),
        rdkit_bounds_bond_mask: Array1::from_elem(0, false),
        rdkit_bounds_angle_mask: Array1::from_elem(0, false),
        rdkit_upper_bounds: Array1::<f32>::zeros(0),
        rdkit_lower_bounds: Array1::<f32>::zeros(0),
        chiral_atom_index: Array2::zeros((4, 0)),
        chiral_reference_mask: Array1::from_elem(0, false),
        chiral_atom_orientations: Array1::from_elem(0, false),
        stereo_bond_index: Array2::zeros((4, 0)),
        stereo_reference_mask: Array1::from_elem(0, false),
        stereo_bond_orientations: Array1::from_elem(0, false),
        planar_bond_index: Array2::zeros((6, 0)),
        planar_ring_5_index: Array2::zeros((5, 0)),
        planar_ring_6_index: Array2::zeros((6, 0)),
    }
}

impl ResidueConstraintTensors {
    /// Convert to FeatureBatch for collation.
    pub fn into_feature_batch(self) -> crate::feature_batch::FeatureBatch {
        let mut batch = crate::feature_batch::FeatureBatch::new();

        batch.insert_i64("rdkit_bounds_index", self.rdkit_bounds_index.into_dyn());
        batch.insert_i64(
            "rdkit_bounds_bond_mask",
            self.rdkit_bounds_bond_mask
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_i64(
            "rdkit_bounds_angle_mask",
            self.rdkit_bounds_angle_mask
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_f32("rdkit_upper_bounds", self.rdkit_upper_bounds.into_dyn());
        batch.insert_f32("rdkit_lower_bounds", self.rdkit_lower_bounds.into_dyn());
        batch.insert_i64("chiral_atom_index", self.chiral_atom_index.into_dyn());
        batch.insert_i64(
            "chiral_reference_mask",
            self.chiral_reference_mask
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_i64(
            "chiral_atom_orientations",
            self.chiral_atom_orientations
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_i64("stereo_bond_index", self.stereo_bond_index.into_dyn());
        batch.insert_i64(
            "stereo_reference_mask",
            self.stereo_reference_mask
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_i64(
            "stereo_bond_orientations",
            self.stereo_bond_orientations
                .mapv(|b| if b { 1_i64 } else { 0 })
                .into_dyn(),
        );
        batch.insert_i64("planar_bond_index", self.planar_bond_index.into_dyn());
        batch.insert_i64("planar_ring_5_index", self.planar_ring_5_index.into_dyn());
        batch.insert_i64("planar_ring_6_index", self.planar_ring_6_index.into_dyn());

        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_constraints_match_python_else_branch() {
        let t = process_residue_constraint_features(None);
        assert_eq!(t.rdkit_bounds_index.shape(), [2, 0]);
        assert_eq!(t.chiral_atom_index.shape(), [4, 0]);
        assert_eq!(t.stereo_bond_index.shape(), [4, 0]);
        assert_eq!(t.planar_bond_index.shape(), [6, 0]);
        assert_eq!(t.planar_ring_5_index.shape(), [5, 0]);
        assert_eq!(t.planar_ring_6_index.shape(), [6, 0]);
        assert_eq!(t.rdkit_bounds_bond_mask.len(), 0);
    }

    #[test]
    fn empty_constraints_object_returns_empty_tensors() {
        let rc = ResidueConstraints::default();
        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.rdkit_bounds_index.shape(), [2, 0]);
    }

    #[test]
    fn rdkit_bounds_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.rdkit_bounds_constraints.push(RDKitBoundsConstraint {
            atom_idxs: [0, 1],
            is_bond: true,
            is_angle: false,
            upper_bound: 2.5,
            lower_bound: 1.0,
        });
        rc.rdkit_bounds_constraints.push(RDKitBoundsConstraint {
            atom_idxs: [2, 3],
            is_bond: false,
            is_angle: true,
            upper_bound: 180.0,
            lower_bound: 120.0,
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.rdkit_bounds_index.shape(), [2, 2]);
        assert_eq!(t.rdkit_bounds_index[[0, 0]], 0);
        assert_eq!(t.rdkit_bounds_index[[1, 0]], 1);
        assert_eq!(t.rdkit_bounds_index[[0, 1]], 2);
        assert_eq!(t.rdkit_bounds_index[[1, 1]], 3);
        assert_eq!(t.rdkit_bounds_bond_mask[0], true);
        assert_eq!(t.rdkit_bounds_bond_mask[1], false);
        assert_eq!(t.rdkit_bounds_angle_mask[0], false);
        assert_eq!(t.rdkit_bounds_angle_mask[1], true);
        assert_eq!(t.rdkit_upper_bounds[0], 2.5);
        assert_eq!(t.rdkit_upper_bounds[1], 180.0);
        assert_eq!(t.rdkit_lower_bounds[0], 1.0);
        assert_eq!(t.rdkit_lower_bounds[1], 120.0);
    }

    #[test]
    fn chiral_atoms_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.chiral_atom_constraints.push(ChiralAtomConstraint {
            atom_idxs: [1, 2, 3, 4],
            is_reference: false,
            is_r: true,
        });
        rc.chiral_atom_constraints.push(ChiralAtomConstraint {
            atom_idxs: [5, 6, 7, 8],
            is_reference: true,
            is_r: false,
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.chiral_atom_index.shape(), [4, 2]);
        assert_eq!(t.chiral_atom_index[[0, 0]], 1);
        assert_eq!(t.chiral_atom_index[[1, 0]], 2);
        assert_eq!(t.chiral_atom_index[[2, 0]], 3);
        assert_eq!(t.chiral_atom_index[[3, 0]], 4);
        assert_eq!(t.chiral_atom_index[[0, 1]], 5);
        assert_eq!(t.chiral_reference_mask[0], false);
        assert_eq!(t.chiral_reference_mask[1], true);
        assert_eq!(t.chiral_atom_orientations[0], true);
        assert_eq!(t.chiral_atom_orientations[1], false);
    }

    #[test]
    fn stereo_bonds_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.stereo_bond_constraints.push(StereoBondConstraint {
            atom_idxs: [0, 1, 2, 3],
            is_reference: false,
            is_e: true,
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.stereo_bond_index.shape(), [4, 1]);
        assert_eq!(t.stereo_bond_index[[0, 0]], 0);
        assert_eq!(t.stereo_bond_index[[1, 0]], 1);
        assert_eq!(t.stereo_bond_index[[2, 0]], 2);
        assert_eq!(t.stereo_bond_index[[3, 0]], 3);
        assert_eq!(t.stereo_reference_mask[0], false);
        assert_eq!(t.stereo_bond_orientations[0], true);
    }

    #[test]
    fn planar_bonds_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.planar_bond_constraints.push(PlanarBondConstraint {
            atom_idxs: [0, 1, 2, 3, 4, 5],
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.planar_bond_index.shape(), [6, 1]);
        for i in 0..6 {
            assert_eq!(t.planar_bond_index[[i, 0]], i as i64);
        }
    }

    #[test]
    fn planar_rings_5_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.planar_ring_5_constraints.push(PlanarRing5Constraint {
            atom_idxs: [0, 1, 2, 3, 4],
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.planar_ring_5_index.shape(), [5, 1]);
        for i in 0..5 {
            assert_eq!(t.planar_ring_5_index[[i, 0]], i as i64);
        }
    }

    #[test]
    fn planar_rings_6_converted_correctly() {
        let mut rc = ResidueConstraints::default();
        rc.planar_ring_6_constraints.push(PlanarRing6Constraint {
            atom_idxs: [0, 1, 2, 3, 4, 5],
        });

        let t = process_residue_constraint_features(Some(&rc));
        assert_eq!(t.planar_ring_6_index.shape(), [6, 1]);
        for i in 0..6 {
            assert_eq!(t.planar_ring_6_index[[i, 0]], i as i64);
        }
    }

    #[test]
    fn into_feature_batch_conversion() {
        let mut rc = ResidueConstraints::default();
        rc.rdkit_bounds_constraints.push(RDKitBoundsConstraint {
            atom_idxs: [0, 1],
            is_bond: true,
            is_angle: false,
            upper_bound: 2.0,
            lower_bound: 1.0,
        });
        rc.chiral_atom_constraints.push(ChiralAtomConstraint {
            atom_idxs: [2, 3, 4, 5],
            is_reference: false,
            is_r: true,
        });

        let t = process_residue_constraint_features(Some(&rc));
        let batch = t.into_feature_batch();

        // Verify all keys are present
        assert!(batch.tensors.contains_key("rdkit_bounds_index"));
        assert!(batch.tensors.contains_key("chiral_atom_index"));
        assert!(batch.tensors.contains_key("stereo_bond_index"));
        assert!(batch.tensors.contains_key("planar_bond_index"));
        assert!(batch.tensors.contains_key("planar_ring_5_index"));
        assert!(batch.tensors.contains_key("planar_ring_6_index"));
    }
}
