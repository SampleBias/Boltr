//! Boltz `process_residue_constraint_features` (`featurizerv2.py`) — RDKit-derived geometry constraints.
//!
//! When `residue_constraints` is `None`, Python builds **empty** tensors with fixed ranks (the
//! `else` branch). This module matches those shapes for inference collate key parity.

use ndarray::{Array1, Array2};

/// Tensors matching Python `process_residue_constraint_features` when constraints are absent.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueConstraintTensors {
    pub rdkit_bounds_index: Array2<i64>,
    pub rdkit_bounds_bond_mask: Array1<bool>,
    pub rdkit_bounds_angle_mask: Array1<bool>,
    pub rdkit_upper_bounds: Array1<f32>,
    pub rdkit_lower_bounds: Array1<f32>,
    pub chiral_atom_index: Array2<i64>,
    pub chiral_reference_mask: Array1<bool>,
    pub chiral_atom_orientations: Array1<bool>,
    pub stereo_bond_index: Array2<i64>,
    pub stereo_reference_mask: Array1<bool>,
    pub stereo_bond_orientations: Array1<bool>,
    pub planar_bond_index: Array2<i64>,
    pub planar_ring_5_index: Array2<i64>,
    pub planar_ring_6_index: Array2<i64>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shapes_match_python_else_branch() {
        let t = inference_residue_constraint_features();
        assert_eq!(t.rdkit_bounds_index.shape(), [2, 0]);
        assert_eq!(t.chiral_atom_index.shape(), [4, 0]);
        assert_eq!(t.stereo_bond_index.shape(), [4, 0]);
        assert_eq!(t.planar_bond_index.shape(), [6, 0]);
        assert_eq!(t.planar_ring_5_index.shape(), [5, 0]);
        assert_eq!(t.planar_ring_6_index.shape(), [6, 0]);
        assert_eq!(t.rdkit_bounds_bond_mask.len(), 0);
    }
}
