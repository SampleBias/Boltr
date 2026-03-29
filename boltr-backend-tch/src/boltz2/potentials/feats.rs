//! Optional batch tensors for inference potentials (`feats` dict in Boltz collate).
//!
//! Keys mirror `boltz-reference/src/boltz/model/potentials/potentials.py` `compute_args`.

use tch::Tensor;

/// Subset of collate `feats` required by `get_potentials(..., boltz2=True)`.
///
/// Batch dimension is the leading dimension of each tensor (`[B, …]`). Callers may pass only
/// the tensors they have; missing entries skip the corresponding potential branches (empty index).
#[derive(Debug, Default, Clone)]
pub struct PotentialBatchFeats<'a> {
    pub atom_to_token: Option<&'a Tensor>,
    pub asym_id: Option<&'a Tensor>,
    pub atom_pad_mask: Option<&'a Tensor>,
    pub ref_element: Option<&'a Tensor>,
    pub token_index: Option<&'a Tensor>,
    pub token_to_rep_atom: Option<&'a Tensor>,
    pub symmetric_chain_index: Option<&'a Tensor>,
    pub connected_chain_index: Option<&'a Tensor>,
    pub connected_atom_index: Option<&'a Tensor>,
    pub rdkit_bounds_index: Option<&'a Tensor>,
    pub rdkit_lower_bounds: Option<&'a Tensor>,
    pub rdkit_upper_bounds: Option<&'a Tensor>,
    pub rdkit_bounds_bond_mask: Option<&'a Tensor>,
    pub rdkit_bounds_angle_mask: Option<&'a Tensor>,
    pub stereo_bond_index: Option<&'a Tensor>,
    pub stereo_bond_orientations: Option<&'a Tensor>,
    pub chiral_atom_index: Option<&'a Tensor>,
    pub chiral_atom_orientations: Option<&'a Tensor>,
    pub planar_bond_index: Option<&'a Tensor>,
    pub template_mask_cb: Option<&'a Tensor>,
    pub template_force: Option<&'a Tensor>,
    pub template_cb: Option<&'a Tensor>,
    pub template_force_threshold: Option<&'a Tensor>,
    pub contact_pair_index: Option<&'a Tensor>,
    pub contact_union_index: Option<&'a Tensor>,
    pub contact_negation_mask: Option<&'a Tensor>,
    pub contact_thresholds: Option<&'a Tensor>,
}
