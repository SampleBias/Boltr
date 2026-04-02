//! Multi-conformer ensemble indices for inference (Boltz `ensemble_ref_idxs`).
//!
//! When `num_ensembles > 1` and the structure has multiple conformers in [`StructureV2Tables::ensemble`],
//! use [`inference_multi_conformer_features`] to reference up to five conformers (indices `0..min(5, n)`).
//! For single-conformer behavior, use [`super::process_ensemble_features::inference_ensemble_features`].

use crate::featurizer::process_ensemble_features::EnsembleFeatures;
use crate::structure_v2::StructureV2Tables;

/// Default cap on how many ensemble conformers to reference at once (Boltz-style multi-conformer).
pub const INFERENCE_MULTI_CONFORMER_MAX: usize = 5;

/// Inference helper: reference up to [`INFERENCE_MULTI_CONFORMER_MAX`] conformers, in order.
///
/// Indices are always valid for `structure`: `0..min(INFERENCE_MULTI_CONFORMER_MAX, n)` where
/// `n` is [`StructureV2Tables::num_ensemble_conformers`].
#[must_use]
pub fn inference_multi_conformer_features(structure: &StructureV2Tables) -> EnsembleFeatures {
    let n = structure.num_ensemble_conformers();
    let take = n.min(INFERENCE_MULTI_CONFORMER_MAX);
    EnsembleFeatures {
        ensemble_ref_idxs: (0..take).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;

    #[test]
    fn inference_multi_conformer_matches_ensemble_count() {
        let s = structure_v2_single_ala();
        let e = inference_multi_conformer_features(&s);
        let n = s.num_ensemble_conformers();
        assert_eq!(
            e.ensemble_ref_idxs.len(),
            n.min(INFERENCE_MULTI_CONFORMER_MAX)
        );
        assert!(e.ensemble_ref_idxs.iter().all(|&i| i < n));
    }

    #[test]
    fn indices_are_sequential_from_zero() {
        let s = structure_v2_single_ala();
        let e = inference_multi_conformer_features(&s);
        let expected: Vec<usize> = (0..s
            .num_ensemble_conformers()
            .min(INFERENCE_MULTI_CONFORMER_MAX))
            .collect();
        assert_eq!(e.ensemble_ref_idxs, expected);
    }
}
