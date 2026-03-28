//! Boltz `process_ensemble_features` (`featurizerv2.py`) — sample which structure conformers to use.

use anyhow::ensure;
use rand::seq::index::sample;
use rand::Rng;

use crate::structure_v2::StructureV2Tables;

/// Indices into [`StructureV2Tables::ensemble`] (Boltz `ensemble_ref_idxs`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleFeatures {
    pub ensemble_ref_idxs: Vec<usize>,
}

/// Inference default: single conformer, index `0` (Boltz `fix_single_ensemble=True`).
#[must_use]
pub fn inference_ensemble_features() -> EnsembleFeatures {
    EnsembleFeatures {
        ensemble_ref_idxs: vec![0],
    }
}

/// Sample ensemble conformer indices (`process_ensemble_features` in Boltz).
///
/// `num_ensembles` must be positive. When `fix_single_ensemble`, requires `num_ensembles == 1`.
#[allow(clippy::cast_possible_wrap)]
pub fn process_ensemble_features(
    structure: &StructureV2Tables,
    rng: &mut impl Rng,
    num_ensembles: usize,
    ensemble_sample_replacement: bool,
    fix_single_ensemble: bool,
) -> anyhow::Result<EnsembleFeatures> {
    ensure!(num_ensembles > 0, "num_ensembles must be > 0");
    let s_ensemble_num = structure.num_ensemble_conformers();

    let ensemble_ref_idxs: Vec<usize> = if fix_single_ensemble {
        ensure!(
            num_ensembles == 1,
            "fix_single_ensemble requires num_ensembles == 1"
        );
        vec![0]
    } else if ensemble_sample_replacement {
        (0..num_ensembles)
            .map(|_| rng.gen_range(0..s_ensemble_num))
            .collect()
    } else if s_ensemble_num < num_ensembles {
        (0..s_ensemble_num).collect()
    } else {
        sample(rng, s_ensemble_num, num_ensembles).into_iter().collect()
    };

    Ok(EnsembleFeatures {
        ensemble_ref_idxs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use rand::SeedableRng;

    #[test]
    fn fix_single_is_index_zero() {
        let s = structure_v2_single_ala();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let e = process_ensemble_features(&s, &mut rng, 1, false, true).unwrap();
        assert_eq!(e.ensemble_ref_idxs, vec![0]);
    }

    #[test]
    fn replacement_deterministic_with_seed() {
        let s = structure_v2_single_ala();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let e = process_ensemble_features(&s, &mut rng, 4, true, false).unwrap();
        assert_eq!(e.ensemble_ref_idxs.len(), 4);
        assert!(e.ensemble_ref_idxs.iter().all(|&i| i < s.num_ensemble_conformers()));
    }
}
