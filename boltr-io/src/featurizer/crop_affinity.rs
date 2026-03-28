//! Affinity cropping for Boltz2 inference.
//!
//! Matches Python [`boltz.data.crop.affinity.AffinityCropper`](../../boltz-reference/src/boltz/data/crop/affinity.py):
//! crop tokenized structure around affinity ligand tokens subject to `max_tokens_protein` /
//! `max_atoms`.
//!
//! **Current behavior:** identity (no crop). The full Python routine depends on token center
//! distances, chain expansion, and bond filtering; call sites can rely on the stable API while a
//! parity implementation is added.

use rand::Rng;

use crate::tokenize::boltz2::{TokenBondV2, TokenData};

/// Main-chain tokenization slice used for affinity cropping (no template token maps).
#[derive(Clone, Debug, PartialEq)]
pub struct AffinityTokenized {
    pub tokens: Vec<TokenData>,
    pub bonds: Vec<TokenBondV2>,
}

/// Affinity cropper for Boltz2 affinity inference (Python `AffinityCropper`).
#[derive(Debug, Clone)]
pub struct AffinityCropper {
    /// Maximum number of protein tokens to keep.
    pub max_tokens_protein: usize,
    /// Maximum number of atoms to keep (`None` = unlimited).
    pub max_atoms: Option<usize>,
}

impl Default for AffinityCropper {
    fn default() -> Self {
        Self {
            max_tokens_protein: 200,
            max_atoms: None,
        }
    }
}

impl AffinityCropper {
    #[must_use]
    pub fn new(max_tokens_protein: usize, max_atoms: Option<usize>) -> Self {
        Self {
            max_tokens_protein,
            max_atoms,
        }
    }

    /// Crop tokenized data toward affinity ligand neighborhoods.
    ///
    /// **No-op:** returns `data.clone()`. When implemented, `max_tokens` bounds the crop alongside
    /// [`Self::max_tokens_protein`] and [`Self::max_atoms`].
    #[must_use]
    pub fn crop(
        &self,
        data: &AffinityTokenized,
        max_tokens: usize,
        _rng: &mut impl Rng,
    ) -> AffinityTokenized {
        let _ = (
            self.max_tokens_protein,
            self.max_atoms,
            max_tokens,
        );
        data.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn stub_cropper_returns_clone() {
        let cropper = AffinityCropper::default();
        let data = AffinityTokenized {
            tokens: vec![],
            bonds: vec![],
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = cropper.crop(&data, 200, &mut rng);
        assert_eq!(out, data);
    }

    #[test]
    fn default_limits() {
        let cropper = AffinityCropper::default();
        assert_eq!(cropper.max_tokens_protein, 200);
        assert_eq!(cropper.max_atoms, None);
    }

    #[test]
    fn custom_limits() {
        let cropper = AffinityCropper::new(150, Some(1000));
        assert_eq!(cropper.max_tokens_protein, 150);
        assert_eq!(cropper.max_atoms, Some(1000));
    }
}
