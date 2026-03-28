//! Affinity cropping for Boltz2 inference.
//!
//! Matches Python `boltz.data.crop.affinity.AffinityCropper` which crops
//! tokenized data to a maximum number of tokens/atoms around affinity ligands.
//!
//! This is a **stub implementation** that documents the expected interface.
//! Full implementation would require:
//! - Token-level distance calculations
//! - Spatial neighbor selection (neighborhood_size parameter)
//! - Chain-level cropping with neighborhood expansion
//! - Bond filtering for cropped tokens
//! - Max tokens/atoms constraints
//!
//! For basic Boltz inference without affinity mode, cropping is not needed.

use crate::tokenize::boltz2::Tokenized;

/// Affinity cropper for Boltz2 affinity inference.
///
/// Matches Python `boltz.data.crop.affinity.AffinityCropper`.
///
/// # Python Behavior Summary
///
/// **When to use:** Only when `affinity=True` in inference
/// **Purpose:** Crop large structures around affinity ligand tokens
/// **Limits:** `max_tokens_protein` (default 200) and `max_atoms`
/// **Algorithm:**
/// 1. Find affinity ligand tokens (`mol_type == NONPOLYMER`)
/// 2. Compute distances from ligand to all protein tokens
/// 3. Sort by distance, select nearest `neighborhood_size` tokens
/// 4. Expand neighborhood (optional spatial expansion)
/// 5. Add whole chains if ≤ `neighborhood_size`
/// 6. Iterate until max tokens/atoms reached
///
/// # Complexity Notes
///
/// The full Python implementation is **non-trivial** (120+ lines):
/// - Distance calculations using center coordinates
/// - Numpy advanced indexing with integer arrays
/// - Chain token filtering by asym_id
/// - Multi-stage expansion logic
/// - Bond filtering for only cropped tokens
///
/// # Implementation Strategy
///
/// **Current (Stub):** Return input unchanged with warning
/// **Future (Full):** Implement when concrete use case emerges
///
/// # References
///
/// - Python: `boltz-reference/src/boltz/data/crop/affinity.py` (lines 1-300)
/// - Python: `boltz-reference/src/boltz/data/module/inferencev2.py` (lines 238-247)
/// - Python: `boltz-reference/src/boltz/data/feature/featurizerv2.py` (tokenized structure)
#[derive(Debug, Clone)]
pub struct AffinityCropper {
    /// Maximum number of protein tokens to keep.
    pub max_tokens_protein: usize,

    /// Maximum number of atoms to keep.
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
    /// Create a new affinity cropper with specified limits.
    ///
    /// # Parameters
    ///
    /// * `max_tokens_protein` - Maximum protein tokens to keep (default: 200)
    /// * `max_atoms` - Maximum atoms to keep (default: None = unlimited)
    #[must_use]
    pub fn new(max_tokens_protein: usize, max_atoms: Option<usize>) -> Self {
        Self {
            max_tokens_protein,
            max_atoms,
        }
    }

    /// Crop tokenized data to maximum number of tokens/atoms around affinity ligands.
    ///
    /// **STUB IMPLEMENTATION:** Currently returns input unchanged.
    ///
    /// # Parameters
    ///
    /// * `data` - The tokenized data to crop
    /// * `max_tokens` - The maximum number of tokens to keep
    /// * `random` - Random state (not used in stub, but part of API)
    ///
    /// # Returns
    ///
    /// Cropped tokenized data (or unchanged in stub)
    ///
    /// # Python Reference
    ///
    /// - `crop(data, max_tokens, max_atoms=None, random=np.random.default_rng(42))`
    /// - `inferencev2.py` lines 238-247: instantiation and cropping logic
    ///
    /// # Algorithm (Python)
    ///
    /// 1. Filter to resolved tokens (token_data["resolved_mask"] == 1)
    /// 2. Find ligand tokens (mol_type == NONPOLYMER) with affinity_mask
    /// 3. Compute distances from ligand center to all protein tokens
    /// 4. Select nearest `neighborhood_size` tokens by distance
    /// 5. If ≤ neighborhood_size, add whole chains by asym_id
    /// 6. Repeat 4-5 until max_tokens or max_atoms reached
    /// 7. Filter bonds to only include bonds between cropped tokens
    ///
    /// # Stub Limitations
    ///
    /// - No cropping performed
    /// - Returns input unchanged
    /// - Does not validate affinity_mask or mol_type
    /// - Does not enforce max_tokens or max_atoms limits
    ///
    /// # When to Use
    ///
    /// - NOT: Standard inference (affinity=False)
    /// - YES: Affinity inference (affinity=True)
    /// - NOT: Structures already small (<200 tokens)
    /// - NOT: Ligand information unavailable in Rust
    ///
    /// # Future Implementation
    ///
    /// Full implementation would require:
    /// - Token center coordinate extraction
    /// - Distance matrix computation (O(N*M))
    /// - Chain filtering and asym_id matching
    /// - Neighborhood expansion logic
    /// - Bond filtering for cropped tokens
    /// - Random state handling
    /// - Proper error messages
    ///
    /// # Testing
    ///
    /// Full implementation should test:
    /// - Single ligand cropping
    /// - Multiple ligands (one is nearest, then another)
    /// - Edge cases (no valid tokens, no ligands)
    /// - Max tokens/atoms enforcement
    /// - Random state reproducibility
    #[must_use]
    pub fn crop(&self, data: Tokenized, max_tokens: usize, _random: () -> Tokenized {
        // STUB: Return input unchanged
        // In Python, this would:
        // 1. Find affinity ligand tokens
        // 2. Compute distances from ligand to protein tokens
        // 3. Select nearest neighborhood_size tokens
        // 4. Expand neighborhood (spatial)
        // 5. Add whole chains if needed
        // 6. Repeat until max_tokens reached
        // 7. Filter bonds to cropped tokens

        // TODO: Implement full cropping when use case emerges

        data.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_cropper_does_nothing() {
        let cropper = AffinityCropper::default();
        let tokens = Tokenized {
            tokens: vec![],
            bonds: vec![],
        };

        let result = cropper.crop(&tokens, 200, ());
        assert_eq!(result.tokens, tokens.tokens);
        assert_eq!(result.bonds, tokens.bonds);
    }

    #[test]
    fn default_values() {
        let cropper = AffinityCropper::default();
        assert_eq!(cropper.max_tokens_protein, 200);
        assert!(cropper.max_atoms, None);
    }

    #[test]
    fn custom_limits() {
        let cropper = AffinityCropper::new(150, Some(1000));
        assert_eq!(cropper.max_tokens_protein, 150);
        assert_eq!(cropper.max_atoms, Some(1000));
    }
}
