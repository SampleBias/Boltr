//! Port of `process_atom_features` from [`featurizerv2.py`](../../../boltz-reference/src/boltz/data/feature/featurizerv2.py).
//!
//! ## Status
//!
//! **Not implemented in Rust.** Python resolves reference chemistry via RDKit `Mol` per residue
//! (`molecules[token["res_name"]]`) and conformer sampling; reproducing that requires CCD/molecule
//! loading or a subprocess strategy (see [`TODO.md`](../../../TODO.md) §2b phase 4, §4.1).
//!
//! ## Golden (parity anchor)
//!
//! Authoritative tensor dict for single-token ALA + canonical `mols/*.pkl`:
//! `boltr-io/tests/fixtures/collate_golden/atom_features_ala_golden.safetensors`.
//! Regenerate:
//!
//! ```text
//! # extract Boltz mols.tar so .../mols/ALA.pkl exists
//! export PYTHONPATH=boltz-reference/src
//! python3 scripts/dump_atom_features_golden.py --mol-dir /path/to/mols
//! ```
//!
//! Schema check: [`atom_features_golden`](crate::featurizer::atom_features_golden) tests.
//!
//! ## Incremental port
//!
//! Prefer **golden-first** subset tests (`allclose` on increasing keys) as Rust gains
//! structure-only or CCD-backed paths. Full parity gates on the same `mols` layout as Boltz.

/// Tensor names in [`atom_features_ala_golden.safetensors`](../../tests/fixtures/collate_golden/)
/// (single-token ALA, padded atom table).
/// Heavy-atom count for a single standard ALA residue (backbone + CB) in the canonical mol layout.
pub const ALA_STANDARD_HEAVY_ATOM_COUNT: usize = 5;

pub const ATOM_FEATURE_KEYS_ALA: &[&str] = &[
    "atom_backbone_feat",
    "atom_pad_mask",
    "atom_resolved_mask",
    "atom_to_token",
    "bfactor",
    "coords",
    "disto_coords_ensemble",
    "disto_target",
    "plddt",
    "r_set_to_rep_atom",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_chirality",
    "ref_element",
    "ref_pos",
    "ref_space_uid",
    "token_to_center_atom",
    "token_to_rep_atom",
];

#[cfg(test)]
mod tests {
    use super::ATOM_FEATURE_KEYS_ALA;

    #[test]
    fn atom_feature_keys_count_matches_golden_file_note() {
        assert_eq!(ATOM_FEATURE_KEYS_ALA.len(), 18);
    }
}
