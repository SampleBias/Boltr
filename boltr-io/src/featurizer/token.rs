//! Token-feature slice toward Python `process_token_features`.
//!
//! Tensor construction lives in [`super::process_token_features`]; this module anchors the
//! contract via key names and a tokenizer smoke path. Golden tests: `token_features_golden`.

use crate::fixtures::structure_v2_single_ala;
use crate::tokenize::boltz2::{tokenize_structure, TokenData};

/// Field names in Python `token_features` dict (`featurizerv2.py` `process_token_features`).
#[must_use]
pub fn token_feature_key_names() -> &'static [&'static str] {
    &[
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "mol_type",
        "res_type",
        "disto_center",
        "token_bonds",
        "type_bonds",
        "token_pad_mask",
        "token_resolved_mask",
        "token_disto_mask",
        "contact_conditioning",
        "contact_threshold",
        "method_feature",
        "modified",
        "cyclic_period",
        "affinity_token_mask",
    ]
}

/// Single-chain ALA tokens for unit tests / collate shape probes (`N = 1` token).
#[must_use]
pub fn ala_tokenized_smoke() -> Vec<TokenData> {
    let s = structure_v2_single_ala();
    let (t, _) = tokenize_structure(&s, None);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ala_smoke_has_one_token() {
        let t = ala_tokenized_smoke();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].res_name, "ALA");
    }
}
