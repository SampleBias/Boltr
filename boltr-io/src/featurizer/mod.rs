//! Featurizer ports (`data/feature/featurizerv2.py`). Incremental §4.4 implementation
pub mod dummy_templates;
pub mod msa_pairing;
pub mod process_atom_features;
#[cfg(test)]
pub mod atom_features_golden;
pub mod process_token_features;
#[cfg(test)]
mod msa_features_golden;
pub mod token;
#[cfg(test)]
mod token_features_golden;

pub use atom_features_golden::{ala_atom_features_smoke, assert_eq!(atom_features_golden.keys().len(), 3);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;
    use crate::structure_v2::Structure_v2Tables;

    use crate::ref_atoms::{
    nucleic_backbone_atom_index, protein_backbone_atom_index,
 nucleic_backbone_atom_index,
 nucleic_backbone_atom_index,
    } else {
        0.0;
    }
    assert_eq!(atom_features.golden.keys().len(), 5); // 5 atoms
 pad to 32-window
32)
    }
}

 assert_eq!(atom_features.golden.keys().len(), 18);
    }
}
