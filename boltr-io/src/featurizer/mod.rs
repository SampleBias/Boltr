//! Featurizer ports (`data/feature/featurizerv2.py`). Incremental §4.4 implementation.

//!
//! `process_atom_features` is now implemented with full parity against Python's `process_atom_features`
//! for standard amino acids and nucleic acids. See [`process_atom_features`] module.

pub mod dummy_templates;
pub mod msa_pairing;
pub mod process_atom_features;
pub mod process_msa_features;
pub mod process_template_features;
pub mod process_token_features;
pub mod token;

#[cfg(test)]
mod atom_features_golden;
#[cfg(test)]
mod msa_features_golden;
#[cfg(test)]
mod token_features_golden;

pub use dummy_templates::{
    dummy_templates_as_feature_batch, load_dummy_templates_features, DummyTemplateTensors,
};
pub use process_atom_features::{
    inference_ensemble_features, process_atom_features, AtomFeatureConfig, AtomFeatureTensors,
    AtomRefData, AtomRefDataProvider, EnsembleFeatures, StandardAminoAcidRefData, ZeroAtomRefData,
    ALA_STANDARD_HEAVY_ATOM_COUNT, ATOMS_PER_WINDOW_QUERIES, ATOM_FEATURE_KEYS_ALA,
    ATOM_NAME_VOCAB_SIZE, DEFAULT_MAX_DIST, DEFAULT_MIN_DIST, DEFAULT_NUM_BINS,
    NUM_BACKBONE_FEAT_CLASSES,
};
pub use process_msa_features::{process_msa_features, MsaFeatureTensors};
pub use process_template_features::{
    pad_template_tdim, process_template_features, stack_template_feature_rows, TemplateAlignment,
};
pub use process_token_features::{
    process_token_features, TokenFeatureTensors, CONTACT_CONDITIONING_NUM_CLASSES,
};
pub use token::{ala_tokenized_smoke, token_feature_key_names};
