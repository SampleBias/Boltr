//! Featurizer ports (`data/feature/featurizerv2.py`). Incremental §4.4 implementation

pub mod dummy_templates;
pub mod msa_pairing;
pub mod process_atom_features;
#[cfg(test)]
pub mod atom_features_golden;
pub mod process_msa_features;
pub mod process_token_features;
pub mod token;
#[cfg(test)]
mod msa_features_golden;
#[cfg(test)]
mod token_features_golden;

pub use dummy_templates::{
    dummy_templates_as_feature_batch, load_dummy_templates_features, DummyTemplateTensors,
};
pub use process_atom_features::{
    inference_ensemble_features, process_atom_features, AtomFeatureConfig, AtomFeatureTensors,
    AtomRefDataProvider, StandardAminoAcidRefData, ALA_STANDARD_HEAVY_ATOM_COUNT, ATOM_FEATURE_KEYS_ALA,
};
pub use process_msa_features::{process_msa_features, MsaFeatureTensors};
pub use process_token_features::{
    process_token_features, TokenFeatureTensors, CONTACT_CONDITIONING_NUM_CLASSES,
};
pub use token::{ala_tokenized_smoke, token_feature_key_names};
