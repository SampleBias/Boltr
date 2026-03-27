//! Featurizer ports (`data/feature/featurizerv2.py`). Incremental §4.4 implementation.

pub mod dummy_templates;
pub mod process_token_features;
#[cfg(test)]
mod token_features_golden;
pub mod token;

pub use dummy_templates::{load_dummy_templates_features, DummyTemplateTensors};
pub use process_token_features::{
    process_token_features, TokenFeatureTensors, CONTACT_CONDITIONING_NUM_CLASSES,
};
pub use token::{ala_tokenized_smoke, token_feature_key_names};
