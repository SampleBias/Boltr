//! Preprocess bundle generation for Boltz-compatible inference ([`crate::inference_dataset::load_input`]).
//!
//! - [`bundle`]: copy artifacts from a Boltz staging directory into a flat layout.
//! - [`native`]: minimal protein-only bundle without Python Boltz (placeholder coordinates).

pub mod bundle;
pub mod native;
pub mod paths;

pub use bundle::{copy_flat_preprocess_bundle, find_boltz_manifest_path};
pub use native::{validate_native_eligible, write_native_preprocess_bundle, NativePreprocessError};
pub use paths::{canonical_yaml_parent, copy_msa_a3m_to_output, preprocess_bundle_ready};
