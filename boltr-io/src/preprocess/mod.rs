//! Preprocess bundle generation for Boltz-compatible inference ([`crate::inference_dataset::load_input`]).
//!
//! - [`bundle`]: copy artifacts from a Boltz staging directory into a flat layout.
//! - [`native`]: minimal protein-only bundle without Python Boltz (placeholder coordinates).

pub mod bundle;
pub mod native;

pub use bundle::{copy_flat_preprocess_bundle, find_boltz_manifest_path};
pub use native::{validate_native_eligible, write_native_preprocess_bundle, NativePreprocessError};
