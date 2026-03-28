//! Prediction output writers — Rust counterparts to Boltz [`data/write/writer.py`](../../../boltz-reference/src/boltz/data/write/writer.py),
//! [`mmcif.py`](../../../boltz-reference/src/boltz/data/write/mmcif.py), [`pdb.py`](../../../boltz-reference/src/boltz/data/write/pdb.py).
//!
//! This crate does **not** run Lightning callbacks; it provides **file layouts**, serde types, and
//! **structure serialization** for CLI and backend integration.

pub mod affinity_writer;
pub mod mmcif;
pub mod pdb;
pub mod writer;

pub use affinity_writer::{AffinitySummary, write_affinity_json};
pub use mmcif::structure_v2_to_mmcif;
pub use pdb::structure_v2_to_pdb;
pub use writer::{
    confidence_json_filename, write_confidence_json, ConfidenceSummary, PredictionFileNames,
};
