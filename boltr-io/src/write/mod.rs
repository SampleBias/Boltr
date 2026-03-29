//! Prediction output writers — layout in [prediction.md](../../../boltz-reference/docs/prediction.md).
//! [`boltz2.py` `predict_step`](../../../boltz-reference/src/boltz/model/models/boltz2.py) shows tensors
//! passed to the upstream Lightning writer (not vendored here).
//!
//! This crate does **not** run Lightning callbacks; it provides **file layouts**, serde types,
//! **structure serialization**, and **confidence `.npz`** helpers for CLI and backend integration.

pub mod affinity_writer;
pub mod mmcif;
pub mod pdb;
pub mod prediction_npz;
pub mod writer;

pub use affinity_writer::{write_affinity_json, AffinitySummary};
pub use mmcif::structure_v2_to_mmcif;
pub use pdb::structure_v2_to_pdb;
pub use prediction_npz::{write_pae_npz_path, write_pde_npz_path, write_plddt_npz_path};
pub use writer::{
    confidence_json_filename, pae_npz_filename, pde_npz_filename, plddt_npz_filename,
    write_confidence_json, ConfidenceSummary, PredictionFileNames,
};
