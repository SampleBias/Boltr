//! Boltz2 model graph (Rust). Matches Python layout in
//! `boltz-reference/src/boltz/model/models/boltz2.py` when `use_kernels=false`.
//!
//! Submodules are introduced incrementally; trunk holds MSA + pairformer + template hooks,
//! `diffusion` the atom score / sampler stack, etc.

pub mod affinity;
pub mod confidence;
pub mod diffusion;
pub mod model;
pub mod trunk;

pub use model::Boltz2Model;
pub use trunk::TrunkV2;
