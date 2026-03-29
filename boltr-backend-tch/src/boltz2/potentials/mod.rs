//! Inference-time potentials (`boltz-reference/.../potentials/potentials.py`).

mod compute;
pub mod consts;
pub mod feats;
pub mod registry;
pub mod schedules;

pub use feats::PotentialBatchFeats;
pub use registry::{get_potentials_boltz2, Potential};
