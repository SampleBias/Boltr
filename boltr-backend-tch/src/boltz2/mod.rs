//! Boltz2 model graph (Rust). Matches Python layout in
//! `boltz-reference/src/boltz/model/models/boltz2.py` when `use_kernels=false`.
//!
//! Submodules are introduced incrementally; trunk holds MSA + pairformer + template hooks,
//! `diffusion` the atom score / sampler stack, etc.

pub mod affinity;
pub mod confidence;
pub mod contact_conditioning;
pub mod diffusion;
pub mod input_embedder;
pub mod model;
pub mod relative_position;
pub mod trunk;

pub use contact_conditioning::{
    ContactConditioning, ContactFeatures, CONTACT_CONDITIONING_CHANNELS,
};
pub use input_embedder::{InputEmbedder, BOLTZ_MSA_PROFILE_IN, BOLTZ_NUM_TOKENS};
pub use model::{Boltz2Model, BOND_TYPE_EMBEDDING_NUM};
pub use relative_position::{RelPosFeatures, RelativePositionEncoder};
pub use trunk::TrunkV2;
