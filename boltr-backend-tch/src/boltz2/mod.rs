//! Boltz2 model graph (Rust). Matches Python layout in
//! `boltz-reference/src/boltz/model/models/boltz2.py` when `use_kernels=false`.
//!
//! Submodules are introduced incrementally; trunk holds MSA + pairformer + template hooks,
//! `diffusion` the atom score / sampler stack, etc.

pub mod affinity;
pub mod confidence;
mod confidence_utils;
pub mod contact_conditioning;
pub mod diffusion;
pub mod diffusion_conditioning;
pub mod distogram;
pub mod encoders;
pub mod featurizer_bridge;
pub mod input_embedder;
pub mod model;
pub mod msa_module;
pub mod relative_position;
pub mod template_module;
pub mod transformers;
pub mod trunk;

pub use affinity::{AffinityHead, AffinityModule};
pub use confidence::{ConfidenceModule, ConfidenceModuleConfig, ConfidenceOutput, ConfidenceV2};
pub use contact_conditioning::{
    ContactConditioning, ContactFeatures, CONTACT_CONDITIONING_CHANNELS,
};
pub use diffusion::{AtomDiffusion, AtomDiffusionConfig, DiffusionModule, DiffusionSampleOutput};
pub use diffusion_conditioning::{DiffusionConditioning, DiffusionConditioningOutput};
pub use distogram::{BFactorModule, DistogramModule};
pub use featurizer_bridge::zeros_atom_attention_out;
pub use input_embedder::{
    AtomEncoderPlaceholder, InputEmbedder, BOLTZ_MSA_PROFILE_IN, BOLTZ_NUM_TOKENS,
};
pub use model::{Boltz2DiffusionArgs, Boltz2Model, PredictStepFeats, PredictStepOutput, BOND_TYPE_EMBEDDING_NUM};
pub use msa_module::{MsaFeatures, MsaModule};
pub use relative_position::{RelPosFeatures, RelativePositionEncoder};
pub use template_module::{TemplateFeatures, TemplateV2Module, TemplateModule};
pub use trunk::TrunkV2;
