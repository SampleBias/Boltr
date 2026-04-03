//! Boltz2 model graph (Rust). Matches Python layout in
//! `boltz-reference/src/boltz/model/models/boltz2.py` when `use_kernels=false`.
//!
//! Submodules are introduced incrementally; trunk holds MSA + pairformer + template hooks,
//! `diffusion` the atom score / sampler stack, etc.

pub mod affinity;
pub mod atom_window_keys;
pub mod confidence;
mod confidence_utils;
pub mod contact_conditioning;
pub mod diffusion;
pub mod diffusion_conditioning;
pub mod diffusion_geometry;
pub mod distogram;
pub mod encoders;
pub mod featurizer_bridge;
pub mod input_embedder;
pub mod model;
pub mod msa_module;
pub mod potentials;
pub mod relative_position;
pub mod steering;
pub mod template_module;
pub mod transformers;
pub mod trunk;

pub use affinity::{
    apply_affinity_mw_correction, AffinityHead, AffinityModule, AffinityModuleConfig,
    AffinityOutput, AFFINITY_MW_BIAS, AFFINITY_MW_COEF, AFFINITY_MW_MODEL_COEF,
};
pub use confidence::{ConfidenceModule, ConfidenceModuleConfig, ConfidenceOutput, ConfidenceV2};
pub use contact_conditioning::{
    ContactConditioning, ContactFeatures, CONTACT_CONDITIONING_CHANNELS,
};
pub use diffusion::{AtomDiffusion, AtomDiffusionConfig, DiffusionModule, DiffusionSampleOutput};
pub use diffusion_conditioning::{DiffusionConditioning, DiffusionConditioningOutput};
pub use distogram::{BFactorModule, DistogramModule};
pub use encoders::{AtomEncoderBatchFeats, AtomEncoderFlags};
pub use featurizer_bridge::zeros_atom_attention_out;
pub use input_embedder::{
    AtomEncoderPlaceholder, InputEmbedder, InputEmbedderFeats, BOLTZ_MSA_PROFILE_IN,
    BOLTZ_NUM_TOKENS,
};
pub use model::{
    Boltz2DiffusionArgs, Boltz2Model, PredictStepFeats, PredictStepOutput, BOND_TYPE_EMBEDDING_NUM,
};
pub use msa_module::{MsaFeatures, MsaModule};
pub use potentials::{get_potentials_boltz2, Potential, PotentialBatchFeats};
pub use relative_position::{RelPosFeatures, RelativePositionEncoder};
pub use steering::SteeringParams;
pub use template_module::{TemplateFeatures, TemplateModule, TemplateV2Module};
pub use trunk::TrunkV2;
