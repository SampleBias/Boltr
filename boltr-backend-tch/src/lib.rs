// Boltr - Rust Native Boltz Implementation
// Tensor/PyTorch Backend using tch-rs
//
// This crate provides the core inference backend for the Boltz model,
// implementing the neural network architecture and tensor operations.

pub mod boltz_hparams;
pub mod inference_keys;

pub use boltz_hparams::Boltz2Hparams;
pub use inference_keys::{
    partition_safetensors_keys_for_inference, BOLTZ2_INFERENCE_TOP_LEVEL_KEYS,
};

#[cfg(feature = "tch-backend")]
pub mod attention;
#[cfg(feature = "tch-backend")]
pub mod boltz2;
#[cfg(feature = "tch-backend")]
pub mod checkpoint;
#[cfg(feature = "tch-backend")]
pub mod device;
#[cfg(feature = "tch-backend")]
pub mod equivariance;
#[cfg(feature = "tch-backend")]
pub mod layers;
#[cfg(feature = "tch-backend")]
pub mod model;
#[cfg(feature = "tch-backend")]
mod tch_compat;

#[cfg(not(feature = "tch-backend"))]
pub mod attention;
#[cfg(not(feature = "tch-backend"))]
pub mod equivariance;
#[cfg(not(feature = "tch-backend"))]
pub mod layers;
#[cfg(not(feature = "tch-backend"))]
pub mod model;

#[cfg(feature = "tch-backend")]
pub use boltz2::{
    apply_affinity_mw_correction, zeros_atom_attention_out, AffinityHead, AffinityModule,
    AffinityModuleConfig, AffinityOutput, AtomDiffusion, AtomDiffusionConfig,
    AtomEncoderPlaceholder, BFactorModule, Boltz2Model, ConfidenceModule, ConfidenceModuleConfig,
    ConfidenceOutput, ConfidenceV2, ContactConditioning, ContactFeatures, DiffusionConditioning,
    DiffusionConditioningOutput, DiffusionModule, DiffusionSampleOutput, DistogramModule,
    InputEmbedder, MsaFeatures, MsaModule, PredictStepFeats, PredictStepOutput, RelPosFeatures,
    RelativePositionEncoder, TemplateFeatures, TemplateModule, TemplateV2Module, AFFINITY_MW_BIAS,
    AFFINITY_MW_COEF, AFFINITY_MW_MODEL_COEF, BOLTZ_MSA_PROFILE_IN, BOLTZ_NUM_TOKENS,
    BOND_TYPE_EMBEDDING_NUM, CONTACT_CONDITIONING_CHANNELS,
};
#[cfg(feature = "tch-backend")]
pub use checkpoint::{
    list_safetensor_names, load_tensor_from_safetensors, safetensor_names_not_in_var_store,
    var_store_keys_missing_in_safetensors,
};
#[cfg(feature = "tch-backend")]
pub use device::{cuda_is_available, parse_device_spec};
#[cfg(feature = "tch-backend")]
pub use model::BoltzModel;

// Re-export layer implementations
#[cfg(feature = "tch-backend")]
pub use attention::AttentionPairBiasV2;
#[cfg(feature = "tch-backend")]
pub use layers::{PairformerLayer, PairformerModule, PairformerNoSeqModule, Transition};
