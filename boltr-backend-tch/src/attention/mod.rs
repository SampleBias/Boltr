//! Attention implementations for Boltz2 model
//!
//! This module contains various attention mechanisms used in the Boltz2 model.

#[cfg(feature = "tch-backend")]
pub mod pair_bias;

#[cfg(feature = "tch-backend")]
pub use pair_bias::AttentionPairBiasV2;
