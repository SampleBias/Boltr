//! Neural network layer implementations
//!
//! This module contains the core layer implementations for the Boltz2 model.

#[cfg(feature = "tch-backend")]
pub mod outer_product_mean;
#[cfg(feature = "tch-backend")]
pub mod outer_product_mean_msa;
#[cfg(feature = "tch-backend")]
pub mod pair_weighted_averaging;
#[cfg(feature = "tch-backend")]
pub mod pairformer;
#[cfg(feature = "tch-backend")]
pub mod pairformer_no_seq;
#[cfg(feature = "tch-backend")]
pub mod transition;
#[cfg(feature = "tch-backend")]
pub mod triangular_attention;
#[cfg(feature = "tch-backend")]
pub mod triangular_mult;

#[cfg(feature = "tch-backend")]
pub use outer_product_mean::OuterProductMean;
#[cfg(feature = "tch-backend")]
pub use outer_product_mean_msa::OuterProductMeanMsa;
#[cfg(feature = "tch-backend")]
pub use pair_weighted_averaging::PairWeightedAveraging;
#[cfg(feature = "tch-backend")]
pub use pairformer::{PairformerLayer, PairformerModule};
#[cfg(feature = "tch-backend")]
pub use pairformer_no_seq::PairformerNoSeqLayer;
#[cfg(feature = "tch-backend")]
pub use transition::Transition;
#[cfg(feature = "tch-backend")]
pub use triangular_attention::{TriangleAttention, TriangleAttentionStartingNode};
#[cfg(feature = "tch-backend")]
pub use triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};
