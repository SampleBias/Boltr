//! Neural network layer implementations
//!
//! This module contains the core layer implementations for the Boltz2 model.

#[cfg(feature = "tch-backend")]
pub mod outer_product_mean;
#[cfg(feature = "tch-backend")]
pub mod pairformer;
#[cfg(feature = "tch-backend")]
pub mod transition;
#[cfg(feature = "tch-backend")]
pub mod triangular_attention;
#[cfg(feature = "tch-backend")]
pub mod triangular_mult;

#[cfg(feature = "tch-backend")]
pub use outer_product_mean::OuterProductMean;
#[cfg(feature = "tch-backend")]
pub use pairformer::{PairformerLayer, PairformerModule};
#[cfg(feature = "tch-backend")]
pub use transition::Transition;
#[cfg(feature = "tch-backend")]
pub use triangular_attention::{TriangleAttention, TriangleAttentionStartingNode};
#[cfg(feature = "tch-backend")]
pub use triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};
