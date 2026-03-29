//! Helpers aligned with **tch 0.16** (`LayerNorm` via `layer_norm` + `LayerNormConfig`, etc.).

use std::borrow::Borrow;
use tch::nn::{layer_norm, LayerNorm, LayerNormConfig, Path};

/// LayerNorm over the last dimension: `normalized_shape = [dim]`, `eps = dim * 1e-5`, affine on.
pub(crate) fn layer_norm_1d<'a, P: Borrow<Path<'a>>>(path: P, dim: i64) -> LayerNorm {
    layer_norm(
        path,
        vec![dim],
        LayerNormConfig {
            eps: dim as f64 * 1e-5,
            elementwise_affine: true,
            ..Default::default()
        },
    )
}

/// Helper function to create a linear layer without bias
///
/// This is used for projections in various modules (e.g., template z_proj, a_proj)
pub(crate) fn linear_no_bias<'a, P: Borrow<tch::nn::Path<'a>>>(
    path: P,
    in_dim: i64,
    out_dim: i64,
) -> tch::nn::Linear {
    tch::nn::linear(
        path,
        in_dim,
        out_dim,
        tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    )
}
