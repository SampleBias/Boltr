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
