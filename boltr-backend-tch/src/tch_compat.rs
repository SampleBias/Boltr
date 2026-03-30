//! Helpers aligned with **tch 0.16** (`LayerNorm` via `layer_norm` + `LayerNormConfig`, etc.).

use std::borrow::Borrow;
use tch::nn::{layer_norm, LayerNorm, LayerNormConfig, Module, Path};
use tch::Tensor;

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

/// Matches PyTorch `LayerNorm(dim, bias=False)`: VarStore holds only `weight` (no `bias`).
/// `tch::nn::layer_norm` always registers weight+bias when affine; we cannot construct
/// `LayerNorm` with `bs: None` from outside the `tch` crate, so this thin `Module` is used
/// for AdaLN `s_norm` and any other weight-only LayerNorm.
#[derive(Debug)]
pub(crate) struct LayerNormWeightOnly {
    weight: Tensor,
    normalized_shape: Vec<i64>,
    eps: f64,
}

impl LayerNormWeightOnly {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, dim: i64) -> Self {
        let vs = path.borrow();
        let weight = vs.var("weight", &[dim], tch::nn::Init::Const(1.));
        Self {
            weight,
            normalized_shape: vec![dim],
            eps: dim as f64 * 1e-5,
        }
    }
}

impl Module for LayerNormWeightOnly {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::layer_norm(
            xs,
            self.normalized_shape.as_slice(),
            Some(&self.weight),
            None,
            self.eps,
            true,
        )
    }
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
