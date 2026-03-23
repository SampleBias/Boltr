//! Transition layer (two-layer MLP with SwiGLU)
//!
//! Reference: boltz-reference/src/boltz/model/layers/transition.py

use crate::tch_compat::layer_norm_1d;
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Tensor};

/// Transition layer (two-layer MLP with SwiGLU activation)
///
/// This implements a transition block with LayerNorm followed by
/// a two-layer MLP with SwiGLU activation:
///   - norm(x)
///   - silu(fc1(x)) * fc2(x)
///   - fc3(result)
pub struct Transition {
    norm: tch::nn::LayerNorm,
    fc1: tch::nn::Linear,
    fc2: tch::nn::Linear,
    fc3: tch::nn::Linear,
    hidden: i64,
}

impl Transition {
    /// Create a new Transition layer
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for this transition block
    /// * `dim` - Input dimension
    /// * `hidden` - Hidden dimension (default: dim * 4)
    /// * `out_dim` - Output dimension (default: dim)
    /// * `device` - Computation device
    pub fn new<'a>(
        path: Path<'a>,
        dim: i64,
        hidden: Option<i64>,
        out_dim: Option<i64>,
        _device: Device,
    ) -> Self {
        let hidden = hidden.unwrap_or(dim * 4);
        let out_dim = out_dim.unwrap_or(dim);

        let norm = layer_norm_1d(path.sub("norm"), dim);

        let fc1 = linear(
            path.sub("fc1"),
            dim,
            hidden,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let fc2 = linear(
            path.sub("fc2"),
            dim,
            hidden,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let fc3 = linear(
            path.sub("fc3"),
            hidden,
            out_dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            norm,
            fc1,
            fc2,
            fc3,
            hidden,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., D]
    /// * `_chunk_size` - Placeholder for chunked computation (not yet implemented)
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., out_dim]
    pub fn forward(&self, x: &Tensor, _chunk_size: Option<i64>) -> Tensor {
        // Apply LayerNorm
        let x_normed = self.norm.forward(x);

        // SiLU activation on fc1
        let fc1_out = self.fc1.forward(&x_normed);
        let silu_out = fc1_out.silu();

        // Gate with fc2
        let fc2_out = self.fc2.forward(&x_normed);
        let gated = silu_out * fc2_out;

        // Final projection
        self.fc3.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;
    use tch::nn::VarStore;

    #[test]
    fn test_transition_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let dim = 128;
        let hidden = 512;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = Transition::new(vs.root(), dim, Some(hidden), None, device);

        let x = Tensor::randn(&[batch_size, seq_len, dim], (Kind::Float, device));

        let output = layer.forward(&x, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, dim]);
    }

    #[test]
    fn test_transition_with_out_dim() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let dim = 128;
        let out_dim = 256;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = Transition::new(vs.root(), dim, None, Some(out_dim), device);

        let x = Tensor::randn(&[batch_size, seq_len, dim], (Kind::Float, device));

        let output = layer.forward(&x, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, out_dim]);
    }
}
