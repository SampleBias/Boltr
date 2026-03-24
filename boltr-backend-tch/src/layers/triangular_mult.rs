//! Triangular multiplication layers (TriangleMultiplicationOutgoing/Incoming)
//!
//! Reference: boltz-reference/src/boltz/model/layers/triangular_mult.py
//! Implements the fallback PyTorch path (use_kernels=False)

use crate::tch_compat::layer_norm_1d;
use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

/// Triangle Multiplication Outgoing layer
///
/// This implements the "outgoing" triangle multiplication operation.
/// The operation computes pairwise updates using the outgoing edges of a triangle.
///
/// For input x of shape [B, N, N, D]:
///   1. Apply LayerNorm and gate projection
///   2. Split into a and b (each shape [B, N, N, D/2])
///   3. Compute einsum("bikd,bjkd->bijd", a, b) to get [B, N, N, D]
///   4. Apply output gating and projection
pub struct TriangleMultiplicationOutgoing {
    norm_in: tch::nn::LayerNorm,
    p_in: tch::nn::Linear,
    g_in: tch::nn::Linear,
    norm_out: tch::nn::LayerNorm,
    p_out: tch::nn::Linear,
    g_out: tch::nn::Linear,
    dim: i64,
}

impl TriangleMultiplicationOutgoing {
    /// Create a new TriangleMultiplicationOutgoing layer
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for this layer
    /// * `dim` - Dimension of the input/output (default 128)
    /// * `device` - Computation device
    pub fn new<'a>(path: Path<'a>, dim: Option<i64>, _device: Device) -> Self {
        let dim = dim.unwrap_or(128);

        let norm_in = layer_norm_1d(path.sub("norm_in"), dim);

        let p_in = linear(
            path.sub("p_in"),
            dim,
            2 * dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let g_in = linear(
            path.sub("g_in"),
            dim,
            2 * dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let norm_out = layer_norm_1d(path.sub("norm_out"), dim);

        let p_out = linear(
            path.sub("p_out"),
            dim,
            dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let g_out = linear(
            path.sub("g_out"),
            dim,
            dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            norm_in,
            p_in,
            g_in,
            norm_out,
            p_out,
            g_out,
            dim,
        }
    }

    /// Forward pass (PyTorch fallback path, not using cuequivariance kernels)
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [B, N, N, D]
    /// * `mask` - Mask tensor of shape [B, N, N]
    /// * `_use_kernels` - Placeholder for kernel usage (always uses fallback in Rust)
    ///
    /// # Returns
    ///
    /// Output tensor of shape [B, N, N, D]
    pub fn forward(&self, x: &Tensor, mask: &Tensor, _use_kernels: bool) -> Tensor {
        // Input gating: D -> 2D
        let x_normed = self.norm_in.forward(x);
        let x_in = x_normed.shallow_clone();

        // Project and gate
        let x_proj = self.p_in.forward(&x_normed);
        let x_gate = self.g_in.forward(&x_normed).sigmoid();
        let x = x_proj * x_gate;

        // Apply mask
        let mask_expanded = mask.unsqueeze(-1);
        let x = x * mask_expanded;

        // Split into a and b (each shape [B, N, N, D/2])
        let chunks = x.chunk(2, -1);
        let a = chunks.get(0).unwrap(); // [B, N, N, D/2]
        let b = chunks.get(1).unwrap(); // [B, N, N, D/2]

        // Cast to float32 for precision
        let a_float = a.to_kind(Kind::Float);
        let b_float = b.to_kind(Kind::Float);

        // Triangular projection: `torch.einsum("bikd,bjkd->bijd", a, b)` (Boltz reference).
        let x_triangular = Tensor::einsum("bikd,bjkd->bijd", &[&a_float, &b_float], None::<i64>);

        // Output gating
        let x_normed_out = self.norm_out.forward(&x_triangular);
        let x_proj_out = self.p_out.forward(&x_normed_out);
        let x_gate_out = self.g_out.forward(&x_in).sigmoid();
        let x = x_proj_out * x_gate_out;

        // Restore original dtype
        x.to_kind(x.kind())
    }
}

/// Triangle Multiplication Incoming layer
///
/// This implements the "incoming" triangle multiplication operation.
/// The operation computes pairwise updates using the incoming edges of a triangle.
///
/// For input x of shape [B, N, N, D]:
///   1. Apply LayerNorm and gate projection
///   2. Split into a and b (each shape [B, N, N, D/2])
///   3. Compute einsum("bkid,bkjd->bijd", a, b) to get [B, N, N, D]
///   4. Apply output gating and projection
pub struct TriangleMultiplicationIncoming {
    norm_in: tch::nn::LayerNorm,
    p_in: tch::nn::Linear,
    g_in: tch::nn::Linear,
    norm_out: tch::nn::LayerNorm,
    p_out: tch::nn::Linear,
    g_out: tch::nn::Linear,
    dim: i64,
}

impl TriangleMultiplicationIncoming {
    /// Create a new TriangleMultiplicationIncoming layer
    ///
    /// # Arguments
    ///
    /// * `path` - VarStore sub-path for this layer
    /// * `dim` - Dimension of the input/output (default 128)
    /// * `device` - Computation device
    pub fn new<'a>(path: Path<'a>, dim: Option<i64>, _device: Device) -> Self {
        let dim = dim.unwrap_or(128);

        let norm_in = layer_norm_1d(path.sub("norm_in"), dim);

        let p_in = linear(
            path.sub("p_in"),
            dim,
            2 * dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let g_in = linear(
            path.sub("g_in"),
            dim,
            2 * dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let norm_out = layer_norm_1d(path.sub("norm_out"), dim);

        let p_out = linear(
            path.sub("p_out"),
            dim,
            dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let g_out = linear(
            path.sub("g_out"),
            dim,
            dim,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            norm_in,
            p_in,
            g_in,
            norm_out,
            p_out,
            g_out,
            dim,
        }
    }

    /// Forward pass (PyTorch fallback path, not using cuequivariance kernels)
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [B, N, N, D]
    /// * `mask` - Mask tensor of shape [B, N, N]
    /// * `_use_kernels` - Placeholder for kernel usage (always uses fallback in Rust)
    ///
    /// # Returns
    ///
    /// Output tensor of shape [B, N, N, D]
    pub fn forward(&self, x: &Tensor, mask: &Tensor, _use_kernels: bool) -> Tensor {
        // Input gating: D -> 2D
        let x_normed = self.norm_in.forward(x);
        let x_in = x_normed.shallow_clone();

        // Project and gate
        let x_proj = self.p_in.forward(&x_normed);
        let x_gate = self.g_in.forward(&x_normed).sigmoid();
        let x = x_proj * x_gate;

        // Apply mask
        let mask_expanded = mask.unsqueeze(-1);
        let x = x * mask_expanded;

        // Split into a and b (each shape [B, N, N, D/2])
        let chunks = x.chunk(2, -1);
        let a = chunks.get(0).unwrap(); // [B, N, N, D/2]
        let b = chunks.get(1).unwrap(); // [B, N, N, D/2]

        // Cast to float32 for precision
        let a_float = a.to_kind(Kind::Float);
        let b_float = b.to_kind(Kind::Float);

        // Triangular projection: `torch.einsum("bkid,bkjd->bijd", a, b)` (Boltz reference).
        let x_triangular = Tensor::einsum("bkid,bkjd->bijd", &[&a_float, &b_float], None::<i64>);

        // Output gating
        let x_normed_out = self.norm_out.forward(&x_triangular);
        let x_proj_out = self.p_out.forward(&x_normed_out);
        let x_gate_out = self.g_out.forward(&x_in).sigmoid();
        let x = x_proj_out * x_gate_out;

        // Restore original dtype
        x.to_kind(x.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_triangular_mult_outgoing_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let dim = 128;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = TriangleMultiplicationOutgoing::new(vs.root(), Some(dim), device);

        let x = Tensor::randn(&[batch_size, seq_len, seq_len, dim], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&x, &mask, false);

        assert_eq!(output.size(), vec![batch_size, seq_len, seq_len, dim]);
    }

    #[test]
    fn test_triangular_mult_incoming_forward() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let dim = 128;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = TriangleMultiplicationIncoming::new(vs.root(), Some(dim), device);

        let x = Tensor::randn(&[batch_size, seq_len, seq_len, dim], (Kind::Float, device));
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        let output = layer.forward(&x, &mask, false);

        assert_eq!(output.size(), vec![batch_size, seq_len, seq_len, dim]);
    }
}
