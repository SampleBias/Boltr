// Boltr - Rust Native Boltz Implementation
// Tensor/PyTorch Backend using tch-rs
//
// This crate provides the core inference backend for the Boltz model,
// implementing the neural network architecture and tensor operations.

pub mod model;
pub mod layers;
pub mod attention;
pub mod equivariance;

pub use model::BoltzModel;
