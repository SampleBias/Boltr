// Boltr - Rust Native Boltz Implementation
// Tensor/PyTorch Backend using tch-rs
//
// This crate provides the core inference backend for the Boltz model,
// implementing the neural network architecture and tensor operations.

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

#[cfg(not(feature = "tch-backend"))]
pub mod attention;
#[cfg(not(feature = "tch-backend"))]
pub mod equivariance;
#[cfg(not(feature = "tch-backend"))]
pub mod layers;
#[cfg(not(feature = "tch-backend"))]
pub mod model;

#[cfg(feature = "tch-backend")]
pub use boltz2::Boltz2Model;
#[cfg(feature = "tch-backend")]
pub use checkpoint::{list_safetensor_names, load_tensor_from_safetensors};
#[cfg(feature = "tch-backend")]
pub use device::{cuda_is_available, parse_device_spec};
